import argparse
import os
import time
from collections import OrderedDict
from itertools import cycle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from torch.func import functional_call
except ImportError:
    from torch.nn.utils.stateless import functional_call

from common import (
    FLOW_TARGET_VOTE_THRESHOLD,
    LAST_LAYER_TAKE_ALL,
    ROUTING_UNIT,
    build_cifar100_datasets,
    build_eval_loaders,
    build_model,
    build_split_subsets,
    compute_anchor_loss,
    compute_fixed_flow_masks,
    compute_layerwise_target_experts,
    compute_metrics,
    compute_selection_score,
    format_layer_targets,
    freeze_to_layer_targets,
    get_layer_param_names,
    get_trainable_param_names,
    load_checkpoint_state,
    load_split_indices,
    set_seed,
)


class BezierCurveModel(nn.Module):
    def __init__(self, template_model, start_state, end_state, trainable_names, mask=None):
        super().__init__()
        self.template_model = template_model
        self.template_model.eval()

        self.parameter_names = [name for name, _ in template_model.named_parameters()]
        self.buffer_names = [name for name, _ in template_model.named_buffers()]
        self._safe_to_name = {}

        self.mid_params = nn.ParameterDict()
        self.start_params = {}
        self.end_params = {}
        self.fixed_mid_params = {}
        self.param_masks = {}
        self.mid_buffers = {}
        self.start_buffers = {}
        self.end_buffers = {}

        trainable_names = set(trainable_names)
        self.trainable_names = trainable_names
        mask = mask or {}

        for name in self.parameter_names:
            safe_name = name.replace(".", "__")
            self._safe_to_name[safe_name] = name

            start_tensor = start_state[name].detach().clone()
            end_tensor = end_state[name].detach().clone()
            mid_tensor = 0.5 * start_tensor + 0.5 * end_tensor

            self.start_params[name] = start_tensor
            self.end_params[name] = end_tensor
            self.fixed_mid_params[name] = mid_tensor.clone()

            mask_value = mask.get(name, 1) if name in trainable_names else 0
            if torch.is_tensor(mask_value):
                if mask_value.shape != mid_tensor.shape:
                    raise ValueError(
                        f"Mask shape mismatch for {name}: expected {tuple(mid_tensor.shape)}, "
                        f"got {tuple(mask_value.shape)}"
                    )
                param_mask = mask_value.to(device=mid_tensor.device, dtype=mid_tensor.dtype)
                selected = bool(param_mask.any().item())
            else:
                param_mask = torch.tensor(float(bool(mask_value)), device=mid_tensor.device, dtype=mid_tensor.dtype)
                selected = bool(mask_value)
            self.param_masks[name] = param_mask

            trainable = name in trainable_names and selected
            self.mid_params[safe_name] = nn.Parameter(mid_tensor, requires_grad=trainable)

        for name in self.buffer_names:
            start_tensor = start_state[name].detach().clone()
            end_tensor = end_state[name].detach().clone()
            if torch.is_floating_point(start_tensor):
                mid_tensor = 0.5 * start_tensor + 0.5 * end_tensor
            else:
                mid_tensor = start_tensor.clone()
            self.start_buffers[name] = start_tensor
            self.end_buffers[name] = end_tensor
            self.mid_buffers[name] = mid_tensor

    def coeffs(self, t):
        if not torch.is_tensor(t):
            t = torch.tensor([t], dtype=torch.float32, device=next(self.template_model.parameters()).device)
        if t.dim() == 0:
            t = t.unsqueeze(0)
        one_minus_t = 1.0 - t
        return one_minus_t * one_minus_t, 2.0 * t * one_minus_t, t * t

    def interpolated_state(self, t, active_names=None):
        c0, c1, c2 = self.coeffs(t)
        state = OrderedDict()
        active_names = None if active_names is None else set(active_names)
        for safe_name, param in self.mid_params.items():
            name = self._safe_to_name[safe_name]
            if name not in self.trainable_names:
                state[name] = self.start_params[name]
                continue
            current_param = param if active_names is None or name in active_names else param.detach()
            mask = self.param_masks[name]
            curve_param = c0 * self.start_params[name] + c1 * current_param + c2 * self.end_params[name]
            fixed_curve_param = (
                c0 * self.start_params[name]
                + c1 * self.fixed_mid_params[name]
                + c2 * self.end_params[name]
            )
            state[name] = fixed_curve_param + mask * (curve_param - fixed_curve_param)
        for name in self.buffer_names:
            state[name] = self.start_buffers[name]
        return state

    def forward(self, inputs, t=None, active_names=None):
        if t is None:
            t = torch.rand(1, device=inputs.device)
        state = self.interpolated_state(t, active_names=active_names)
        return functional_call(self.template_model, state, (inputs,))

    def export_state_dict(self, t):
        state = self.interpolated_state(t)
        return OrderedDict((name, tensor.detach().cpu()) for name, tensor in state.items())

    def trainable_parameters(self):
        return [param for param in self.mid_params.parameters() if param.requires_grad]

    def curve_definition(self):
        start_state = OrderedDict((name, value.detach().cpu()) for name, value in self.start_params.items())
        end_state = OrderedDict((name, value.detach().cpu()) for name, value in self.end_params.items())
        mid_state = OrderedDict(
            (self._safe_to_name[safe_name], value.detach().cpu())
            for safe_name, value in self.mid_params.items()
        )
        for name in self.buffer_names:
            start_state[name] = self.start_buffers[name].detach().cpu()
            end_state[name] = self.end_buffers[name].detach().cpu()
            mid_state[name] = self.mid_buffers[name].detach().cpu()
        return {
            "start_state_dict": start_state,
            "end_state_dict": end_state,
            "mid_state_dict": mid_state,
            "mask": OrderedDict((name, value.detach().cpu()) for name, value in self.param_masks.items()),
        }


def resolve_default_target_model_path(split_mode):
    return os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "unnlearn_model",
        f"neggrad_plus_best_{split_mode}.pth",
    )


def resolve_default_mask_path(split_mode):
    return os.path.join(
        os.path.dirname(__file__),
        "masks",
        f"neus_mask_{split_mode}_k0.5_kr0.0.pt",
    )


def load_mask(mask_path):
    if mask_path is None:
        return None
    payload = torch.load(mask_path, map_location="cpu")
    if isinstance(payload, dict) and "mask" in payload:
        return payload
    return {"mask": payload}


def batch_accuracy(logits, targets):
    preds = logits.argmax(dim=1)
    return 100.0 * preds.eq(targets).sum().item() / max(targets.size(0), 1)


def adaptive_beta(retain_acc, forget_acc, retain_base, forget_base, beta_low, beta_high):
    if forget_acc <= forget_base:
        return 0.0

    forget_excess = (forget_acc - forget_base) / max(100.0 - forget_base, 1.0)
    retain_deficit = max(retain_base - retain_acc, 0.0) / max(abs(retain_base), 1.0)
    if retain_deficit > forget_excess:
        return beta_low
    return beta_high


def resolve_beta_references(args, target_metrics):
    retain_base = args.retain_base if args.retain_base is not None else target_metrics["ra_train"]
    if args.forget_base is not None:
        forget_base = args.forget_base
    elif args.split_mode == "class":
        forget_base = 0.0
    else:
        forget_base = target_metrics["forget_train_acc"]
    return retain_base, forget_base


def save_curve_checkpoint(output_path, epoch, curve_model, best_t, best_metrics, layer_targets, args):
    torch.save(
        {
            "epoch": epoch,
            "best_t": best_t,
            "curve": curve_model.curve_definition(),
            "metrics": best_metrics,
            "layer_targets": dict(layer_targets),
            "args": vars(args),
        },
        output_path,
    )


def save_best_point(output_path, epoch, state_dict, best_t, best_metrics, layer_targets, args):
    torch.save(
        {
            "epoch": epoch,
            "best_t": best_t,
            "model_state_dict": state_dict,
            "metrics": best_metrics,
            "layer_targets": dict(layer_targets),
            "args": vars(args),
        },
        output_path,
    )


def evaluate_curve_points(curve_model, eval_loaders, device, split_mode, t_values):
    best = None
    for t_value in t_values:
        t_tensor = torch.tensor([t_value], dtype=torch.float32, device=device)
        metrics = compute_metrics(curve_model, eval_loaders, device, t=t_tensor)
        selection_score = compute_selection_score(metrics, split_mode)
        current = {
            "t": float(t_value),
            "metrics": metrics,
            "selection_score": selection_score,
        }
        if best is None or selection_score < best["selection_score"]:
            best = current
    return best


def validate_mask_payload(mask_payload, args, layer_targets, parameter_names):
    if mask_payload is None:
        return None

    metadata = mask_payload.get("metadata", {})
    mask_split_mode = metadata.get("split_mode")
    if mask_split_mode is not None and mask_split_mode != args.split_mode:
        raise ValueError(
            f"Mask split mode is {mask_split_mode!r}, but unlearning split mode is {args.split_mode!r}."
        )

    mask_routing_unit = metadata.get("routing_unit")
    if mask_routing_unit != ROUTING_UNIT:
        raise ValueError(
            "Mask routing metadata is missing or incompatible. "
            "Regenerate the NEUS mask with the current thresholded patch-majority flow code."
        )
    if metadata.get("flow_threshold") != FLOW_TARGET_VOTE_THRESHOLD:
        raise ValueError(
            "Mask flow threshold is missing or incompatible. Regenerate the NEUS mask."
        )
    if metadata.get("last_layer_take_all") != LAST_LAYER_TAKE_ALL:
        raise ValueError(
            "Mask last-layer fallback setting is missing or incompatible. Regenerate the NEUS mask."
        )

    mask_original_path = metadata.get("original_model_path")
    if mask_original_path is not None and os.path.abspath(mask_original_path) != os.path.abspath(args.original_model_path):
        raise ValueError(
            f"Mask was generated from {mask_original_path}, but the current original model is "
            f"{args.original_model_path}."
        )

    for config_name in ("image_size", "num_classes", "embed_dim", "depth", "num_heads"):
        mask_value = metadata.get(config_name)
        current_value = getattr(args, config_name)
        if mask_value is not None and mask_value != current_value:
            raise ValueError(
                f"Mask {config_name}={mask_value} does not match current {config_name}={current_value}."
            )

    saved_targets = mask_payload.get("layer_targets")
    if saved_targets is not None:
        saved_targets = {int(layer_idx): int(expert_idx) for layer_idx, expert_idx in saved_targets.items()}
        if saved_targets != dict(layer_targets):
            raise ValueError(
                f"Mask layer targets {saved_targets} do not match current targets {dict(layer_targets)}. "
                "Regenerate the mask with the same model and split."
            )

    mask = mask_payload["mask"]
    unknown_names = set(mask) - set(parameter_names)
    if unknown_names:
        raise ValueError(
            "Mask contains parameter names that do not exist in the current model. "
            f"Unknown={sorted(unknown_names)}"
        )
    return mask


def forward_with_gate_probs(curve_model, inputs, t, active_names=None, capture_layers=None):
    gate_logits = OrderedDict()
    handles = []
    capture_layers = None if capture_layers is None else set(capture_layers)
    for layer_idx, blk in enumerate(curve_model.template_model.blocks):
        if (
            getattr(blk, "use_moe", False)
            and hasattr(blk.mlp, "gate")
            and (capture_layers is None or layer_idx in capture_layers)
        ):
            handles.append(
                blk.mlp.gate.register_forward_hook(
                    lambda module, module_inputs, output, idx=layer_idx: gate_logits.__setitem__(idx, output)
                )
            )

    try:
        logits, aux = curve_model(inputs, t=t, active_names=active_names)
    finally:
        for handle in handles:
            handle.remove()

    layer_gate_probs = OrderedDict((layer_idx, torch.softmax(logits_i, dim=-1)) for layer_idx, logits_i in gate_logits.items())
    return logits, aux, layer_gate_probs


def main():
    parser = argparse.ArgumentParser(description="NEUS unlearning with layerwise routing flow and MCU curve")
    parser.add_argument(
        "--original-model-path",
        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "model", "moevit_cifar100_best.pth"),
    )
    parser.add_argument("--target-model-path", default=None)
    parser.add_argument(
        "--split-dir",
        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "split_results_cifar100_fine"),
    )
    parser.add_argument("--split-mode", default="class", choices=["class", "random"])
    parser.add_argument("--mask-path", default=None)
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "unnlearn_model"),
    )
    parser.add_argument(
        "--curve-output-dir",
        default=os.path.join(os.path.dirname(__file__), "checkpoints"),
    )
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--train-batch-size", default=128, type=int)
    parser.add_argument("--eval-batch-size", default=128, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--lr", default=3e-2, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight-decay", default=0.0, type=float)
    parser.add_argument("--train-t-min", default=0.0, type=float)
    parser.add_argument("--train-t-max", default=1.0, type=float)
    parser.add_argument("--eval-start-t", default=0.75, type=float)
    parser.add_argument("--eval-end-t", default=1.0, type=float)
    parser.add_argument("--eval-num-points", default=11, type=int)
    parser.add_argument("--beta-mode", default="fixed", choices=["fixed", "adaptive"])
    parser.add_argument("--beta", default=0.15, type=float)
    parser.add_argument("--beta-low", default=0.01, type=float)
    parser.add_argument("--beta-high", default=0.5, type=float)
    parser.add_argument("--anchor-weight", default=0.1, type=float)
    parser.add_argument("--retain-base", default=None, type=float)
    parser.add_argument("--forget-base", default=None, type=float)
    parser.add_argument("--save-every", default=1, type=int)
    parser.add_argument("--image-size", default=32, type=int)
    parser.add_argument("--num-classes", default=100, type=int)
    parser.add_argument("--embed-dim", default=256, type=int)
    parser.add_argument("--depth", default=12, type=int)
    parser.add_argument("--num-heads", default=8, type=int)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.target_model_path is None:
        args.target_model_path = resolve_default_target_model_path(args.split_mode)
    if args.mask_path is None:
        args.mask_path = resolve_default_mask_path(args.split_mode)

    if not os.path.exists(args.original_model_path):
        raise FileNotFoundError(f"Original checkpoint not found: {args.original_model_path}")
    if not os.path.exists(args.target_model_path):
        raise FileNotFoundError(f"Target checkpoint not found: {args.target_model_path}")
    if not os.path.exists(args.mask_path):
        raise FileNotFoundError(f"Mask file not found: {args.mask_path}")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.curve_output_dir, exist_ok=True)
    curve_output_path = os.path.join(args.curve_output_dir, f"neus_curve_{args.split_mode}.pth")
    best_output_path = os.path.join(args.output_dir, f"neus_best_{args.split_mode}.pth")

    trainset, trainset_eval, testset = build_cifar100_datasets()
    split_indices = load_split_indices(args.split_dir, args.split_mode)
    subsets = build_split_subsets(trainset, trainset_eval, testset, split_indices)
    eval_loaders = build_eval_loaders(subsets, testset, args.eval_batch_size, args.num_workers)

    forget_loader = DataLoader(
        subsets["forget_train"],
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    retain_loader = DataLoader(
        subsets["retain_train"],
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    forget_target_loader = DataLoader(
        subsets["forget_train_eval"],
        batch_size=args.train_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    template_model = build_model(args, device)
    load_checkpoint_state(template_model, args.original_model_path, device)
    layer_targets, per_layer_stats = compute_layerwise_target_experts(template_model, forget_target_loader, device)
    freeze_to_layer_targets(template_model, layer_targets)
    trainable_names = get_trainable_param_names(template_model, layer_targets)
    original_state = OrderedDict((name, tensor.detach().clone()) for name, tensor in template_model.state_dict().items())

    target_model = build_model(args, device)
    load_checkpoint_state(target_model, args.target_model_path, device)
    target_state = OrderedDict((name, tensor.detach().clone()) for name, tensor in target_model.state_dict().items())
    target_metrics = compute_metrics(target_model, eval_loaders, device)
    retain_base, forget_base = resolve_beta_references(args, target_metrics)
    del target_model

    mask_payload = load_mask(args.mask_path)
    parameter_names = [name for name, _ in template_model.named_parameters()]
    mask = validate_mask_payload(mask_payload, args, layer_targets, parameter_names)
    curve_model = BezierCurveModel(template_model, original_state, target_state, trainable_names, mask=mask).to(device)
    del original_state, target_state
    trainable_params = curve_model.trainable_parameters()
    if not trainable_params:
        raise ValueError("No trainable parameters left for NEUS. Check mask and layer target selection.")

    optimizer = optim.SGD(trainable_params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    criterion = nn.CrossEntropyLoss()

    print(f"Using device: {device}")
    print(f"Original model: {args.original_model_path}")
    print(f"Target model: {args.target_model_path}")
    print(f"Split mode: {args.split_mode}")
    print(f"Mask path: {args.mask_path}")
    print(f"Layer targets: {format_layer_targets(layer_targets)}")
    for layer_idx, stats in per_layer_stats.items():
        print(
            f"  Layer {layer_idx}: expert={stats['target_idx']} | "
            f"remaining={stats['remaining_before']} | matched={stats['matched_count']} | "
            f"next={stats['remaining_after']}"
        )
    print(
        "Target metrics | "
        f"ForgetAcc(train): {target_metrics['forget_train_acc']:.2f}% | "
        f"RA(train): {target_metrics['ra_train']:.2f}% | "
        f"ForgetAcc(test): {target_metrics['forget_test_acc']:.2f}% | "
        f"RA(test): {target_metrics['ra_test']:.2f}%"
    )

    best_selection_score = float("inf")
    best_epoch = 0
    start_time = time.time()
    t_values = torch.linspace(args.eval_start_t, args.eval_end_t, steps=args.eval_num_points).tolist()

    for epoch in range(1, args.epochs + 1):
        curve_model.train()
        retain_iter = cycle(retain_loader)
        running_loss = 0.0
        running_retain = 0.0
        running_forget = 0.0
        running_anchor = 0.0
        running_beta = 0.0
        running_matched = 0

        pbar = tqdm(forget_loader, desc=f"NEUS Epoch {epoch}/{args.epochs}")
        for step, (inputs_f, targets_f) in enumerate(pbar, start=1):
            inputs_r, targets_r = next(retain_iter)
            inputs_f = inputs_f.to(device, non_blocking=True)
            targets_f = targets_f.to(device, non_blocking=True)
            inputs_r = inputs_r.to(device, non_blocking=True)
            targets_r = targets_r.to(device, non_blocking=True)
            t = torch.empty(1, device=device).uniform_(args.train_t_min, args.train_t_max)

            optimizer.zero_grad(set_to_none=True)
            logits_r, _ = curve_model(inputs_r, t=t)
            loss_retain = criterion(logits_r, targets_r)

            flow_masks, _ = compute_fixed_flow_masks(template_model, inputs_f, layer_targets)
            forget_sum = torch.tensor(0.0, device=device)
            anchor_sum = torch.tensor(0.0, device=device)
            matched_count = 0
            forget_correct = 0

            for layer_idx, target_idx in layer_targets.items():
                current_mask = flow_masks[layer_idx]
                current_count = int(current_mask.sum().item())
                if current_count == 0:
                    continue

                current_inputs = inputs_f[current_mask]
                current_targets = targets_f[current_mask]
                active_names = get_layer_param_names(template_model, layer_idx, target_idx)
                current_logits, _, current_gate_probs = forward_with_gate_probs(
                    curve_model,
                    current_inputs,
                    t,
                    active_names=active_names,
                    capture_layers={layer_idx},
                )
                forget_sum = forget_sum + F.cross_entropy(
                    current_logits, current_targets, reduction="sum"
                )
                anchor_sum = anchor_sum + current_count * compute_anchor_loss(
                    current_gate_probs[layer_idx], target_idx
                )
                forget_correct += current_logits.detach().argmax(dim=1).eq(current_targets).sum().item()
                matched_count += current_count

            if matched_count > 0:
                loss_forget = forget_sum / matched_count
                loss_anchor = anchor_sum / matched_count
                forget_acc = 100.0 * forget_correct / matched_count
            else:
                loss_forget = torch.tensor(0.0, device=device)
                loss_anchor = torch.tensor(0.0, device=device)
                forget_acc = forget_base

            if args.beta_mode == "adaptive":
                retain_acc = batch_accuracy(logits_r.detach(), targets_r)
                beta = adaptive_beta(retain_acc, forget_acc, retain_base, forget_base, args.beta_low, args.beta_high)
            else:
                beta = args.beta

            loss = loss_retain - beta * loss_forget + args.anchor_weight * loss_anchor
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            running_retain += loss_retain.item()
            running_forget += loss_forget.item()
            running_anchor += loss_anchor.item()
            running_beta += beta
            running_matched += matched_count
            pbar.set_postfix({
                "loss": f"{running_loss / step:.4f}",
                "retain": f"{running_retain / step:.4f}",
                "forget": f"{running_forget / step:.4f}",
                "anchor": f"{running_anchor / step:.4f}",
                "beta": f"{running_beta / step:.4f}",
                "matched": f"{running_matched / step:.1f}",
            })

        scheduler.step()
        best_curve_point = evaluate_curve_points(curve_model, eval_loaders, device, args.split_mode, t_values)
        metrics = best_curve_point["metrics"]
        selection_score = best_curve_point["selection_score"]
        best_t = best_curve_point["t"]

        print(
            f"Epoch {epoch}/{args.epochs} | best_t={best_t:.4f} | "
            f"ForgetAcc(train): {metrics['forget_train_acc']:.2f}% | "
            f"RA(train): {metrics['ra_train']:.2f}% | "
            f"ForgetAcc(test): {metrics['forget_test_acc']:.2f}% | "
            f"RA(test): {metrics['ra_test']:.2f}% | "
            f"selection={selection_score:.6f}"
        )

        is_best = selection_score < best_selection_score
        if is_best:
            best_selection_score = selection_score
            best_epoch = epoch
            export_state = curve_model.export_state_dict(torch.tensor([best_t], dtype=torch.float32, device=device))
            save_best_point(best_output_path, epoch, export_state, best_t, metrics, layer_targets, args)

        if epoch % args.save_every == 0 or is_best:
            save_curve_checkpoint(curve_output_path, epoch, curve_model, best_t, metrics, layer_targets, args)

    total_minutes = (time.time() - start_time) / 60.0
    print(f"Best epoch: {best_epoch}")
    print(f"Best selection score: {best_selection_score:.6f}")
    print(f"Saved best model to: {best_output_path}")
    print(f"Saved curve checkpoint to: {curve_output_path}")
    print(f"Total time: {total_minutes:.2f} minutes")


if __name__ == "__main__":
    main()
