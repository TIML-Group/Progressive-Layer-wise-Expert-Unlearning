import argparse
import os
import time
from collections import OrderedDict
from itertools import cycle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from torch.func import functional_call
except ImportError:
    from torch.nn.utils.stateless import functional_call

from common import (
    build_cifar100_datasets,
    build_eval_loaders,
    build_model,
    build_split_subsets,
    compute_metrics,
    compute_selection_score,
    load_checkpoint_state,
    load_split_indices,
    set_seed,
)


class BezierCurveModel(nn.Module):
    def __init__(self, template_model, start_state, end_state, mask=None):
        super().__init__()
        self.template_model = template_model
        self.template_model.eval()

        self.parameter_names = [name for name, _ in template_model.named_parameters()]
        self.buffer_names = [name for name, _ in template_model.named_buffers()]
        self._safe_to_name = {}

        self.mid_params = nn.ParameterDict()
        self.start_params = {}
        self.end_params = {}

        self.mid_buffers = {}
        self.start_buffers = {}
        self.end_buffers = {}

        mask = mask or {}

        for name in self.parameter_names:
            safe_name = name.replace(".", "__")
            self._safe_to_name[safe_name] = name

            start_tensor = start_state[name].detach().clone()
            end_tensor = end_state[name].detach().clone()
            mid_tensor = 0.5 * start_tensor + 0.5 * end_tensor

            self.start_params[name] = start_tensor
            self.end_params[name] = end_tensor

            trainable = bool(mask.get(name, 1))
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

    def interpolated_state(self, t):
        c0, c1, c2 = self.coeffs(t)
        state = OrderedDict()

        for safe_name, param in self.mid_params.items():
            name = self._safe_to_name[safe_name]
            state[name] = c0 * self.start_params[name] + c1 * param + c2 * self.end_params[name]

        for name in self.buffer_names:
            if torch.is_floating_point(self.start_buffers[name]):
                state[name] = c0 * self.start_buffers[name] + c1 * self.mid_buffers[name] + c2 * self.end_buffers[name]
            else:
                state[name] = self.start_buffers[name]

        return state

    def forward(self, inputs, t=None):
        if t is None:
            t = torch.rand(1, device=inputs.device)
        state = self.interpolated_state(t)
        return functional_call(self.template_model, state, (inputs,))

    def export_state_dict(self, t):
        state = self.interpolated_state(t)
        return OrderedDict((name, tensor.detach().cpu()) for name, tensor in state.items())

    def trainable_parameters(self):
        return [param for param in self.mid_params.parameters() if param.requires_grad]

    def trainable_summary(self):
        total = 0
        names = []
        for safe_name, param in self.mid_params.items():
            if param.requires_grad:
                total += param.numel()
                names.append(self._safe_to_name[safe_name])
        return total, names


def load_mask(mask_path):
    if mask_path is None:
        return None
    mask_payload = torch.load(mask_path, map_location="cpu")
    if isinstance(mask_payload, dict) and "mask" in mask_payload:
        return mask_payload["mask"]
    return mask_payload


def resolve_beta_references(args, target_metrics):
    retain_base = args.retain_base if args.retain_base is not None else target_metrics["ra_train"]
    if args.forget_base is not None:
        forget_base = args.forget_base
    elif args.split_mode == "class":
        forget_base = 0.0
    else:
        forget_base = target_metrics["forget_train_acc"]
    return retain_base, forget_base


def safe_relative_gap(current, base):
    if abs(base) < 1e-8:
        return 0.0 if current <= base else float("inf")
    return (current - base) / base


def adaptive_beta(retain_acc, forget_acc, retain_base, forget_base, beta_low, beta_high):
    if forget_acc <= forget_base:
        return 0.0
    forget_gap = safe_relative_gap(forget_acc, forget_base)
    retain_gap = safe_relative_gap(retain_acc, retain_base)
    if forget_gap < retain_gap:
        return beta_low
    return beta_high


def batch_accuracy(logits, targets):
    preds = logits.argmax(dim=1)
    return 100.0 * preds.eq(targets).sum().item() / max(targets.size(0), 1)


def evaluate_curve_points(curve_model, eval_loaders, device, split_mode, t_values):
    best = None
    for t_value in t_values:
        t_tensor = torch.tensor([t_value], dtype=torch.float32, device=device)
        metrics = compute_metrics(curve_model, eval_loaders, device, t=t_tensor)
        selection_score = compute_selection_score(metrics, split_mode)
        candidate = {
            "t": float(t_value),
            "metrics": metrics,
            "selection_score": selection_score,
        }
        if best is None or selection_score < best["selection_score"]:
            best = candidate
    return best


def save_curve_checkpoint(output_path, epoch, curve_model, best_t, best_metrics, args):
    torch.save(
        {
            "epoch": epoch,
            "best_t": best_t,
            "mid_state_dict": curve_model.state_dict(),
            "metrics": best_metrics,
            "args": vars(args),
        },
        output_path,
    )


def save_best_point(output_path, epoch, state_dict, best_t, best_metrics, args):
    torch.save(
        {
            "epoch": epoch,
            "best_t": best_t,
            "model_state_dict": state_dict,
            "metrics": best_metrics,
            "args": vars(args),
        },
        output_path,
    )


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
        f"mask_{split_mode}_k0.5_kr0.0.pt",
    )


def main():
    parser = argparse.ArgumentParser(description="MCU unlearning on CIFAR100 with MoE-ViT")
    parser.add_argument(
        "--original-model-path",
        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "model", "moevit_cifar100_best.pth"),
        help="Original pretrained checkpoint used as the start point of the curve",
    )
    parser.add_argument(
        "--target-model-path",
        default=None,
        help="Existing unlearned checkpoint used as the end point of the curve. Defaults to neggrad_plus_best_{split_mode}.pth",
    )
    parser.add_argument(
        "--split-dir",
        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "split_results_cifar100_fine"),
    )
    parser.add_argument("--split-mode", default="random", choices=["class", "random"])
    parser.add_argument("--mask-path", default=None, help="Optional mask generated by generate_mask.py")
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "unnlearn_model"),
        help="Directory used to save the best exported MCU model",
    )
    parser.add_argument(
        "--curve-output-dir",
        default=os.path.join(os.path.dirname(__file__), "checkpoints"),
        help="Directory used to save the trainable MCU curve checkpoint",
    )
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--train-batch-size", default=128, type=int)
    parser.add_argument("--eval-batch-size", default=128, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight-decay", default=0.0, type=float)
    parser.add_argument("--train-t-min", default=0.0, type=float)
    parser.add_argument("--train-t-max", default=1.0, type=float)
    parser.add_argument("--eval-start-t", default=0.0, type=float)
    parser.add_argument("--eval-end-t", default=1.0, type=float)
    parser.add_argument("--eval-num-points", default=11, type=int)
    parser.add_argument("--beta-mode", default="fixed", choices=["fixed", "adaptive"])
    parser.add_argument("--beta", default=0.15, type=float)
    parser.add_argument("--beta-low", default=0.01, type=float)
    parser.add_argument("--beta-high", default=0.5, type=float)
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
    curve_output_path = os.path.join(args.curve_output_dir, f"mcu_curve_{args.split_mode}.pth")
    best_output_path = os.path.join(args.output_dir, f"mcu_best_{args.split_mode}.pth")

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

    template_model = build_model(args, device)
    load_checkpoint_state(template_model, args.original_model_path, device)
    original_state = OrderedDict((name, tensor.detach().clone()) for name, tensor in template_model.state_dict().items())

    target_model = build_model(args, device)
    load_checkpoint_state(target_model, args.target_model_path, device)
    target_state = OrderedDict((name, tensor.detach().clone()) for name, tensor in target_model.state_dict().items())

    target_metrics = compute_metrics(target_model, eval_loaders, device)
    retain_base, forget_base = resolve_beta_references(args, target_metrics)

    mask = load_mask(args.mask_path)
    curve_model = BezierCurveModel(template_model, original_state, target_state, mask=mask).to(device)
    trainable_params = curve_model.trainable_parameters()
    if not trainable_params:
        raise ValueError("No trainable parameters remain in the MCU curve model. Check the provided mask.")

    trainable_count, trainable_names = curve_model.trainable_summary()
    optimizer = optim.SGD(trainable_params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    criterion = nn.CrossEntropyLoss()

    print(f"Using device: {device}")
    print(f"Original model: {args.original_model_path}")
    print(f"Target model: {args.target_model_path}")
    print(f"Split mode: {args.split_mode}")
    print(f"Mask path: {args.mask_path}")
    print(f"Trainable tensors: {len(trainable_names)}")
    print(f"Trainable parameters: {trainable_count}")
    print(
        "Target metrics | "
        f"ForgetAcc(train): {target_metrics['forget_train_acc']:.2f}% | "
        f"RA(train): {target_metrics['ra_train']:.2f}% | "
        f"ForgetAcc(test): {target_metrics['forget_test_acc']:.2f}% | "
        f"RA(test): {target_metrics['ra_test']:.2f}%"
    )
    print(f"Adaptive beta references | retain_base={retain_base:.4f}, forget_base={forget_base:.4f}")

    best_selection_score = float("inf")
    best_epoch = 0
    start_time = time.time()

    steps_per_epoch = max(len(forget_loader), len(retain_loader))
    t_values = torch.linspace(args.eval_start_t, args.eval_end_t, steps=args.eval_num_points).tolist()

    for epoch in range(1, args.epochs + 1):
        curve_model.train()
        forget_iter = cycle(forget_loader)
        retain_iter = cycle(retain_loader)

        running_loss = 0.0
        running_retain = 0.0
        running_forget = 0.0
        running_beta = 0.0

        pbar = tqdm(range(1, steps_per_epoch + 1), desc=f"MCU Epoch {epoch}/{args.epochs}")
        for step in pbar:
            inputs_f, targets_f = next(forget_iter)
            inputs_r, targets_r = next(retain_iter)
            inputs_f = inputs_f.to(device, non_blocking=True)
            targets_f = targets_f.to(device, non_blocking=True)
            inputs_r = inputs_r.to(device, non_blocking=True)
            targets_r = targets_r.to(device, non_blocking=True)

            t = torch.empty(1, device=device).uniform_(args.train_t_min, args.train_t_max)

            optimizer.zero_grad(set_to_none=True)
            logits_f, _ = curve_model(inputs_f, t=t)
            logits_r, _ = curve_model(inputs_r, t=t)

            loss_forget = criterion(logits_f, targets_f)
            loss_retain = criterion(logits_r, targets_r)

            if args.beta_mode == "adaptive":
                forget_acc = batch_accuracy(logits_f.detach(), targets_f)
                retain_acc = batch_accuracy(logits_r.detach(), targets_r)
                beta = adaptive_beta(retain_acc, forget_acc, retain_base, forget_base, args.beta_low, args.beta_high)
            else:
                beta = args.beta

            loss = loss_retain - beta * loss_forget
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            running_retain += loss_retain.item()
            running_forget += loss_forget.item()
            running_beta += beta

            pbar.set_postfix({
                "loss": f"{running_loss / step:.4f}",
                "retain": f"{running_retain / step:.4f}",
                "forget": f"{running_forget / step:.4f}",
                "beta": f"{running_beta / step:.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.5f}",
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

        if epoch % args.save_every == 0:
            save_curve_checkpoint(curve_output_path, epoch, curve_model, best_t, metrics, args)

        if selection_score < best_selection_score:
            best_selection_score = selection_score
            best_epoch = epoch
            export_state = curve_model.export_state_dict(torch.tensor([best_t], dtype=torch.float32, device=device))
            save_best_point(best_output_path, epoch, export_state, best_t, metrics, args)
            save_curve_checkpoint(curve_output_path, epoch, curve_model, best_t, metrics, args)

    total_minutes = (time.time() - start_time) / 60.0
    print(f"Best epoch: {best_epoch}")
    print(f"Best selection score: {best_selection_score:.6f}")
    print(f"Curve checkpoint: {curve_output_path}")
    print(f"Best exported checkpoint: {best_output_path}")
    print(f"Total time: {total_minutes:.2f} minutes")


if __name__ == "__main__":
    main()
