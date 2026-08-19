import argparse
import os
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import (
    FLOW_TARGET_VOTE_THRESHOLD,
    LAST_LAYER_TAKE_ALL,
    ROUTING_UNIT,
    build_cifar100_datasets,
    build_model,
    build_split_subsets,
    compute_fixed_flow_masks,
    compute_layerwise_target_experts,
    freeze_to_layer_targets,
    get_layer_param_names,
    get_trainable_param_names,
    load_checkpoint_state,
    load_split_indices,
    set_seed,
)


def average_gradient_signal(
    model,
    loader,
    device,
    criterion,
    sign,
    allowed_names,
    layer_targets=None,
    max_batches=None,
):
    named_params = dict(model.named_parameters())
    gradients = OrderedDict(
        (name, torch.zeros_like(named_params[name], device=device))
        for name in sorted(allowed_names)
    )
    processed_batches = 0

    model.eval()
    for batch_idx, (inputs, targets) in enumerate(tqdm(loader, desc="Grad Signal", leave=False), start=1):
        if max_batches is not None and batch_idx > max_batches:
            break
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        processed_batches += 1
        if layer_targets is None:
            outputs, _ = model(inputs)
            loss = sign * criterion(outputs, targets)
            names = list(gradients)
            grads = torch.autograd.grad(loss, [named_params[name] for name in names], allow_unused=True)
            with torch.no_grad():
                for name, grad in zip(names, grads):
                    if grad is not None:
                        gradients[name] += grad.detach().abs()
            continue

        flow_masks, _ = compute_fixed_flow_masks(model, inputs, layer_targets)
        for layer_idx, target_idx in layer_targets.items():
            current_mask = flow_masks[layer_idx]
            if not current_mask.any():
                continue
            outputs, _ = model(inputs[current_mask])
            loss = sign * criterion(outputs, targets[current_mask])
            names = sorted(get_layer_param_names(model, layer_idx, target_idx))
            grads = torch.autograd.grad(loss, [named_params[name] for name in names], allow_unused=True)
            with torch.no_grad():
                for name, grad in zip(names, grads):
                    if grad is not None:
                        gradients[name] += grad.detach().abs()

    if processed_batches == 0:
        raise RuntimeError("No batch was processed while generating the mask.")

    with torch.no_grad():
        scores = OrderedDict((name, grad / processed_batches) for name, grad in gradients.items())
    return scores


def build_mask(retain_scores, forget_scores, mask_ratio, retain_ratio):
    if not 0.0 <= mask_ratio <= 1.0:
        raise ValueError("mask_ratio must be in [0, 1].")
    if not 0.0 <= retain_ratio <= 1.0:
        raise ValueError("retain_ratio must be in [0, 1].")

    names = list(forget_scores)
    flat_forget = torch.cat([forget_scores[name].flatten().cpu() for name in names])
    flat_retain = torch.cat([retain_scores[name].flatten().cpu() for name in names])
    total_params = flat_forget.numel()

    protected = torch.zeros(total_params, dtype=torch.bool)
    protected_count = int(total_params * retain_ratio)
    if protected_count > 0:
        protected_idx = torch.topk(flat_retain, protected_count, largest=True).indices
        protected[protected_idx] = True

    selected = torch.zeros(total_params, dtype=torch.bool)
    requested_count = int(total_params * mask_ratio)
    selected_count = min(requested_count, int((~protected).sum().item()))
    if selected_count > 0:
        eligible_scores = flat_forget.clone()
        eligible_scores[protected] = float("-inf")
        selected_idx = torch.topk(eligible_scores, selected_count, largest=True).indices
        selected[selected_idx] = True

    mask = OrderedDict()
    offset = 0
    for name in names:
        numel = forget_scores[name].numel()
        mask[name] = selected[offset:offset + numel].reshape(forget_scores[name].shape)
        offset += numel

    return (
        mask,
        selected_count / max(total_params, 1),
        protected_count / max(total_params, 1),
    )


def main():
    parser = argparse.ArgumentParser(description="Generate NEUS mask on trainable gate+expert parameters only")
    parser.add_argument(
        "--original-model-path",
        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "model", "moevit_cifar100_best.pth"),
    )
    parser.add_argument(
        "--split-dir",
        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "split_results_cifar100_fine"),
    )
    parser.add_argument("--split-mode", default="class", choices=["class", "random"])
    parser.add_argument("--output-dir", default=os.path.join(os.path.dirname(__file__), "masks"))
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--mask-ratio", default=0.5, type=float)
    parser.add_argument("--retain-ratio", default=0.0, type=float)
    parser.add_argument("--max-batches", default=None, type=int)
    parser.add_argument("--image-size", default=32, type=int)
    parser.add_argument("--num-classes", default=100, type=int)
    parser.add_argument("--embed-dim", default=256, type=int)
    parser.add_argument("--depth", default=12, type=int)
    parser.add_argument("--num-heads", default=8, type=int)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.output_dir, exist_ok=True)
    if args.output_name is None:
        output_name = f"neus_mask_{args.split_mode}_k{args.mask_ratio}_kr{args.retain_ratio}.pt"
    else:
        output_name = args.output_name
    output_path = os.path.join(args.output_dir, output_name)

    trainset, trainset_eval, testset = build_cifar100_datasets()
    split_indices = load_split_indices(args.split_dir, args.split_mode)
    subsets = build_split_subsets(trainset, trainset_eval, testset, split_indices)

    forget_loader = DataLoader(
        subsets["forget_train"],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    retain_loader = DataLoader(
        subsets["retain_train"],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = build_model(args, device)
    load_checkpoint_state(model, args.original_model_path, device)

    target_loader = DataLoader(
        subsets["forget_train_eval"],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    layer_targets, per_layer_stats = compute_layerwise_target_experts(model, target_loader, device)
    allowed_names = get_trainable_param_names(model, layer_targets)
    freeze_to_layer_targets(model, layer_targets)

    criterion = nn.CrossEntropyLoss()
    retain_scores = average_gradient_signal(
        model, retain_loader, device, criterion, sign=1.0, allowed_names=allowed_names, max_batches=args.max_batches
    )
    forget_scores = average_gradient_signal(
        model,
        forget_loader,
        device,
        criterion,
        sign=-1.0,
        allowed_names=allowed_names,
        layer_targets=layer_targets,
        max_batches=args.max_batches,
    )

    mask, real_mask_ratio, real_retain_ratio = build_mask(
        retain_scores, forget_scores, args.mask_ratio, args.retain_ratio
    )

    torch.save(
        {
            "mask": mask,
            "layer_targets": dict(layer_targets),
            "per_layer_stats": dict(per_layer_stats),
            "metadata": {
                "split_mode": args.split_mode,
                "mask_ratio": args.mask_ratio,
                "retain_ratio": args.retain_ratio,
                "real_mask_ratio": real_mask_ratio,
                "real_retain_ratio": real_retain_ratio,
                "routing_unit": ROUTING_UNIT,
                "flow_threshold": FLOW_TARGET_VOTE_THRESHOLD,
                "last_layer_take_all": LAST_LAYER_TAKE_ALL,
                "original_model_path": args.original_model_path,
                "image_size": args.image_size,
                "num_classes": args.num_classes,
                "embed_dim": args.embed_dim,
                "depth": args.depth,
                "num_heads": args.num_heads,
                "max_batches": args.max_batches,
                "seed": args.seed,
            },
        },
        output_path,
    )
    print(f"Saved mask to: {output_path}")
    print(f"Trainable tensor count: {len(allowed_names)}")
    print(f"Real selected parameter ratio: {real_mask_ratio:.6f}")
    print(f"Real retain-filter ratio: {real_retain_ratio:.6f}")


if __name__ == "__main__":
    main()
