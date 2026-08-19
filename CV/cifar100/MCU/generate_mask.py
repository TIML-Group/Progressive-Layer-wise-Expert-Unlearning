import argparse
import os
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import (
    build_cifar100_datasets,
    build_model,
    build_split_subsets,
    load_checkpoint_state,
    load_split_indices,
    set_seed,
)


def average_gradient_signal(model, loader, device, criterion, sign, max_batches=None):
    gradients = OrderedDict((name, torch.zeros_like(param, device=device)) for name, param in model.named_parameters())
    model.eval()

    for batch_idx, (inputs, targets) in enumerate(tqdm(loader, desc="Grad Signal", leave=False), start=1):
        if max_batches is not None and batch_idx > max_batches:
            break
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        model.zero_grad(set_to_none=True)
        outputs, _ = model(inputs)
        loss = sign * criterion(outputs, targets)
        loss.backward()

        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is not None:
                    gradients[name] += param.grad.detach()

    scores = {}
    param_sizes = {}
    with torch.no_grad():
        for name, grad in gradients.items():
            numel = grad.numel()
            scores[name] = torch.norm(grad.abs(), p=2).item() / max(numel, 1)
            param_sizes[name] = numel
    return scores, param_sizes


def build_mask(retain_scores, forget_scores, param_sizes, mask_ratio, retain_ratio):
    total_params = sum(param_sizes.values())

    sorted_retain = sorted(retain_scores.items(), key=lambda item: item[1], reverse=True)
    retain_threshold = int(total_params * retain_ratio)

    filtered_forget = {}
    covered = 0
    for name, _ in sorted_retain:
        if covered < retain_threshold:
            filtered_forget[name] = 0.0
            covered += param_sizes[name]
        else:
            filtered_forget[name] = forget_scores[name]

    sorted_forget = sorted(filtered_forget.items(), key=lambda item: item[1], reverse=True)
    forget_threshold = int(total_params * mask_ratio)

    mask = {}
    selected = 0
    for name, _ in sorted_forget:
        if selected < forget_threshold:
            mask[name] = 1
            selected += param_sizes[name]
        else:
            mask[name] = 0

    return mask, selected / max(total_params, 1), covered / max(total_params, 1)


def main():
    parser = argparse.ArgumentParser(description="Generate MCU parameter mask for CIFAR100 MoE-ViT")
    parser.add_argument(
        "--original-model-path",
        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "model", "moevit_cifar100_best.pth"),
        help="Original pretrained checkpoint used as MCU start point",
    )
    parser.add_argument(
        "--split-dir",
        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "split_results_cifar100_fine"),
    )
    parser.add_argument("--split-mode", default="random", choices=["class", "random"])
    parser.add_argument("--output-dir", default=os.path.join(os.path.dirname(__file__), "masks"))
    parser.add_argument("--output-name", default=None, help="Optional explicit mask filename")
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

    if not os.path.exists(args.original_model_path):
        raise FileNotFoundError(f"Original checkpoint not found: {args.original_model_path}")

    os.makedirs(args.output_dir, exist_ok=True)
    if args.output_name is None:
        output_name = f"mask_{args.split_mode}_k{args.mask_ratio}_kr{args.retain_ratio}.pt"
    else:
        output_name = args.output_name
    output_path = os.path.join(args.output_dir, output_name)

    trainset, trainset_eval, testset = build_cifar100_datasets()
    split_indices = load_split_indices(args.split_dir, args.split_mode)
    subsets = build_split_subsets(trainset, trainset_eval, testset, split_indices)

    retain_loader = DataLoader(
        subsets["retain_train"],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    forget_loader = DataLoader(
        subsets["forget_train"],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = build_model(args, device)
    load_checkpoint_state(model, args.original_model_path, device)
    criterion = nn.CrossEntropyLoss()

    print(f"Using device: {device}")
    print(f"Original model: {args.original_model_path}")
    print(f"Split mode: {args.split_mode}")
    print(f"Retain train size: {len(subsets['retain_train'])}")
    print(f"Forget train size: {len(subsets['forget_train'])}")

    retain_scores, param_sizes = average_gradient_signal(
        model, retain_loader, device, criterion, sign=1.0, max_batches=args.max_batches
    )
    forget_scores, _ = average_gradient_signal(
        model, forget_loader, device, criterion, sign=-1.0, max_batches=args.max_batches
    )

    mask, real_mask_ratio, real_retain_ratio = build_mask(
        retain_scores,
        forget_scores,
        param_sizes,
        args.mask_ratio,
        args.retain_ratio,
    )

    payload = {
        "mask": mask,
        "metadata": {
            "split_mode": args.split_mode,
            "mask_ratio": args.mask_ratio,
            "retain_ratio": args.retain_ratio,
            "real_mask_ratio": real_mask_ratio,
            "real_retain_ratio": real_retain_ratio,
            "original_model_path": args.original_model_path,
            "max_batches": args.max_batches,
            "seed": args.seed,
        },
    }
    torch.save(payload, output_path)

    selected_names = [name for name, flag in mask.items() if flag == 1]
    print(f"Saved mask to: {output_path}")
    print(f"Selected tensors: {len(selected_names)}/{len(mask)}")
    print(f"Real selected parameter ratio: {real_mask_ratio:.6f}")
    print(f"Real retain-filter ratio: {real_retain_ratio:.6f}")


if __name__ == "__main__":
    main()
