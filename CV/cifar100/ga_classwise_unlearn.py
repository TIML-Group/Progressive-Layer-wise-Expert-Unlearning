import argparse
import os
import sys
import time
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CV_DIR = os.path.dirname(BASE_DIR)
if CV_DIR not in sys.path:
    sys.path.insert(0, CV_DIR)

from try_moevit import MoEVisionTransformer  # noqa: E402


DATA_ROOT = "/root/autodl-tmp/SEUF_MCU/data_cv"


def compute_mia(net, forget_loader, retain_loader, device):
    net.eval()
    forget_confs = []
    retain_confs = []
    with torch.no_grad():
        for inputs, _ in forget_loader:
            inputs = inputs.to(device)
            outputs, _ = net(inputs)
            confs = torch.softmax(outputs, dim=1).max(dim=1).values
            forget_confs.extend(confs.cpu().numpy())
        for inputs, _ in retain_loader:
            inputs = inputs.to(device)
            outputs, _ = net(inputs)
            confs = torch.softmax(outputs, dim=1).max(dim=1).values
            retain_confs.extend(confs.cpu().numpy())
    return abs(np.mean(forget_confs) - np.mean(retain_confs))


def build_datasets():
    transform_train = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=7),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    trainset = torchvision.datasets.CIFAR100(
        root=DATA_ROOT,
        train=True,
        download=True,
        transform=transform_train,
    )
    testset = torchvision.datasets.CIFAR100(
        root=DATA_ROOT,
        train=False,
        download=True,
        transform=transform_test,
    )
    return trainset, testset


def evaluate(model, loader, device, desc):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        pbar = tqdm(loader, desc=desc, leave=False)
        for inputs, targets in pbar:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs, _ = model(inputs)
            preds = outputs.argmax(dim=1)
            total += targets.size(0)
            correct += preds.eq(targets).sum().item()
            pbar.set_postfix({"acc": f"{100.0 * correct / max(total, 1):.2f}%"})
    return 100.0 * correct / max(total, 1)


def collect_moe_gate_probs(model, inputs):
    x = model.patch_embed(inputs)
    layer_gate_probs = {}
    for layer_idx, blk in enumerate(model.blocks):
        residual = x
        attn_in = blk.norm1(x)
        attn_out, _ = blk.attn(attn_in, attn_in, attn_in)
        x = residual + attn_out

        residual = x
        mlp_in = blk.norm2(x)
        if getattr(blk, "use_moe", False) and hasattr(blk.mlp, "experts"):
            gate_logits = blk.mlp.gate(mlp_in)
            gate_probs = torch.softmax(gate_logits, dim=-1)
            layer_gate_probs[layer_idx] = gate_probs
            mlp_out, _ = blk.mlp(mlp_in)
        else:
            mlp_out = blk.mlp(mlp_in)
        x = residual + mlp_out
    return layer_gate_probs


def compute_target_experts(model, dataloader, device):
    moe_layers = []
    for layer_idx, blk in enumerate(model.blocks):
        if getattr(blk, "use_moe", False) and hasattr(blk.mlp, "experts"):
            moe_layers.append((layer_idx, len(blk.mlp.experts)))

    expert_softmax_sum = {
        layer_idx: torch.zeros(num_experts, device=device) for layer_idx, num_experts in moe_layers
    }
    expert_top1_count = {
        layer_idx: torch.zeros(num_experts, device=device) for layer_idx, num_experts in moe_layers
    }
    total_tokens = {layer_idx: 0 for layer_idx, _ in moe_layers}

    model.eval()
    with torch.no_grad():
        for inputs, _ in tqdm(dataloader, desc="Computing target experts", leave=False):
            inputs = inputs.to(device)
            layer_gate_probs = collect_moe_gate_probs(model, inputs)
            for layer_idx, gate_probs in layer_gate_probs.items():
                expert_softmax_sum[layer_idx] += gate_probs.sum(dim=(0, 1))
                top1_idx = torch.argmax(gate_probs, dim=-1)
                num_experts = gate_probs.size(-1)
                for expert_id in range(num_experts):
                    expert_top1_count[layer_idx][expert_id] += (top1_idx == expert_id).sum().float()
                total_tokens[layer_idx] += gate_probs.size(0) * gate_probs.size(1)

    target_experts = {}
    for layer_idx, _ in moe_layers:
        softmax_avg = (expert_softmax_sum[layer_idx] / total_tokens[layer_idx]).cpu().numpy()
        top1_ratio = (expert_top1_count[layer_idx] / total_tokens[layer_idx]).cpu().numpy()
        target_idx = int(np.argmax(top1_ratio))
        target_experts[layer_idx] = {
            "target_idx": target_idx,
            "softmax_avg": softmax_avg.tolist(),
            "top1_ratio": top1_ratio.tolist(),
        }
    return target_experts


def freeze_non_target_experts(model, target_experts):
    for layer_idx, blk in enumerate(model.blocks):
        if getattr(blk, "use_moe", False) and hasattr(blk.mlp, "experts"):
            target_idx = target_experts[layer_idx]["target_idx"]
            for expert_idx, expert in enumerate(blk.mlp.experts):
                requires_grad = expert_idx == target_idx
                for param in expert.parameters():
                    param.requires_grad = requires_grad


def compute_anchor_loss(layer_gate_probs, target_experts, device):
    losses = []
    for layer_idx, gate_probs in layer_gate_probs.items():
        target_idx = target_experts[layer_idx]["target_idx"]
        anchor = torch.zeros_like(gate_probs, device=device)
        anchor[..., target_idx] = 1.0
        loss = torch.norm(gate_probs - anchor, p=2, dim=-1).mean()
        losses.append(loss)
    if not losses:
        return torch.tensor(0.0, device=device)
    return torch.stack(losses).mean()


def main():
    parser = argparse.ArgumentParser(description="SEUF-GA unlearning on CIFAR100 fine-class split")
    parser.add_argument(
        "--checkpoint-path",
        default=os.path.join(BASE_DIR, "model", "moevit_cifar100_best.pth"),
        help="Path to the pretrained CIFAR100 MoEVisionTransformer checkpoint",
    )
    parser.add_argument(
        "--split-dir",
        default=os.path.join(BASE_DIR, "split_results_cifar100_fine"),
        help="Directory containing forget/retain index files",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(BASE_DIR, "unnlearn_model"),
        help="Directory to save unlearned checkpoints",
    )
    parser.add_argument("--lr", default=5e-5, type=float)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--anchor-weight", default=0.1, type=float)
    parser.add_argument("--image-size", default=32, type=int)
    parser.add_argument("--num-classes", default=100, type=int)
    parser.add_argument("--embed-dim", default=256, type=int)
    parser.add_argument("--depth", default=12, type=int)
    parser.add_argument("--num-heads", default=8, type=int)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    required_split_files = [
        os.path.join(args.split_dir, "forget_indices_fine_train.npy"),
        os.path.join(args.split_dir, "retain_indices_fine_train.npy"),
        os.path.join(args.split_dir, "forget_indices_fine_test.npy"),
        os.path.join(args.split_dir, "retain_indices_fine_test.npy"),
    ]
    missing_split_files = [path for path in required_split_files if not os.path.exists(path)]
    if missing_split_files:
        raise FileNotFoundError("Missing split files:\n" + "\n".join(missing_split_files))

    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint_path}. "
            "Please place a pretrained CIFAR100 MoEVisionTransformer checkpoint there or override --checkpoint-path."
        )

    trainset, testset = build_datasets()

    forget_train = Subset(trainset, np.load(required_split_files[0]))
    retain_train = Subset(trainset, np.load(required_split_files[1]))
    forget_test = Subset(testset, np.load(required_split_files[2]))
    retain_test = Subset(testset, np.load(required_split_files[3]))

    forget_loader = DataLoader(forget_train, batch_size=args.batch_size, shuffle=True, num_workers=4)
    retain_loader = DataLoader(retain_train, batch_size=args.batch_size, shuffle=True, num_workers=4)
    forget_test_loader = DataLoader(forget_test, batch_size=args.batch_size, shuffle=False, num_workers=4)
    retain_test_loader = DataLoader(retain_test, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = MoEVisionTransformer(
        img_size=args.image_size,
        patch_size=4,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        num_classes=args.num_classes,
    ).to(device)

    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    print(f"Loaded checkpoint: {args.checkpoint_path}")

    target_experts = compute_target_experts(model, forget_loader, device)
    print("=== Target Experts Per MoE Layer ===")
    for layer_idx in sorted(target_experts.keys()):
        info = target_experts[layer_idx]
        softmax_str = ", ".join(f"{x:.4f}" for x in info["softmax_avg"])
        top1_str = ", ".join(f"{x:.4f}" for x in info["top1_ratio"])
        print(
            f"Layer {layer_idx} | Softmax Avg: {softmax_str} | "
            f"Top-1 Ratio: {top1_str} | Target Expert -> {info['target_idx']}"
        )

    freeze_non_target_experts(model, target_experts)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    def compute_metrics():
        forget_train_acc = evaluate(model, forget_loader, device, "Forget Train")
        ra_train = evaluate(model, retain_loader, device, "Retain Train")
        forget_test_acc = evaluate(model, forget_test_loader, device, "Forget Test")
        ra_test = evaluate(model, retain_test_loader, device, "Retain Test")
        ta = evaluate(model, test_loader, device, "Full Test")
        mia = compute_mia(model, forget_loader, retain_loader, device)
        return {
            "forget_train_acc": forget_train_acc,
            "ra_train": ra_train,
            "forget_test_acc": forget_test_acc,
            "ra_test": ra_test,
            "ta": ta,
            "mia": mia,
        }

    initial_metrics = compute_metrics()
    print("\n=== Initial Metrics ===")
    print(
        f"ForgetAcc(train): {initial_metrics['forget_train_acc']:.2f}% | "
        f"RA(train): {initial_metrics['ra_train']:.2f}% | "
        f"ForgetAcc(test): {initial_metrics['forget_test_acc']:.2f}% | "
        f"RA(test): {initial_metrics['ra_test']:.2f}% | "
        f"TA: {initial_metrics['ta']:.2f}% | "
        f"MIA: {initial_metrics['mia']:.4f}"
    )

    best_forget_train_acc = initial_metrics["forget_train_acc"]
    best_ta = initial_metrics["ta"]
    start_time = time.time()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        forget_iter = iter(forget_loader)
        retain_iter = iter(retain_loader)
        max_batches = max(len(forget_loader), len(retain_loader))

        pbar = tqdm(range(max_batches), desc=f"Epoch {epoch + 1}/{args.epochs}")
        for step in pbar:
            optimizer.zero_grad()

            try:
                inputs_f, targets_f = next(forget_iter)
            except StopIteration:
                forget_iter = iter(forget_loader)
                inputs_f, targets_f = next(forget_iter)
            inputs_f = inputs_f.to(device)
            targets_f = targets_f.to(device)
            outputs_f, _ = model(inputs_f)
            loss_forget = criterion(outputs_f, targets_f)

            try:
                inputs_r, targets_r = next(retain_iter)
            except StopIteration:
                retain_iter = iter(retain_loader)
                inputs_r, targets_r = next(retain_iter)
            inputs_r = inputs_r.to(device)
            targets_r = targets_r.to(device)
            outputs_r, _ = model(inputs_r)
            loss_retain = criterion(outputs_r, targets_r)

            layer_gate_probs = collect_moe_gate_probs(model, inputs_f)
            loss_anchor = compute_anchor_loss(layer_gate_probs, target_experts, device)

            loss = loss_retain - loss_forget + args.anchor_weight * loss_anchor
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"avg_loss": f"{total_loss / (step + 1):.4f}"})

        metrics = compute_metrics()
        if metrics["forget_train_acc"] < best_forget_train_acc and metrics["ta"] > initial_metrics["ta"] * 0.9:
            best_forget_train_acc = metrics["forget_train_acc"]
            best_ta = metrics["ta"]
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "metrics": metrics,
                    "target_experts": target_experts,
                },
                os.path.join(args.output_dir, "seuf_ga_unlearned_best.pth"),
            )

        print(
            f"\nEpoch {epoch + 1}: "
            f"ForgetAcc(train)={metrics['forget_train_acc']:.2f}% | "
            f"RA(train)={metrics['ra_train']:.2f}% | "
            f"ForgetAcc(test)={metrics['forget_test_acc']:.2f}% | "
            f"RA(test)={metrics['ra_test']:.2f}% | "
            f"TA={metrics['ta']:.2f}% | "
            f"MIA={metrics['mia']:.4f}"
        )

    final_metrics = compute_metrics()
    total_minutes = (time.time() - start_time) / 60.0
    torch.save(
        {
            "epoch": args.epochs,
            "model_state_dict": model.state_dict(),
            "metrics": final_metrics,
            "target_experts": target_experts,
        },
        os.path.join(args.output_dir, "seuf_ga_unlearned_final.pth"),
    )

    print("\n=== Final Metrics ===")
    print(
        f"ForgetAcc(train): {final_metrics['forget_train_acc']:.2f}% | "
        f"RA(train): {final_metrics['ra_train']:.2f}% | "
        f"ForgetAcc(test): {final_metrics['forget_test_acc']:.2f}% | "
        f"RA(test): {final_metrics['ra_test']:.2f}% | "
        f"TA: {final_metrics['ta']:.2f}% | "
        f"MIA: {final_metrics['mia']:.4f}"
    )
    print(f"Best ForgetAcc(train): {best_forget_train_acc:.2f}% | Best TA: {best_ta:.2f}%")
    print(f"RTE: {total_minutes:.2f} minutes")
    print(f"Saved models to: {args.output_dir}")


if __name__ == "__main__":
    main()
