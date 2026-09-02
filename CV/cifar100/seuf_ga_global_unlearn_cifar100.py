import argparse
import os
import random
import sys
import time

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
RANDOM_TARGET_METRICS = {
    "forget_train_acc": 70.0,
    "ra_train": 99.28,
    "forget_test_acc": 66.2,
    "ra_test": 67.39,
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
    transform_eval = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    trainset = torchvision.datasets.CIFAR100(root=DATA_ROOT, train=True, download=True, transform=transform_train)
    trainset_eval = torchvision.datasets.CIFAR100(root=DATA_ROOT, train=True, download=True, transform=transform_eval)
    testset = torchvision.datasets.CIFAR100(root=DATA_ROOT, train=False, download=True, transform=transform_eval)
    return trainset, trainset_eval, testset


def evaluate(model, loader, device, desc):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        pbar = tqdm(loader, desc=desc, leave=False)
        for inputs, targets in pbar:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
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


def compute_patch_majority_experts(gate_probs):
    patch_probs = gate_probs[:, 1:, :]
    if patch_probs.size(1) == 0:
        patch_probs = gate_probs

    patch_top1 = patch_probs.argmax(dim=-1)
    num_experts = gate_probs.size(-1)
    expert_votes = torch.stack(
        [(patch_top1 == expert_idx).sum(dim=1) for expert_idx in range(num_experts)],
        dim=1,
    )
    return expert_votes.argmax(dim=1), expert_votes


def compute_global_target_expert(model, dataloader, device):
    moe_layers = []
    for layer_idx, blk in enumerate(model.blocks):
        if getattr(blk, "use_moe", False) and hasattr(blk.mlp, "experts"):
            moe_layers.append((layer_idx, len(blk.mlp.experts)))

    patch_softmax_sum = {layer_idx: torch.zeros(num_experts, device=device) for layer_idx, num_experts in moe_layers}
    patch_top1_count = {layer_idx: torch.zeros(num_experts, device=device) for layer_idx, num_experts in moe_layers}
    sample_top1_count = {layer_idx: torch.zeros(num_experts, device=device) for layer_idx, num_experts in moe_layers}
    total_patch_tokens = {layer_idx: 0 for layer_idx, _ in moe_layers}
    total_samples = {layer_idx: 0 for layer_idx, _ in moe_layers}

    model.eval()
    with torch.no_grad():
        for inputs, _ in tqdm(dataloader, desc="Computing global target expert", leave=False):
            inputs = inputs.to(device, non_blocking=True)
            layer_gate_probs = collect_moe_gate_probs(model, inputs)
            for layer_idx, gate_probs in layer_gate_probs.items():
                patch_probs = gate_probs[:, 1:, :]
                if patch_probs.size(1) == 0:
                    patch_probs = gate_probs
                patch_softmax_sum[layer_idx] += patch_probs.sum(dim=(0, 1))
                patch_top1_idx = torch.argmax(patch_probs, dim=-1)
                sample_experts, _ = compute_patch_majority_experts(gate_probs)
                num_experts = gate_probs.size(-1)
                for expert_id in range(num_experts):
                    patch_top1_count[layer_idx][expert_id] += (patch_top1_idx == expert_id).sum().float()
                    sample_top1_count[layer_idx][expert_id] += (sample_experts == expert_id).sum().float()
                total_patch_tokens[layer_idx] += patch_probs.size(0) * patch_probs.size(1)
                total_samples[layer_idx] += patch_probs.size(0)

    best = None
    per_layer_stats = {}
    for layer_idx, _ in moe_layers:
        patch_softmax_avg = (patch_softmax_sum[layer_idx] / total_patch_tokens[layer_idx]).cpu().numpy()
        patch_top1_ratio = (patch_top1_count[layer_idx] / total_patch_tokens[layer_idx]).cpu().numpy()
        sample_top1_ratio = (sample_top1_count[layer_idx] / total_samples[layer_idx]).cpu().numpy()
        per_layer_stats[layer_idx] = {
            "patch_softmax_avg": patch_softmax_avg.tolist(),
            "patch_top1_ratio": patch_top1_ratio.tolist(),
            "sample_top1_ratio": sample_top1_ratio.tolist(),
            "routing_unit": "patch_majority",
        }
        for expert_idx, score in enumerate(sample_top1_ratio):
            candidate = {
                "layer_idx": layer_idx,
                "target_idx": int(expert_idx),
                "top1_ratio": float(score),
                "sample_top1_ratio": float(score),
                "patch_top1_ratio": float(patch_top1_ratio[expert_idx]),
                "patch_softmax_avg": float(patch_softmax_avg[expert_idx]),
                "routing_unit": "patch_majority",
            }
            if best is None or candidate["top1_ratio"] > best["top1_ratio"]:
                best = candidate

    if best is None:
        raise RuntimeError("No MoE expert found in model.")
    return best, per_layer_stats


def freeze_to_global_target_expert(model, target_layer_idx, target_expert_idx):
    for param in model.parameters():
        param.requires_grad = False

    target_block = model.blocks[target_layer_idx]
    for param in target_block.mlp.experts[target_expert_idx].parameters():
        param.requires_grad = True
    for param in target_block.mlp.gate.parameters():
        param.requires_grad = True


def compute_anchor_loss(layer_gate_probs, target_layer_idx, target_expert_idx, device):
    gate_probs = layer_gate_probs.get(target_layer_idx)
    if gate_probs is None:
        return torch.tensor(0.0, device=device)
    anchor = torch.zeros_like(gate_probs, device=device)
    anchor[..., target_expert_idx] = 1.0
    return torch.norm(gate_probs - anchor, p=2, dim=-1).mean()


def build_model(args, device):
    return MoEVisionTransformer(
        img_size=args.image_size,
        patch_size=4,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        num_classes=args.num_classes,
    ).to(device)


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)


def save_best(model, output_path, epoch, metrics, global_target, per_layer_stats):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "metrics": metrics,
            "global_target": global_target,
            "per_layer_stats": per_layer_stats,
        },
        output_path,
    )


def compute_selection_score(metrics, split_mode):
    if split_mode == "random":
        return abs(metrics["forget_train_acc"] - RANDOM_TARGET_METRICS["forget_train_acc"])
    return -(metrics["ra_train"] + metrics["ra_test"] - metrics["forget_train_acc"] - metrics["forget_test_acc"])


def main():
    parser = argparse.ArgumentParser(description="Global-single-expert SEUF-GA on CIFAR100 fine-class split")
    parser.add_argument("--checkpoint-path", default=os.path.join(BASE_DIR, "model", "moevit_cifar100_best.pth"))
    parser.add_argument("--split-dir", default=os.path.join(BASE_DIR, "split_results_cifar100_fine"))
    parser.add_argument("--output-dir", default=os.path.join(BASE_DIR, "unnlearn_model"))
    parser.add_argument(
        "--split-mode",
        default="random",
        choices=["class", "random"],
        help="Which forget/retain split to use",
    )
    parser.add_argument("--lr", default=5e-4, type=float)
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--anchor-weight", default=0.1, type=float)
    parser.add_argument("--image-size", default=32, type=int)
    parser.add_argument("--num-classes", default=100, type=int)
    parser.add_argument("--embed-dim", default=256, type=int)
    parser.add_argument("--depth", default=12, type=int)
    parser.add_argument("--num-heads", default=8, type=int)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"seuf_ga_global_unlearned_best_{args.split_mode}.pth")

    required_split_files = [
        os.path.join(args.split_dir, f"forget_indices_{args.split_mode}_train.npy"),
        os.path.join(args.split_dir, f"retain_indices_{args.split_mode}_train.npy"),
        os.path.join(args.split_dir, f"forget_indices_{args.split_mode}_test.npy"),
        os.path.join(args.split_dir, f"retain_indices_{args.split_mode}_test.npy"),
    ]
    missing_split_files = [path for path in required_split_files if not os.path.exists(path)]
    if missing_split_files:
        raise FileNotFoundError("Missing split files:\n" + "\n".join(missing_split_files))
    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")
    print(f"Using split mode: {args.split_mode}")

    trainset, trainset_eval, testset = build_datasets()
    forget_train_idx = np.load(required_split_files[0])
    retain_train_idx = np.load(required_split_files[1])
    forget_test_idx = np.load(required_split_files[2])
    retain_test_idx = np.load(required_split_files[3])

    forget_train = Subset(trainset, forget_train_idx)
    forget_train_eval = Subset(trainset_eval, forget_train_idx)
    retain_train_eval = Subset(trainset_eval, retain_train_idx)
    forget_test = Subset(testset, forget_test_idx)
    retain_test = Subset(testset, retain_test_idx)

    forget_loader = DataLoader(forget_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    forget_target_loader = DataLoader(forget_train_eval, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    forget_train_eval_loader = DataLoader(forget_train_eval, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    retain_train_eval_loader = DataLoader(retain_train_eval, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    forget_test_loader = DataLoader(forget_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    retain_test_loader = DataLoader(retain_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    full_test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = build_model(args, device)
    load_checkpoint(model, args.checkpoint_path, device)
    print(f"Loaded checkpoint: {args.checkpoint_path}")

    global_target, per_layer_stats = compute_global_target_expert(model, forget_target_loader, device)
    print(
        f"Global target expert: layer={global_target['layer_idx']}, "
        f"expert={global_target['target_idx']}, "
        f"top1_ratio={global_target['top1_ratio']:.4f}, "
        f"patch_top1_ratio={global_target['patch_top1_ratio']:.4f}, "
        f"patch_softmax_avg={global_target['patch_softmax_avg']:.4f}"
    )

    freeze_to_global_target_expert(model, global_target["layer_idx"], global_target["target_idx"])
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters after freezing: {trainable_params}")

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    def compute_metrics():
        forget_train_acc = evaluate(model, forget_train_eval_loader, device, "Forget Train")
        ra_train = evaluate(model, retain_train_eval_loader, device, "Retain Train")
        forget_test_acc = evaluate(model, forget_test_loader, device, "Forget Test")
        ra_test = evaluate(model, retain_test_loader, device, "Retain Test")
        ta = evaluate(model, full_test_loader, device, "Full Test")
        score = ra_train + ra_test - forget_train_acc - forget_test_acc
        return {
            "forget_train_acc": forget_train_acc,
            "ra_train": ra_train,
            "forget_test_acc": forget_test_acc,
            "ra_test": ra_test,
            "ta": ta,
            "score": score,
        }

    start_time = time.time()
    best_score = float("inf")
    best_metrics = None
    best_epoch = 0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        pbar = tqdm(forget_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for step, (inputs_f, targets_f) in enumerate(pbar, start=1):
            inputs_f = inputs_f.to(device, non_blocking=True)
            targets_f = targets_f.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs_f, _ = model(inputs_f)
            loss_forget = criterion(outputs_f, targets_f)
            layer_gate_probs = collect_moe_gate_probs(model, inputs_f)
            loss_anchor = compute_anchor_loss(
                layer_gate_probs,
                target_layer_idx=global_target["layer_idx"],
                target_expert_idx=global_target["target_idx"],
                device=device,
            )
            loss = -loss_forget + args.anchor_weight * loss_anchor
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{total_loss / step:.4f}"})

        metrics = compute_metrics()
        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"ForgetAcc(train): {metrics['forget_train_acc']:.2f}% | "
            f"RA(train): {metrics['ra_train']:.2f}% | "
            f"ForgetAcc(test): {metrics['forget_test_acc']:.2f}% | "
            f"RA(test): {metrics['ra_test']:.2f}%"
        )
        selection_score = compute_selection_score(metrics, args.split_mode)
        if selection_score < best_score:
            best_score = selection_score
            best_metrics = dict(metrics)
            best_metrics["selection_score"] = selection_score
            best_epoch = epoch + 1
            save_best(model, output_path, best_epoch, best_metrics, global_target, per_layer_stats)

    total_minutes = (time.time() - start_time) / 60.0
    if best_metrics is None:
        raise RuntimeError("No checkpoint was saved during unlearning.")

    print("\n=== Best Metrics ===")
    print(
        f"Epoch: {best_epoch} | "
        f"ForgetAcc(train): {best_metrics['forget_train_acc']:.2f}% | "
        f"RA(train): {best_metrics['ra_train']:.2f}% | "
        f"ForgetAcc(test): {best_metrics['forget_test_acc']:.2f}% | "
        f"RA(test): {best_metrics['ra_test']:.2f}% | "
        f"TA: {best_metrics['ta']:.2f}% | "
        f"Score: {best_metrics['score']:.2f} | "
        f"BestSel: {best_metrics['selection_score']:.2f}"
    )
    print(f"RTE: {total_minutes:.2f} minutes")
    print(f"Saved best model to: {output_path}")


if __name__ == "__main__":
    main()
