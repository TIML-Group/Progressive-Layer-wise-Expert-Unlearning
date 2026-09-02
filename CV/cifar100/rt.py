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
    trainset = torchvision.datasets.CIFAR100(
        root=DATA_ROOT,
        train=True,
        download=True,
        transform=transform_train,
    )
    trainset_eval = torchvision.datasets.CIFAR100(
        root=DATA_ROOT,
        train=True,
        download=True,
        transform=transform_eval,
    )
    testset = torchvision.datasets.CIFAR100(
        root=DATA_ROOT,
        train=False,
        download=True,
        transform=transform_eval,
    )
    return trainset, trainset_eval, testset


def rand_bbox(size, lam):
    height = size[2]
    width = size[3]
    cut_ratio = np.sqrt(1.0 - lam)
    cut_w = int(width * cut_ratio)
    cut_h = int(height * cut_ratio)

    cx = np.random.randint(width)
    cy = np.random.randint(height)

    x1 = np.clip(cx - cut_w // 2, 0, width)
    y1 = np.clip(cy - cut_h // 2, 0, height)
    x2 = np.clip(cx + cut_w // 2, 0, width)
    y2 = np.clip(cy + cut_h // 2, 0, height)
    return x1, y1, x2, y2


def cutmix_data(inputs, targets, alpha=1.0):
    if alpha <= 0.0:
        return inputs, targets, targets, 1.0

    lam = np.random.beta(alpha, alpha)
    batch_size = inputs.size(0)
    index = torch.randperm(batch_size, device=inputs.device)
    targets_a = targets
    targets_b = targets[index]

    x1, y1, x2, y2 = rand_bbox(inputs.size(), lam)
    mixed_inputs = inputs.clone()
    mixed_inputs[:, :, y1:y2, x1:x2] = inputs[index, :, y1:y2, x1:x2]

    box_area = (x2 - x1) * (y2 - y1)
    lam = 1.0 - box_area / float(inputs.size(2) * inputs.size(3))

    return mixed_inputs, targets_a, targets_b, lam


def cutmix_criterion(criterion_fn, outputs, targets_a, targets_b, lam):
    return lam * criterion_fn(outputs, targets_a) + (1 - lam) * criterion_fn(outputs, targets_b)


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


def build_model(args, device):
    return MoEVisionTransformer(
        img_size=args.image_size,
        patch_size=4,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        num_classes=args.num_classes,
    ).to(device)


def main():
    parser = argparse.ArgumentParser(description="Retrain on retain set for CIFAR100 unlearning baseline")
    parser.add_argument(
        "--split-dir",
        default=os.path.join(BASE_DIR, "split_results_cifar100_fine"),
        help="Directory containing forget/retain index files",
    )
    parser.add_argument(
        "--split-mode",
        default="random",
        choices=["class", "random"],
        help="Which forget/retain split to use",
    )
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--weight-decay", default=0.01, type=float)
    parser.add_argument("--epochs", default=250, type=int)
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--image-size", default=32, type=int)
    parser.add_argument("--num-classes", default=100, type=int)
    parser.add_argument("--embed-dim", default=256, type=int)
    parser.add_argument("--depth", default=12, type=int)
    parser.add_argument("--num-heads", default=8, type=int)
    parser.add_argument("--label-smoothing", default=0.05, type=float)
    parser.add_argument("--cutmix-alpha", default=1.0, type=float)
    parser.add_argument("--warmup-epochs", default=15, type=int)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Using split mode: {args.split_mode}")

    required_split_files = [
        os.path.join(args.split_dir, f"forget_indices_{args.split_mode}_train.npy"),
        os.path.join(args.split_dir, f"retain_indices_{args.split_mode}_train.npy"),
        os.path.join(args.split_dir, f"forget_indices_{args.split_mode}_test.npy"),
        os.path.join(args.split_dir, f"retain_indices_{args.split_mode}_test.npy"),
    ]
    missing_split_files = [path for path in required_split_files if not os.path.exists(path)]
    if missing_split_files:
        raise FileNotFoundError("Missing split files:\n" + "\n".join(missing_split_files))

    trainset, trainset_eval, testset = build_datasets()

    forget_train_idx = np.load(required_split_files[0])
    retain_train_idx = np.load(required_split_files[1])
    forget_test_idx = np.load(required_split_files[2])
    retain_test_idx = np.load(required_split_files[3])

    retain_train = Subset(trainset, retain_train_idx)
    forget_train_eval = Subset(trainset_eval, forget_train_idx)
    retain_train_eval = Subset(trainset_eval, retain_train_idx)
    forget_test = Subset(testset, forget_test_idx)
    retain_test = Subset(testset, retain_test_idx)

    retain_train_loader = DataLoader(
        retain_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    forget_train_eval_loader = DataLoader(
        forget_train_eval,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    retain_train_eval_loader = DataLoader(
        retain_train_eval,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    forget_test_loader = DataLoader(
        forget_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    retain_test_loader = DataLoader(
        retain_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = build_model(args, device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    cosine_epochs = max(args.epochs - args.warmup_epochs, 1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_epochs)

    def compute_metrics():
        forget_train_acc = evaluate(model, forget_train_eval_loader, device, "Forget Train")
        ra_train = evaluate(model, retain_train_eval_loader, device, "Retain Train")
        forget_test_acc = evaluate(model, forget_test_loader, device, "Forget Test")
        ra_test = evaluate(model, retain_test_loader, device, "Retain Test")
        score = ra_train + ra_test - forget_train_acc - forget_test_acc
        return {
            "forget_train_acc": forget_train_acc,
            "ra_train": ra_train,
            "forget_test_acc": forget_test_acc,
            "ra_test": ra_test,
            "score": score,
        }

    start_time = time.time()
    best_score = float("-inf")
    best_metrics = None
    best_epoch = 0

    for epoch in range(args.epochs):
        if epoch < args.warmup_epochs:
            warmup_factor = float(epoch + 1) / float(max(args.warmup_epochs, 1))
            for param_group in optimizer.param_groups:
                param_group["lr"] = args.lr * warmup_factor

        model.train()
        total_loss = 0.0
        pbar = tqdm(retain_train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for step, (inputs, targets) in enumerate(pbar, start=1):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            mixed_inputs, targets_a, targets_b, lam = cutmix_data(inputs, targets, args.cutmix_alpha)
            outputs, _ = model(mixed_inputs)
            loss = cutmix_criterion(criterion, outputs, targets_a, targets_b, lam)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{total_loss / step:.4f}"})

        if epoch >= args.warmup_epochs:
            scheduler.step()
        metrics = compute_metrics()
        if metrics["score"] > best_score:
            best_score = metrics["score"]
            best_metrics = dict(metrics)
            best_epoch = epoch + 1

    total_minutes = (time.time() - start_time) / 60.0
    if best_metrics is None:
        raise RuntimeError("No checkpoint was saved during retraining.")

    print("\n=== Best Metrics ===")
    print(
        f"Epoch: {best_epoch} | "
        f"ForgetAcc(train): {best_metrics['forget_train_acc']:.2f}% | "
        f"RA(train): {best_metrics['ra_train']:.2f}% | "
        f"ForgetAcc(test): {best_metrics['forget_test_acc']:.2f}% | "
        f"RA(test): {best_metrics['ra_test']:.2f}% | "
        f"Score: {best_metrics['score']:.2f}"
    )
    print(f"RTE: {total_minutes:.2f} minutes")


if __name__ == "__main__":
    main()
