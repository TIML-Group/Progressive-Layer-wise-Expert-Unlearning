"""Train the origin MoE-ViT model on the local ImageNet-100 directory.

The expected input layout is:

    data_cv/imagenet100/<synset>/*.JPEG

Images are split per class into train and validation subsets without copying
the 14 GB dataset. The split metadata is saved next to the checkpoint so the
same class-wise partition can be reused by later unlearning experiments.
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


ImageFile.LOAD_TRUNCATED_IMAGES = True

SCRIPT_DIR = Path(__file__).resolve().parent
CV_DIR = SCRIPT_DIR.parent
if str(CV_DIR) not in sys.path:
    sys.path.insert(0, str(CV_DIR))

from try_moevit import MoEVisionTransformer  # noqa: E402


DEFAULT_DATA_ROOT = Path("/root/autodl-tmp/SEUF_MCU/data_cv/imagenet100")
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "model"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
Sample = Tuple[Path, int]


def discover_samples(data_root: Path) -> Tuple[List[Sample], Dict[str, int]]:
    """Discover image files and assign deterministic integer labels by class."""
    data_root = Path(data_root)
    if not data_root.is_dir():
        raise FileNotFoundError(f"ImageNet-100 root does not exist: {data_root}")

    class_dirs = sorted(path for path in data_root.iterdir() if path.is_dir())
    if not class_dirs:
        raise RuntimeError(f"No class directories found under: {data_root}")

    class_to_idx = {path.name: index for index, path in enumerate(class_dirs)}
    samples: List[Sample] = []
    for class_dir in class_dirs:
        image_paths = sorted(
            path
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not image_paths:
            raise RuntimeError(f"Class directory contains no supported images: {class_dir}")
        samples.extend((path, class_to_idx[class_dir.name]) for path in image_paths)

    return samples, class_to_idx


def build_classwise_groups(
    class_to_idx: Dict[str, int],
    num_forget_classes: int,
    seed: int,
) -> Tuple[List[str], List[str]]:
    """Select forget/retain classes deterministically for class-wise evaluation."""
    class_names = sorted(class_to_idx)
    if not 1 <= num_forget_classes < len(class_names):
        raise ValueError("num_forget_classes must leave at least one retain class.")

    selected = sorted(random.Random(seed).sample(class_names, num_forget_classes))
    selected_set = set(selected)
    retain = [name for name in class_names if name not in selected_set]
    return selected, retain


def split_samples_by_class(
    samples: Sequence[Sample],
    class_to_idx: Dict[str, int],
    val_ratio: float,
    seed: int,
) -> Tuple[List[Sample], List[Sample]]:
    """Split samples per class, preserving every class in both subsets."""
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1.")

    samples_by_label: Dict[int, List[Sample]] = {label: [] for label in class_to_idx.values()}
    for sample in samples:
        samples_by_label[sample[1]].append(sample)

    train_samples: List[Sample] = []
    val_samples: List[Sample] = []
    for label in sorted(samples_by_label):
        class_samples = list(samples_by_label[label])
        if len(class_samples) < 2:
            raise ValueError(
                f"Class index {label} has only {len(class_samples)} image; "
                "at least two images are required for train/val splitting."
            )

        random.Random(seed + label).shuffle(class_samples)
        val_count = max(1, min(len(class_samples) - 1, int(round(len(class_samples) * val_ratio))))
        val_samples.extend(class_samples[:val_count])
        train_samples.extend(class_samples[val_count:])

    train_samples.sort(key=lambda item: (item[1], str(item[0])))
    val_samples.sort(key=lambda item: (item[1], str(item[0])))
    return train_samples, val_samples


def evaluate_group(model, loader, device: torch.device, class_labels: Sequence[int]):
    model.eval()
    class_labels = set(class_labels)
    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=device.type == "cuda")
            targets = targets.to(device, non_blocking=device.type == "cuda")
            logits, _ = model(inputs)
            keep = torch.tensor(
                [int(target.item()) in class_labels for target in targets],
                dtype=torch.bool,
                device=device,
            )
            if not keep.any():
                continue
            selected_targets = targets[keep]
            total += selected_targets.numel()
            correct += logits[keep].argmax(dim=1).eq(selected_targets).sum().item()
    return 100.0 * correct / max(total, 1), total


def save_split_metadata(
    output_path: Path,
    data_root: Path,
    class_to_idx: Dict[str, int],
    train_samples: Sequence[Sample],
    val_samples: Sequence[Sample],
    val_ratio: float,
    seed: int,
) -> None:
    def serialize(items: Iterable[Sample]):
        return [
            {"path": str(path.relative_to(data_root)), "label": int(label)}
            for path, label in items
        ]

    metadata = {
        "data_root": str(data_root),
        "class_to_idx": class_to_idx,
        "num_classes": len(class_to_idx),
        "val_ratio": val_ratio,
        "seed": seed,
        "train_count": len(train_samples),
        "val_count": len(val_samples),
        "train": serialize(train_samples),
        "val": serialize(val_samples),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


class ImageNet100Dataset(Dataset):
    def __init__(self, samples: Sequence[Sample], transform):
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def build_transforms(image_size: int):
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(int(image_size * 256 / 224)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, eval_transform


def build_model(args, device: torch.device) -> nn.Module:
    return MoEVisionTransformer(
        img_size=args.image_size,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        num_classes=args.num_classes,
    ).to(device)


def mixup(inputs, targets, alpha: float):
    if alpha <= 0:
        return inputs, targets, targets, 1.0
    lam = float(np.random.beta(alpha, alpha))
    permutation = torch.randperm(inputs.size(0), device=inputs.device)
    return inputs, targets, targets[permutation], lam


def run_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    scaler,
    train: bool,
    aux_weight: float,
    mixup_alpha: float = 0.0,
    grad_clip: float = 0.0,
):
    model.train(train)
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    autocast_enabled = device.type == "cuda"

    for inputs, targets in tqdm(loader, leave=False):
        inputs = inputs.to(device, non_blocking=autocast_enabled)
        targets = targets.to(device, non_blocking=autocast_enabled)
        if train:
            optimizer.zero_grad(set_to_none=True)
            inputs, targets_a, targets_b, lam = mixup(inputs, targets, mixup_alpha)
        else:
            targets_a, targets_b, lam = targets, targets, 1.0

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=autocast_enabled):
            logits, aux_loss = model(inputs)
            loss = (
                lam * criterion(logits, targets_a)
                + (1.0 - lam) * criterion(logits, targets_b)
                + aux_weight * aux_loss
            )

        if train:
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.detach().item() * targets.size(0)
        total_correct += logits.detach().argmax(dim=1).eq(targets).sum().item()
        total_count += targets.size(0)

    return total_loss / max(total_count, 1), 100.0 * total_correct / max(total_count, 1)


def evaluate_groups(model, loader, device: torch.device, groups):
    model.eval()
    stats = {name: [0, 0] for name in groups}
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=device.type == "cuda")
            targets = targets.to(device, non_blocking=device.type == "cuda")
            predictions = model(inputs)[0].argmax(dim=1)
            for name, labels in groups.items():
                keep = torch.zeros_like(targets, dtype=torch.bool)
                for label in labels:
                    keep |= targets == label
                stats[name][1] += keep.sum().item()
                stats[name][0] += predictions[keep].eq(targets[keep]).sum().item()
    return {
        name: {
            "correct": correct,
            "total": total,
            "accuracy": 100.0 * correct / max(total, 1),
        }
        for name, (correct, total) in stats.items()
    }


class ModelEMA:
    def __init__(self, model, decay: float = 0.9998):
        self.decay = decay
        self.shadow = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
            if torch.is_floating_point(value)
        }

    @torch.no_grad()
    def update(self, model):
        for name, value in model.state_dict().items():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def copy_to(self, model):
        for name, value in model.state_dict().items():
            if name in self.shadow:
                value.copy_(self.shadow[name])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the ImageNet-100 origin MoE-ViT model.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-classes", type=int, default=100)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--num-forget-classes", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--aux-weight", type=float, default=0.02)
    parser.add_argument("--mixup-alpha", type=float, default=0.2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.9998)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.num_classes <= 0:
        raise ValueError("--num-classes must be positive.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples, class_to_idx = discover_samples(args.data_root)
    if len(class_to_idx) != args.num_classes:
        raise ValueError(
            f"Expected {args.num_classes} classes, found {len(class_to_idx)} under {args.data_root}."
        )
    train_samples, val_samples = split_samples_by_class(
        samples, class_to_idx, args.val_ratio, args.seed
    )
    forget_classes, retain_classes = build_classwise_groups(
        class_to_idx, args.num_forget_classes, args.seed
    )
    forget_labels = [class_to_idx[name] for name in forget_classes]
    retain_labels = [class_to_idx[name] for name in retain_classes]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_metadata_path = args.output_dir / "imagenet100_classwise_split.json"
    save_split_metadata(
        split_metadata_path,
        args.data_root,
        class_to_idx,
        train_samples,
        val_samples,
        args.val_ratio,
        args.seed,
    )
    split_metadata = json.loads(split_metadata_path.read_text(encoding="utf-8"))
    split_metadata.update(
        {
            "forget_classes": forget_classes,
            "retain_classes": retain_classes,
            "forget_labels": forget_labels,
            "retain_labels": retain_labels,
        }
    )
    split_metadata_path.write_text(json.dumps(split_metadata, indent=2), encoding="utf-8")

    train_transform, eval_transform = build_transforms(args.image_size)
    train_loader = DataLoader(
        ImageNet100Dataset(train_samples, train_transform),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    train_eval_loader = DataLoader(
        ImageNet100Dataset(train_samples, eval_transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        ImageNet100Dataset(val_samples, eval_transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    model = build_model(args, device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup_epochs = min(args.warmup_epochs, max(args.epochs - 1, 0))

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / max(warmup_epochs, 1)
        progress = (epoch - warmup_epochs) / max(args.epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    ema = ModelEMA(model, args.ema_decay)

    best_val_acc = -1.0
    start_time = time.time()
    for epoch in range(args.epochs):
        train_loss, train_acc = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler,
            True,
            args.aux_weight,
            args.mixup_alpha,
            args.grad_clip,
        )
        ema.update(model)
        with torch.no_grad():
            val_loss, val_acc = run_epoch(
                model, val_loader, criterion, optimizer, device, scaler, False, args.aux_weight
            )
        eval_model = build_model(args, device)
        eval_model.load_state_dict(model.state_dict())
        ema.copy_to(eval_model)
        train_group_metrics = evaluate_groups(
            eval_model,
            train_eval_loader,
            device,
            {"forget": forget_labels, "retain": retain_labels},
        )
        val_group_metrics = evaluate_groups(
            eval_model,
            val_loader,
            device,
            {"forget": forget_labels, "retain": retain_labels},
        )
        del eval_model
        scheduler.step()

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"train loss={train_loss:.4f} acc={train_acc:.2f}% | "
            f"val loss={val_loss:.4f} acc={val_acc:.2f}% | "
            f"forget train={train_group_metrics['forget']['accuracy']:.2f}% | "
            f"retain train={train_group_metrics['retain']['accuracy']:.2f}% | "
            f"forget test={val_group_metrics['forget']['accuracy']:.2f}% | "
            f"retain test={val_group_metrics['retain']['accuracy']:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = build_model(args, device)
            best_model.load_state_dict(model.state_dict())
            ema.copy_to(best_model)
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": best_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_acc": best_val_acc,
                    "metrics": {
                        "forget_train_acc": train_group_metrics["forget"]["accuracy"],
                        "retain_train_acc": train_group_metrics["retain"]["accuracy"],
                        "forget_test_acc": val_group_metrics["forget"]["accuracy"],
                        "retain_test_acc": val_group_metrics["retain"]["accuracy"],
                    },
                    "forget_classes": forget_classes,
                    "retain_classes": retain_classes,
                    "class_to_idx": class_to_idx,
                    "args": vars(args),
                },
                args.output_dir / "moevit_imagenet100_origin_best.pth",
            )
            del best_model

    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Training time: {(time.time() - start_time) / 60:.2f} minutes")


if __name__ == "__main__":
    main()
