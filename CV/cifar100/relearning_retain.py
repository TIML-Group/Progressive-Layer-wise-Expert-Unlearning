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
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "unnlearn_model", "unlearned_best_class.pth")
# neggrad_plus_best_class
# seuf_neggrad_plus_global_best_class
# unlearned_best_class
# seuf_ga_global_unlearned_best_class
# neus_best_class


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_cifar_datasets():
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


def load_checkpoint_state(model, model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    cleaned_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            cleaned_state_dict[key[len("module."):]] = value
        else:
            cleaned_state_dict[key] = value

    model.load_state_dict(cleaned_state_dict, strict=True)
    return checkpoint


def train_head_only(model):
    for param in model.parameters():
        param.requires_grad = False
    for param in model.head.parameters():
        param.requires_grad = True


def main():
    parser = argparse.ArgumentParser(description="Relearning on CIFAR100 forget set from an unlearned model")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Checkpoint to load before relearning")
    parser.add_argument(
        "--split-dir",
        default=os.path.join(BASE_DIR, "split_results_cifar100_fine"),
        help="Directory containing forget/retain index files",
    )
    parser.add_argument(
        "--split-mode",
        default="class",
        choices=["class", "random"],
        help="Which forget/retain split to use for retain relearning and final metrics",
    )
    parser.add_argument("--lr", default=1e-2, type=float)
    parser.add_argument("--weight-decay", default=0.01, type=float)
    parser.add_argument("--train-steps", default=10, type=int)
    parser.add_argument("--train-batch-size", default=16, type=int)
    parser.add_argument("--eval-batch-size", default=128, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--image-size", default=32, type=int)
    parser.add_argument("--num-classes", default=100, type=int)
    parser.add_argument("--embed-dim", default=256, type=int)
    parser.add_argument("--depth", default=12, type=int)
    parser.add_argument("--num-heads", default=8, type=int)
    parser.add_argument("--label-smoothing", default=0.05, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--eval-original",
        type=int,
        default=0,
        choices=[0, 1],
        help="Whether to evaluate the original checkpoint on the 4 CIFAR100 metrics before relearning.",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Model path: {args.model_path}")
    print(f"Split mode: {args.split_mode}")

    required_split_files = [
        os.path.join(args.split_dir, f"forget_indices_{args.split_mode}_train.npy"),
        os.path.join(args.split_dir, f"retain_indices_{args.split_mode}_train.npy"),
        os.path.join(args.split_dir, f"forget_indices_{args.split_mode}_test.npy"),
        os.path.join(args.split_dir, f"retain_indices_{args.split_mode}_test.npy"),
    ]
    missing_split_files = [path for path in required_split_files if not os.path.exists(path)]
    if missing_split_files:
        raise FileNotFoundError("Missing split files:\n" + "\n".join(missing_split_files))

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {args.model_path}")

    trainset, trainset_eval, testset = build_cifar_datasets()

    forget_train_idx = np.load(required_split_files[0])
    retain_train_idx = np.load(required_split_files[1])
    forget_test_idx = np.load(required_split_files[2])
    retain_test_idx = np.load(required_split_files[3])

    print("Loaded split files:")
    for path in required_split_files:
        print(f"  {path}")

    forget_train = Subset(trainset, forget_train_idx)
    forget_train_eval = Subset(trainset_eval, forget_train_idx)
    retain_train_eval = Subset(trainset_eval, retain_train_idx)
    forget_test = Subset(testset, forget_test_idx)
    retain_test = Subset(testset, retain_test_idx)

    retain_train_loader = DataLoader(
        forget_train,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    forget_train_eval_loader = DataLoader(
        forget_train_eval,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    retain_train_eval_loader = DataLoader(
        retain_train_eval,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    forget_test_loader = DataLoader(
        forget_test,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    retain_test_loader = DataLoader(
        retain_test,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = build_model(args, device)
    checkpoint = load_checkpoint_state(model, args.model_path, device)
    train_head_only(model)
    print("Checkpoint loaded.")
    print("Only the classification head will be updated during relearning.")
    if isinstance(checkpoint, dict):
        print(f"Checkpoint keys: {list(checkpoint.keys())}")

    if args.eval_original == 1:
        print("\n=== Original Metrics On CIFAR100 ===")
        original_forget_train_acc = evaluate(model, forget_train_eval_loader, device, "Original Forget Train")
        original_retain_train_acc = evaluate(model, retain_train_eval_loader, device, "Original Retain Train")
        original_forget_test_acc = evaluate(model, forget_test_loader, device, "Original Forget Test")
        original_retain_test_acc = evaluate(model, retain_test_loader, device, "Original Retain Test")
        original_score = (
            original_retain_train_acc
            + original_retain_test_acc
            - original_forget_train_acc
            - original_forget_test_acc
        )
        print(f"ForgetAcc(train): {original_forget_train_acc:.2f}%")
        print(f"RA(train): {original_retain_train_acc:.2f}%")
        print(f"ForgetAcc(test): {original_forget_test_acc:.2f}%")
        print(f"RA(test): {original_retain_test_acc:.2f}%")
        print(f"Score: {original_score:.2f}")

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.train_steps, 1))

    start_time = time.time()
    model.train()
    total_loss = 0.0
    total_samples = 0
    data_iter = iter(retain_train_loader)
    pbar = tqdm(range(1, args.train_steps + 1), desc=f"Forget Relearning {args.train_steps} steps")
    for step in pbar:
        current_lr = optimizer.param_groups[0]["lr"]

        try:
            inputs, targets = next(data_iter)
        except StopIteration:
            data_iter = iter(retain_train_loader)
            inputs, targets = next(data_iter)

        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs, _ = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)
        pbar.set_postfix({
            "loss": f"{total_loss / max(total_samples, 1):.4f}",
            "lr": f"{current_lr:.6f}",
        })

    print(
        f"Forget-set relearning finished | steps={args.train_steps} | "
        f"train_loss={total_loss / max(total_samples, 1):.4f}"
    )

    forget_train_acc = evaluate(model, forget_train_eval_loader, device, "Forget Train")
    retain_train_acc = evaluate(model, retain_train_eval_loader, device, "Retain Train")
    forget_test_acc = evaluate(model, forget_test_loader, device, "Forget Test")
    retain_test_acc = evaluate(model, retain_test_loader, device, "Retain Test")
    score = retain_train_acc + retain_test_acc - forget_train_acc - forget_test_acc

    total_minutes = (time.time() - start_time) / 60.0
    print("\n=== Forget-Set Relearning Metrics On CIFAR100 ===")
    print(f"ForgetAcc(train): {forget_train_acc:.2f}%")
    print(f"RA(train): {retain_train_acc:.2f}%")
    print(f"ForgetAcc(test): {forget_test_acc:.2f}%")
    print(f"RA(test): {retain_test_acc:.2f}%")
    print(f"Score: {score:.2f}")
    print(f"RTE: {total_minutes:.2f} minutes")


if __name__ == "__main__":
    main()
