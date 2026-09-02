import os
import random
import sys

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


MCU_DIR = os.path.dirname(os.path.abspath(__file__))
CIFAR_DIR = os.path.dirname(MCU_DIR)
CV_DIR = os.path.dirname(CIFAR_DIR)
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


def build_cifar100_datasets():
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


def build_model(args, device):
    return MoEVisionTransformer(
        img_size=args.image_size,
        patch_size=4,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        num_classes=args.num_classes,
    ).to(device)


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state" in checkpoint:
            state_dict = checkpoint["model_state"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            cleaned[key[len("module."):]] = value
        else:
            cleaned[key] = value
    return cleaned


def load_checkpoint_state(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = extract_state_dict(checkpoint)
    model.load_state_dict(state_dict, strict=True)
    return checkpoint


def get_split_paths(split_dir, split_mode):
    return {
        "forget_train": os.path.join(split_dir, f"forget_indices_{split_mode}_train.npy"),
        "retain_train": os.path.join(split_dir, f"retain_indices_{split_mode}_train.npy"),
        "forget_test": os.path.join(split_dir, f"forget_indices_{split_mode}_test.npy"),
        "retain_test": os.path.join(split_dir, f"retain_indices_{split_mode}_test.npy"),
    }


def load_split_indices(split_dir, split_mode):
    split_paths = get_split_paths(split_dir, split_mode)
    missing = [path for path in split_paths.values() if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError("Missing split files:\n" + "\n".join(missing))
    return {name: np.load(path) for name, path in split_paths.items()}


def build_split_subsets(trainset, trainset_eval, testset, split_indices):
    return {
        "forget_train": Subset(trainset, split_indices["forget_train"]),
        "retain_train": Subset(trainset, split_indices["retain_train"]),
        "forget_train_eval": Subset(trainset_eval, split_indices["forget_train"]),
        "retain_train_eval": Subset(trainset_eval, split_indices["retain_train"]),
        "forget_test": Subset(testset, split_indices["forget_test"]),
        "retain_test": Subset(testset, split_indices["retain_test"]),
    }


def build_eval_loaders(subsets, testset, eval_batch_size, num_workers):
    return {
        "forget_train_eval": DataLoader(
            subsets["forget_train_eval"],
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "retain_train_eval": DataLoader(
            subsets["retain_train_eval"],
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "forget_test": DataLoader(
            subsets["forget_test"],
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "retain_test": DataLoader(
            subsets["retain_test"],
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "full_test": DataLoader(
            testset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
    }


def _forward_logits(model, inputs, t=None):
    outputs = model(inputs) if t is None else model(inputs, t=t)
    if isinstance(outputs, tuple):
        return outputs[0]
    return outputs


def evaluate(model, loader, device, desc, t=None):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        pbar = tqdm(loader, desc=desc, leave=False)
        for inputs, targets in pbar:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = _forward_logits(model, inputs, t=t)
            preds = logits.argmax(dim=1)
            total += targets.size(0)
            correct += preds.eq(targets).sum().item()
            pbar.set_postfix({"acc": f"{100.0 * correct / max(total, 1):.2f}%"})
    return 100.0 * correct / max(total, 1)


def compute_metrics(model, eval_loaders, device, t=None):
    forget_train_acc = evaluate(model, eval_loaders["forget_train_eval"], device, "Forget Train", t=t)
    ra_train = evaluate(model, eval_loaders["retain_train_eval"], device, "Retain Train", t=t)
    forget_test_acc = evaluate(model, eval_loaders["forget_test"], device, "Forget Test", t=t)
    ra_test = evaluate(model, eval_loaders["retain_test"], device, "Retain Test", t=t)
    ta = evaluate(model, eval_loaders["full_test"], device, "Full Test", t=t)
    score = ra_train + ra_test - forget_train_acc - forget_test_acc
    return {
        "forget_train_acc": forget_train_acc,
        "ra_train": ra_train,
        "forget_test_acc": forget_test_acc,
        "ra_test": ra_test,
        "ta": ta,
        "score": score,
    }


def compute_selection_score(metrics, split_mode):
    if split_mode == "random":
        return abs(metrics["forget_train_acc"] - RANDOM_TARGET_METRICS["forget_train_acc"])
    return -(metrics["ra_train"] + metrics["ra_test"] - metrics["forget_train_acc"] - metrics["forget_test_acc"])
