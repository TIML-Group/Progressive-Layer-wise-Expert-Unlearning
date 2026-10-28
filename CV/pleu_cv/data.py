"""Datasets, forget/retain splits and the benign relearning set."""

import json
import os
import random
from pathlib import Path

import numpy as np
import torchvision
import torchvision.transforms as transforms
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset, Subset

ImageFile.LOAD_TRUNCATED_IMAGES = True

CV_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = Path(os.environ.get("PLEU_CV_DATA", CV_DIR / "data"))
DEFAULT_OUTPUT_ROOT = Path(os.environ.get("PLEU_CV_OUTPUT", CV_DIR / "outputs"))
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_MODES = ("class", "random")
SPLIT_PARTS = ("forget_train", "retain_train", "forget_test", "retain_test")

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Architecture, training recipe and the RT forget accuracy that random-split
# checkpoint selection aims for (Table 2, RT row).
DATASET_CONFIGS = {
    "cifar100": {
        "num_classes": 100,
        "image_size": 32,
        "patch_size": 4,
        "embed_dim": 256,
        "depth": 12,
        "num_heads": 8,
        "train": {
            "epochs": 250, "warmup_epochs": 15, "lr": 1e-3, "weight_decay": 0.01,
            "batch_size": 128, "label_smoothing": 0.05, "cutmix_alpha": 1.0,
            "mixup_alpha": 0.0, "aux_weight": 0.01, "ema_decay": 0.0, "amp": False,
        },
        "rt_forget_train_acc": 65.70,
    },
    "imagenet100": {
        "num_classes": 100,
        "image_size": 224,
        "patch_size": 16,
        "embed_dim": 256,
        "depth": 12,
        "num_heads": 8,
        "train": {
            "epochs": 200, "warmup_epochs": 10, "lr": 8e-4, "weight_decay": 0.05,
            "batch_size": 32, "label_smoothing": 0.1, "cutmix_alpha": 0.0,
            "mixup_alpha": 0.2, "aux_weight": 0.02, "ema_decay": 0.9998, "amp": True,
        },
        "rt_forget_train_acc": 88.01,
    },
}


def build_transforms(dataset_name, image_size=None):
    if dataset_name == "cifar100":
        train = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=7),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
        ])
        evaluation = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ])
        return train, evaluation

    size = image_size or DATASET_CONFIGS[dataset_name]["image_size"]
    train = transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
    ])
    evaluation = transforms.Compose([
        transforms.Resize(int(size * 256 / 224)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train, evaluation


class ImageListDataset(Dataset):
    """(path, label) list dataset; exposes ``targets`` like torchvision datasets."""

    def __init__(self, samples, transform):
        self.samples = [(str(path), int(label)) for path, label in samples]
        self.targets = [label for _, label in self.samples]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


# ---------------------------------------------------------------- ImageNet-100

def discover_image_folder(root):
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Image folder root does not exist: {root}")
    class_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    if not class_dirs:
        raise RuntimeError(f"No class directories found under {root}")
    class_to_idx = {path.name: idx for idx, path in enumerate(class_dirs)}
    samples = []
    for class_dir in class_dirs:
        images = sorted(
            path for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        samples.extend((path, class_to_idx[class_dir.name]) for path in images)
    return samples, class_to_idx


def split_per_class(samples, class_to_idx, test_ratio, seed):
    """Per-class train/test split; identical to the original ImageNet-100 training script."""
    by_label = {label: [] for label in class_to_idx.values()}
    for sample in samples:
        by_label[sample[1]].append(sample)
    train, test = [], []
    for label in sorted(by_label):
        items = list(by_label[label])
        if len(items) < 2:
            raise ValueError(f"Class {label} needs at least two images, found {len(items)}")
        random.Random(seed + label).shuffle(items)
        count = max(1, min(len(items) - 1, int(round(len(items) * test_ratio))))
        test.extend(items[:count])
        train.extend(items[count:])
    key = lambda item: (item[1], str(item[0]))  # noqa: E731
    return sorted(train, key=key), sorted(test, key=key)


def imagenet100_split_json(output_root):
    return Path(output_root) / "imagenet100" / "splits" / "imagenet100_split.json"


def write_imagenet100_split_json(path, data_root, class_to_idx, train, test, test_ratio, seed):
    data_root = Path(data_root)

    def serialize(items):
        return [{"path": str(Path(p).relative_to(data_root)), "label": int(label)} for p, label in items]

    payload = {
        "data_root": str(data_root),
        "class_to_idx": class_to_idx,
        "num_classes": len(class_to_idx),
        "val_ratio": test_ratio,
        "seed": seed,
        "train": serialize(train),
        "val": serialize(test),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_imagenet100_samples(split_json, data_root=None):
    """Read the train/test lists. Also accepts ``imagenet100_classwise_split.json``
    written by the original ImageNet-100 training script."""
    payload = json.loads(Path(split_json).read_text(encoding="utf-8"))
    root = Path(data_root) if data_root is not None else Path(payload["data_root"])

    def deserialize(items):
        return [(root / item["path"], item["label"]) for item in items]

    return deserialize(payload["train"]), deserialize(payload["val"]), payload


# ---------------------------------------------------------------- dataset API

def imagenet100_root(data_root):
    return Path(data_root) / "imagenet100"


def relearn_root(data_root):
    return Path(data_root) / "data_re"


def build_datasets(dataset_name, data_root=DEFAULT_DATA_ROOT, output_root=DEFAULT_OUTPUT_ROOT, image_size=None):
    """Return (train with augmentation, train with eval transform, test, train transform)."""
    train_tf, eval_tf = build_transforms(dataset_name, image_size)
    if dataset_name == "cifar100":
        kwargs = {"root": str(data_root), "download": True}
        trainset = torchvision.datasets.CIFAR100(train=True, transform=train_tf, **kwargs)
        trainset_eval = torchvision.datasets.CIFAR100(train=True, transform=eval_tf, **kwargs)
        testset = torchvision.datasets.CIFAR100(train=False, transform=eval_tf, **kwargs)
        return trainset, trainset_eval, testset, train_tf

    if dataset_name == "imagenet100":
        split_json = imagenet100_split_json(output_root)
        if not split_json.exists():
            raise FileNotFoundError(f"{split_json} not found; run prepare_splits.py --dataset imagenet100 first.")
        train, test, _ = load_imagenet100_samples(split_json, imagenet100_root(data_root))
        return (
            ImageListDataset(train, train_tf),
            ImageListDataset(train, eval_tf),
            ImageListDataset(test, eval_tf),
            train_tf,
        )
    raise ValueError(f"Unknown dataset: {dataset_name}")


def split_dir(output_root, dataset_name):
    return Path(output_root) / dataset_name / "splits"


def split_paths(directory, split_mode):
    return {
        "forget_train": Path(directory) / f"forget_indices_{split_mode}_train.npy",
        "retain_train": Path(directory) / f"retain_indices_{split_mode}_train.npy",
        "forget_test": Path(directory) / f"forget_indices_{split_mode}_test.npy",
        "retain_test": Path(directory) / f"retain_indices_{split_mode}_test.npy",
    }


def load_split_indices(directory, split_mode):
    paths = split_paths(directory, split_mode)
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing split files (run prepare_splits.py):\n" + "\n".join(missing))
    return {name: np.load(path) for name, path in paths.items()}


def build_subsets(trainset, trainset_eval, testset, indices):
    return {
        "forget_train": Subset(trainset, indices["forget_train"]),
        "retain_train": Subset(trainset, indices["retain_train"]),
        "forget_train_eval": Subset(trainset_eval, indices["forget_train"]),
        "retain_train_eval": Subset(trainset_eval, indices["retain_train"]),
        "forget_test": Subset(testset, indices["forget_test"]),
        "retain_test": Subset(testset, indices["retain_test"]),
        "full_test": testset,
    }


def make_loader(dataset, batch_size, num_workers, shuffle=False, drop_last=False):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
    )


def build_eval_loaders(subsets, batch_size, num_workers):
    names = ("forget_train_eval", "retain_train_eval", "forget_test", "retain_test", "full_test")
    return {name: make_loader(subsets[name], batch_size, num_workers) for name in names}


def fixed_fraction(indices, fraction, seed):
    """Deterministic subset of ``indices`` (used for the 50% retain data in MCU/PLEU)."""
    indices = np.asarray(indices)
    if fraction >= 1.0:
        return indices
    count = max(1, int(round(len(indices) * fraction)))
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(indices, size=count, replace=False))


def build_relearn_dataset(data_root, train_transform):
    root = relearn_root(data_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Relearning set not found at {root}; run build_relearn_set.py first.")
    return torchvision.datasets.ImageFolder(str(root), transform=train_transform)
