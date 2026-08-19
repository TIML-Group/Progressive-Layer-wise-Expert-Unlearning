import argparse
import os
import random

import numpy as np
import torchvision
import torchvision.transforms as transforms


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = "/root/autodl-tmp/SEUF_MCU/data_cv"


def load_cifar100():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    trainset = torchvision.datasets.CIFAR100(
        root=DATA_ROOT,
        train=True,
        download=True,
        transform=transform,
    )
    testset = torchvision.datasets.CIFAR100(
        root=DATA_ROOT,
        train=False,
        download=True,
        transform=transform,
    )
    return trainset, testset


def save_indices(save_dir, prefix, split_name, forget_indices, retain_indices):
    np.save(
        os.path.join(save_dir, f"forget_indices_{prefix}_{split_name}.npy"),
        np.array(forget_indices, dtype=np.int64),
    )
    np.save(
        os.path.join(save_dir, f"retain_indices_{prefix}_{split_name}.npy"),
        np.array(retain_indices, dtype=np.int64),
    )


def split_by_random_classes(dataset, forget_class_indices, save_dir, split_name):
    targets = np.array(dataset.targets)
    forget_mask = np.isin(targets, forget_class_indices)
    forget_indices = np.where(forget_mask)[0]
    retain_indices = np.where(~forget_mask)[0]

    save_indices(save_dir, "class", split_name, forget_indices, retain_indices)
    print(
        f"[class-{split_name}] done: forget={len(forget_indices)}, retain={len(retain_indices)}"
    )


def split_by_random_percentage(dataset, forget_ratio, rng, save_dir, split_name):
    total_size = len(dataset)
    all_indices = np.arange(total_size)
    rng.shuffle(all_indices)

    forget_count = int(round(total_size * forget_ratio))
    forget_indices = np.sort(all_indices[:forget_count])
    retain_indices = np.sort(all_indices[forget_count:])

    save_indices(save_dir, "random", split_name, forget_indices, retain_indices)
    print(
        f"[random-{split_name}] done: forget={len(forget_indices)}, retain={len(retain_indices)}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Create CIFAR100 forget/retain splits with class-based and random-sample modes"
    )
    parser.add_argument(
        "--save-dir",
        default=os.path.join(BASE_DIR, "split_results_cifar100_fine"),
        help="Directory to save forget/retain index files",
    )
    parser.add_argument(
        "--num-forget-classes",
        default=10,
        type=int,
        help="Number of random classes used in class-based split",
    )
    parser.add_argument(
        "--forget-ratio",
        default=0.1,
        type=float,
        help="Forget ratio used in random-sample split",
    )
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    trainset, testset = load_cifar100()
    print(f"Loaded CIFAR100: train={len(trainset)}, test={len(testset)}")

    num_classes = len(trainset.classes)
    if not 1 <= args.num_forget_classes <= num_classes:
        raise ValueError(f"--num-forget-classes must be in [1, {num_classes}]")
    if not 0.0 < args.forget_ratio < 1.0:
        raise ValueError("--forget-ratio must be in (0, 1)")

    forget_class_indices = np.sort(
        rng.choice(num_classes, size=args.num_forget_classes, replace=False)
    )
    forget_class_names = [trainset.classes[idx] for idx in forget_class_indices]
    print(
        f"[class] selected {args.num_forget_classes} classes: "
        + ", ".join(forget_class_names)
    )

    split_by_random_classes(trainset, forget_class_indices, args.save_dir, "train")
    split_by_random_classes(testset, forget_class_indices, args.save_dir, "test")

    print(f"[random] forget ratio: {args.forget_ratio:.2%}")
    split_by_random_percentage(trainset, args.forget_ratio, rng, args.save_dir, "train")
    split_by_random_percentage(testset, args.forget_ratio, rng, args.save_dir, "test")

    print("\nSaved files:")
    print("  class split:  forget_indices_class_{train,test}.npy / retain_indices_class_{train,test}.npy")
    print("  random split: forget_indices_random_{train,test}.npy / retain_indices_random_{train,test}.npy")


if __name__ == "__main__":
    main()
