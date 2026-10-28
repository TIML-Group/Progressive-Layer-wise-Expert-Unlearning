"""Create the class-wise and random (10%) forget/retain splits.

CIFAR-100 reproduces ``split_cifar100_fine.py``; ImageNet-100 first splits every class
into 90% train / 10% test exactly like the original ImageNet-100 training script.

    python prepare_splits.py --dataset cifar100
    python prepare_splits.py --dataset imagenet100 [--from-json old/imagenet100_classwise_split.json]
"""

import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np
import torchvision

from pleu_cv.data import (
    DEFAULT_DATA_ROOT,
    DEFAULT_OUTPUT_ROOT,
    discover_image_folder,
    imagenet100_root,
    imagenet100_split_json,
    load_imagenet100_samples,
    split_dir,
    split_paths,
    split_per_class,
    write_imagenet100_split_json,
)


def save_split(directory, mode, part, forget, retain):
    paths = split_paths(directory, mode)
    np.save(paths[f"forget_{part}"], np.asarray(forget, dtype=np.int64))
    np.save(paths[f"retain_{part}"], np.asarray(retain, dtype=np.int64))
    print(f"[{mode}-{part}] forget={len(forget)} retain={len(retain)}")


def class_split(targets, forget_labels):
    targets = np.asarray(targets)
    forget = np.isin(targets, forget_labels)
    return np.where(forget)[0], np.where(~forget)[0]


def random_split(num_samples, ratio, rng):
    order = np.arange(num_samples)
    rng.shuffle(order)
    count = int(round(num_samples * ratio))
    return np.sort(order[:count]), np.sort(order[count:])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="cifar100", choices=["cifar100", "imagenet100"])
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--num-forget-classes", type=int, default=10)
    parser.add_argument("--forget-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1, help="ImageNet-100 per-class test split")
    parser.add_argument("--from-json", type=Path, default=None,
                        help="ImageNet-100: reuse an existing imagenet100_classwise_split.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = split_dir(args.output_root, args.dataset)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    metadata = {"seed": args.seed, "forget_ratio": args.forget_ratio}

    if args.dataset == "cifar100":
        trainset = torchvision.datasets.CIFAR100(root=str(args.data_root), train=True, download=True)
        testset = torchvision.datasets.CIFAR100(root=str(args.data_root), train=False, download=True)
        train_targets, test_targets = trainset.targets, testset.targets
        class_names = trainset.classes
        forget_labels = np.sort(rng.choice(len(class_names), size=args.num_forget_classes, replace=False))
    else:
        json_path = imagenet100_split_json(args.output_root)
        if args.from_json is not None:
            shutil.copyfile(args.from_json, json_path)
            print(f"Reusing ImageNet-100 split from {args.from_json}")
        else:
            samples, class_to_idx = discover_image_folder(imagenet100_root(args.data_root))
            train, test = split_per_class(samples, class_to_idx, args.test_ratio, args.seed)
            write_imagenet100_split_json(json_path, imagenet100_root(args.data_root), class_to_idx,
                                         train, test, args.test_ratio, args.seed)
        train, test, payload = load_imagenet100_samples(json_path, imagenet100_root(args.data_root))
        train_targets = [label for _, label in train]
        test_targets = [label for _, label in test]
        class_to_idx = payload["class_to_idx"]
        class_names = sorted(class_to_idx, key=class_to_idx.get)
        if "forget_labels" in payload:
            forget_labels = np.asarray(sorted(payload["forget_labels"]))
        else:
            chosen = sorted(random.Random(args.seed).sample(sorted(class_to_idx), args.num_forget_classes))
            forget_labels = np.asarray(sorted(class_to_idx[name] for name in chosen))

    metadata["forget_labels"] = [int(label) for label in forget_labels]
    metadata["forget_classes"] = [class_names[label] for label in forget_labels]
    print("Forget classes:", ", ".join(metadata["forget_classes"]))

    save_split(out_dir, "class", "train", *class_split(train_targets, forget_labels))
    save_split(out_dir, "class", "test", *class_split(test_targets, forget_labels))
    save_split(out_dir, "random", "train", *random_split(len(train_targets), args.forget_ratio, rng))
    save_split(out_dir, "random", "test", *random_split(len(test_targets), args.forget_ratio, rng))

    (out_dir / "split_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved splits to {out_dir}")


if __name__ == "__main__":
    main()
