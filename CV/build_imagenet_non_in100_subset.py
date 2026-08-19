import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import argparse
import json
import random
import shutil
from collections import defaultdict
from io import BytesIO
from pathlib import Path

from datasets import load_dataset
from datasets.exceptions import DatasetNotFoundError
from PIL import Image
from timm.data.imagenet_info import ImageNetInfo


BASE_DIR = Path("/root/autodl-tmp/SEUF_MCU")
IMAGENET100_ROOT = BASE_DIR / "data_cv" / "imagenet100"
TARGET_ROOT = BASE_DIR / "data_cv" / "data_re"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download a 100-class ImageNet subset excluding classes already present in imagenet100."
    )
    parser.add_argument(
        "--dataset-name",
        default="imagenet-1k",
        help="Hugging Face dataset name.",
    )
    parser.add_argument(
        "--imagenet100-root",
        default=str(IMAGENET100_ROOT),
        help="Path to existing imagenet100 root. Synset folders here will be excluded.",
    )
    parser.add_argument(
        "--target-root",
        default=str(TARGET_ROOT),
        help="Target directory for the new subset.",
    )
    parser.add_argument(
        "--num-classes",
        default=100,
        type=int,
        help="Number of classes to keep after excluding imagenet100 classes.",
    )
    parser.add_argument(
        "--seed",
        default=123,
        type=int,
        help="Random seed for class sampling.",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(BASE_DIR / "data_cv" / "hf_cache"),
        help="Hugging Face datasets cache directory.",
    )
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN", os.environ.get("HUGGINGFACE_TOKEN")),
        help="Hugging Face token. Defaults to HF_TOKEN/HUGGINGFACE_TOKEN env var.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete target directory first if it already exists.",
    )
    return parser.parse_args()


def list_synset_dirs(root: Path):
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root}")
    return sorted([p.name for p in root.iterdir() if p.is_dir()])


def safe_mkdir(path: Path, overwrite: bool):
    if path.exists() and overwrite:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def get_imagenet_label_map():
    info = ImageNetInfo()
    synsets = [info.index_to_label_name(i) for i in range(1000)]
    return synsets


def resolve_selected_classes(excluded_synsets, num_classes, seed):
    synsets = get_imagenet_label_map()
    candidate = [(idx, synset) for idx, synset in enumerate(synsets) if synset not in excluded_synsets]
    if len(candidate) < num_classes:
        raise ValueError(
            f"Only {len(candidate)} candidate classes remain after exclusion, "
            f"but num_classes={num_classes} was requested."
        )
    rng = random.Random(seed)
    selected = sorted(rng.sample(candidate, num_classes), key=lambda x: x[0])
    selected_indices = {idx for idx, _ in selected}
    selected_synsets = [synset for _, synset in selected]
    return selected, selected_indices, selected_synsets, len(candidate)


def normalize_image(image_field):
    if isinstance(image_field, Image.Image):
        return image_field.convert("RGB")
    if isinstance(image_field, dict):
        if image_field.get("bytes") is not None:
            return Image.open(BytesIO(image_field["bytes"])).convert("RGB")
        if image_field.get("path"):
            return Image.open(image_field["path"]).convert("RGB")
    raise TypeError(f"Unsupported image field type: {type(image_field)}")


def dump_split(dataset_name, split_name, selected_indices, index_to_synset, target_root, cache_dir, hf_token):
    try:
        dataset = load_dataset(
            dataset_name,
            split=split_name,
            streaming=True,
            cache_dir=cache_dir,
            trust_remote_code=True,
            token=hf_token,
        )
    except DatasetNotFoundError as exc:
        raise RuntimeError(
            f"Failed to access dataset '{dataset_name}'. "
            "This dataset is gated on Hugging Face. "
            "You need a Hugging Face account with access permission and a valid HF token. "
            "Set it with `export HF_TOKEN=...` or pass `--hf-token`."
        ) from exc

    split_counters = defaultdict(int)
    total_saved = 0

    for example in dataset:
        label = int(example["label"])
        if label not in selected_indices:
            continue

        synset = index_to_synset[label]
        class_dir = target_root / synset
        class_dir.mkdir(parents=True, exist_ok=True)

        image = normalize_image(example["image"])
        split_prefix = "train" if split_name == "train" else "val"
        filename = f"{split_prefix}_{synset}_{split_counters[synset]:08d}.jpg"
        image.save(class_dir / filename, format="JPEG", quality=95)

        split_counters[synset] += 1
        total_saved += 1

    return total_saved, dict(split_counters)


def main():
    args = parse_args()

    imagenet100_root = Path(args.imagenet100_root)
    target_root = Path(args.target_root)
    cache_dir = Path(args.cache_dir)

    if not args.hf_token:
        raise RuntimeError(
            "Missing Hugging Face token. "
            "Dataset 'imagenet-1k' is gated. "
            "Set `HF_TOKEN` in the environment or pass `--hf-token`."
        )

    excluded_synsets = set(list_synset_dirs(imagenet100_root))
    selected, selected_indices, selected_synsets, candidate_count = resolve_selected_classes(
        excluded_synsets=excluded_synsets,
        num_classes=args.num_classes,
        seed=args.seed,
    )
    index_to_synset = {idx: synset for idx, synset in selected}

    safe_mkdir(target_root, args.overwrite)
    target_root.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Excluded imagenet100 classes: {len(excluded_synsets)}")
    print(f"Remaining candidate classes: {candidate_count}")
    print(f"Selected classes: {len(selected_synsets)}")
    print("Start downloading train split ...")
    train_image_count, train_per_class = dump_split(
        dataset_name=args.dataset_name,
        split_name="train",
        selected_indices=selected_indices,
        index_to_synset=index_to_synset,
        target_root=target_root,
        cache_dir=str(cache_dir),
        hf_token=args.hf_token,
    )
    print("Start downloading validation split ...")
    val_image_count, val_per_class = dump_split(
        dataset_name=args.dataset_name,
        split_name="validation",
        selected_indices=selected_indices,
        index_to_synset=index_to_synset,
        target_root=target_root,
        cache_dir=str(cache_dir),
        hf_token=args.hf_token,
    )

    metadata = {
        "dataset_name": args.dataset_name,
        "hf_endpoint": os.environ["HF_ENDPOINT"],
        "imagenet100_root": str(imagenet100_root),
        "target_root": str(target_root),
        "cache_dir": str(cache_dir),
        "num_classes": args.num_classes,
        "seed": args.seed,
        "excluded_count": len(excluded_synsets),
        "candidate_count": candidate_count,
        "selected_classes": [{"label_index": idx, "synset": synset} for idx, synset in selected],
        "train_image_count": train_image_count,
        "val_image_count": val_image_count,
        "train_per_class": train_per_class,
        "val_per_class": val_per_class,
    }

    (target_root / "selected_synsets.txt").write_text("\n".join(selected_synsets) + "\n", encoding="utf-8")
    (target_root / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Train images written: {train_image_count}")
    print(f"Val images written: {val_image_count}")
    print(f"Saved subset to: {target_root}")


if __name__ == "__main__":
    main()
