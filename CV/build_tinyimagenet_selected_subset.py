import argparse
import json
import re
import shutil
from collections import defaultdict
from difflib import get_close_matches
from pathlib import Path


DATA_ROOT = Path("/root/autodl-tmp/SEUF_MCU/data_cv")
SOURCE_ROOT = DATA_ROOT / "tiny-imagenet-200"
TARGET_ROOT = DATA_ROOT / "select_tiinyimage"

TARGET_LABELS = [
    "alp",
    "bicycle-built-for-two",
    "bison",
    "beer_bottle",
    "chest",
    "cliff",
    "lakeside",
    "lawn_mower",
    "mountain_bike",
    "ox",
    "pill_bottle",
    "pop_bottle",
    "seashore",
    "street_sign",
    "sunflower",
    "traffic_light",
    "tractor",
    "tricycle",
    "valley",
    "water_bottle",
    "water_buffalo",
    "willow_tree",
    "wine_bottle",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Select requested classes from a local Tiny-ImageNet dataset."
    )
    parser.add_argument(
        "--source-root",
        default=str(SOURCE_ROOT),
        help="Path to local tiny-imagenet-200 directory.",
    )
    parser.add_argument(
        "--target-root",
        default=str(TARGET_ROOT),
        help="Output directory for the selected subset.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete target directory before writing files.",
    )
    return parser.parse_args()


def normalize_name(text):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text.lower())).strip()


def load_words(words_path: Path):
    wnid_to_words = {}
    with words_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            wnid, words = line.split("\t", 1)
            wnid_to_words[wnid] = [item.strip() for item in words.split(",")]
    return wnid_to_words


def resolve_targets(wnid_to_words, target_labels):
    normalized_index = defaultdict(list)
    all_candidates = set()

    for wnid, synonyms in wnid_to_words.items():
        for synonym in synonyms:
            normalized = normalize_name(synonym)
            normalized_index[normalized].append((wnid, synonym))
            all_candidates.add(normalized)

    resolved = []
    for label in target_labels:
        normalized_label = normalize_name(label)
        matches = normalized_index.get(normalized_label, [])
        if not matches:
            suggestions = get_close_matches(normalized_label, sorted(all_candidates), n=5, cutoff=0.6)
            raise ValueError(
                f"Could not match target label '{label}'. "
                f"Normalized form: '{normalized_label}'. "
                f"Closest Tiny-ImageNet entries: {suggestions}"
            )

        wnids = sorted({wnid for wnid, _ in matches})
        if len(wnids) > 1:
            raise ValueError(
                f"Target label '{label}' matched multiple wnids: {wnids}. "
                f"Matches: {matches}"
            )

        wnid = wnids[0]
        resolved.append(
            {
                "requested_label": label,
                "normalized_label": normalized_label,
                "wnid": wnid,
                "tinyimagenet_words": wnid_to_words[wnid],
            }
        )
    return resolved


def safe_reset_dir(path: Path, overwrite: bool):
    if path.exists() and overwrite:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_train_split(source_root: Path, target_root: Path, resolved_targets):
    copied = {}
    for item in resolved_targets:
        wnid = item["wnid"]
        src_dir = source_root / "train" / wnid / "images"
        src_boxes = source_root / "train" / wnid / f"{wnid}_boxes.txt"
        dst_dir = target_root / "train" / wnid
        dst_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        for image_path in sorted(src_dir.glob("*")):
            if image_path.is_file():
                shutil.copy2(image_path, dst_dir / image_path.name)
                count += 1

        if src_boxes.exists():
            shutil.copy2(src_boxes, dst_dir / src_boxes.name)

        copied[wnid] = count
    return copied


def copy_val_split(source_root: Path, target_root: Path, selected_wnids):
    annotations_path = source_root / "val" / "val_annotations.txt"
    images_dir = source_root / "val" / "images"
    copied = defaultdict(int)
    selected_annotations = []

    with annotations_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue

            image_name, wnid = parts[0], parts[1]
            if wnid not in selected_wnids:
                continue

            src_path = images_dir / image_name
            dst_dir = target_root / "val" / wnid
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dst_dir / image_name)
            copied[wnid] += 1
            selected_annotations.append(line.rstrip("\n"))

    (target_root / "val" / "val_annotations.txt").write_text(
        "\n".join(selected_annotations) + ("\n" if selected_annotations else ""),
        encoding="utf-8",
    )
    return dict(copied)


def main():
    args = parse_args()
    source_root = Path(args.source_root)
    target_root = Path(args.target_root)

    if not source_root.exists():
        raise FileNotFoundError(f"Tiny-ImageNet directory not found: {source_root}")

    words_path = source_root / "words.txt"
    if not words_path.exists():
        raise FileNotFoundError(f"words.txt not found: {words_path}")

    wnid_to_words = load_words(words_path)
    resolved_targets = resolve_targets(wnid_to_words, TARGET_LABELS)
    selected_wnids = [item["wnid"] for item in resolved_targets]

    safe_reset_dir(target_root, args.overwrite)
    (target_root / "train").mkdir(parents=True, exist_ok=True)
    (target_root / "val").mkdir(parents=True, exist_ok=True)

    train_counts = copy_train_split(source_root, target_root, resolved_targets)
    val_counts = copy_val_split(source_root, target_root, set(selected_wnids))

    metadata = {
        "source_root": str(source_root),
        "target_root": str(target_root),
        "num_requested_classes": len(TARGET_LABELS),
        "resolved_targets": resolved_targets,
        "train_counts": train_counts,
        "val_counts": val_counts,
    }

    (target_root / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (target_root / "selected_wnids.txt").write_text("\n".join(selected_wnids) + "\n", encoding="utf-8")

    print("Selected Tiny-ImageNet classes:")
    for item in resolved_targets:
        print(f"  {item['requested_label']} -> {item['wnid']} ({', '.join(item['tinyimagenet_words'])})")

    print(f"\nSaved subset to: {target_root}")
    print(f"Train classes: {len(train_counts)} | total images: {sum(train_counts.values())}")
    print(f"Val classes: {len(val_counts)} | total images: {sum(val_counts.values())}")


if __name__ == "__main__":
    main()
