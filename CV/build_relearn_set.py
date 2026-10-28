"""Build the benign relearning set: ImageNet-1k classes that are not part of ImageNet-100.

The classes are sampled at random among the ImageNet-1k classes outside ImageNet-100, so
the relearning data is unrelated to the forget set. Images come either from a local
ImageNet-1k train folder (synset sub-directories, symlinked) or from the gated Hugging Face
``imagenet-1k`` dataset (streamed, needs HF_TOKEN).

    python build_relearn_set.py --imagenet-train-dir /path/to/imagenet/train
    python build_relearn_set.py --hf-dataset imagenet-1k --max-per-class 300
"""

import argparse
import json
import os
import random
from collections import defaultdict
from io import BytesIO
from pathlib import Path

from PIL import Image

from pleu_cv.data import DEFAULT_DATA_ROOT, IMAGE_EXTENSIONS, imagenet100_root, relearn_root


def imagenet_synsets():
    from timm.data.imagenet_info import ImageNetInfo

    info = ImageNetInfo()
    return [info.index_to_label_name(i) for i in range(1000)]


def from_local(train_dir, chosen, target, max_per_class):
    counts = {}
    for wnid in chosen:
        images = sorted(p for p in (Path(train_dir) / wnid).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        if max_per_class:
            images = images[:max_per_class]
        (target / wnid).mkdir(parents=True, exist_ok=True)
        for path in images:
            link = target / wnid / path.name
            if not link.exists():
                os.symlink(path.resolve(), link)
        counts[wnid] = len(images)
    return counts


def from_hf(dataset_name, chosen, target, max_per_class, cache_dir):
    from datasets import load_dataset

    synsets = imagenet_synsets()
    wanted = {synsets.index(wnid): wnid for wnid in chosen}
    counts = defaultdict(int)
    stream = load_dataset(dataset_name, split="train", streaming=True, cache_dir=cache_dir,
                          token=os.environ.get("HF_TOKEN"))
    for example in stream:
        wnid = wanted.get(int(example["label"]))
        if wnid is None or (max_per_class and counts[wnid] >= max_per_class):
            continue
        image = example["image"]
        if isinstance(image, dict):
            image = Image.open(BytesIO(image["bytes"])) if image.get("bytes") else Image.open(image["path"])
        (target / wnid).mkdir(parents=True, exist_ok=True)
        image.convert("RGB").save(target / wnid / f"{wnid}_{counts[wnid]:06d}.jpg", quality=95)
        counts[wnid] += 1
        if max_per_class and len(counts) == len(wanted) and all(c >= max_per_class for c in counts.values()):
            break
    return dict(counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--imagenet-train-dir", type=Path, help="Local ImageNet-1k train folder")
    source.add_argument("--hf-dataset", help="Hugging Face dataset name, e.g. imagenet-1k")
    parser.add_argument("--num-classes", type=int, default=100)
    parser.add_argument("--max-per-class", type=int, default=0, help="0 keeps every image")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    excluded = {p.name for p in imagenet100_root(args.data_root).iterdir() if p.is_dir()}
    candidates = [wnid for wnid in imagenet_synsets() if wnid not in excluded]
    chosen = sorted(random.Random(args.seed).sample(candidates, args.num_classes))
    target = relearn_root(args.data_root)
    target.mkdir(parents=True, exist_ok=True)
    print(f"excluded {len(excluded)} ImageNet-100 classes, selected {len(chosen)} of {len(candidates)}")

    if args.imagenet_train_dir is not None:
        counts = from_local(args.imagenet_train_dir, chosen, target, args.max_per_class)
    else:
        counts = from_hf(args.hf_dataset, chosen, target, args.max_per_class, args.cache_dir)
    metadata = {"seed": args.seed, "classes": chosen, "counts": counts,
                "excluded_imagenet100": sorted(excluded)}
    (target / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"wrote {sum(counts.values())} images to {target}")


if __name__ == "__main__":
    main()
