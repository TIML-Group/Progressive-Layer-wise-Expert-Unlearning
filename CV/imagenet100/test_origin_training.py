import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_origin import (
    build_classwise_groups,
    discover_samples,
    split_samples_by_class,
)


def _write_image(path: Path, color: int) -> None:
    Image.new("RGB", (4, 4), (color, color, color)).save(path)


def test_discover_samples_returns_sorted_class_mapping(tmp_path):
    for class_name, color in (("n0002", 20), ("n0001", 10)):
        class_dir = tmp_path / class_name
        class_dir.mkdir()
        _write_image(class_dir / f"{class_name}_b.JPEG", color)
        _write_image(class_dir / f"{class_name}_a.JPEG", color)

    samples, class_to_idx = discover_samples(tmp_path)

    assert class_to_idx == {"n0001": 0, "n0002": 1}
    assert [path.name for path, _ in samples] == [
        "n0001_a.JPEG",
        "n0001_b.JPEG",
        "n0002_a.JPEG",
        "n0002_b.JPEG",
    ]
    assert [label for _, label in samples] == [0, 0, 1, 1]


def test_split_samples_is_class_wise_reproducible_and_disjoint(tmp_path):
    for class_name, color in (("n0001", 10), ("n0002", 20)):
        class_dir = tmp_path / class_name
        class_dir.mkdir()
        for index in range(5):
            _write_image(class_dir / f"{index}.JPEG", color)

    samples, class_to_idx = discover_samples(tmp_path)
    train_a, val_a = split_samples_by_class(samples, class_to_idx, val_ratio=0.4, seed=42)
    train_b, val_b = split_samples_by_class(samples, class_to_idx, val_ratio=0.4, seed=42)

    assert train_a == train_b
    assert val_a == val_b
    assert set(train_a).isdisjoint(val_a)
    assert len(train_a) == 6
    assert len(val_a) == 4

    train_labels = [label for _, label in train_a]
    val_labels = [label for _, label in val_a]
    assert train_labels.count(0) == train_labels.count(1) == 3
    assert val_labels.count(0) == val_labels.count(1) == 2


def test_split_metadata_can_be_saved_for_later_classwise_unlearning(tmp_path):
    for class_name in ("n0001", "n0002"):
        class_dir = tmp_path / class_name
        class_dir.mkdir()
        for index in range(2):
            _write_image(class_dir / f"{index}.JPEG", index)

    samples, class_to_idx = discover_samples(tmp_path)
    train_samples, val_samples = split_samples_by_class(
        samples, class_to_idx, val_ratio=0.5, seed=42
    )
    metadata = {
        "class_to_idx": class_to_idx,
        "train": [(str(path.relative_to(tmp_path)), label) for path, label in train_samples],
        "val": [(str(path.relative_to(tmp_path)), label) for path, label in val_samples],
    }
    metadata_path = tmp_path / "split.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    loaded = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert loaded["class_to_idx"] == class_to_idx
    assert len(loaded["train"]) == 2
    assert len(loaded["val"]) == 2


def test_classwise_groups_are_reproducible_and_cover_all_classes(tmp_path):
    for class_name in ("n0001", "n0002", "n0003", "n0004"):
        class_dir = tmp_path / class_name
        class_dir.mkdir()
        _write_image(class_dir / "0.JPEG", 10)

    _, class_to_idx = discover_samples(tmp_path)
    forget, retain = build_classwise_groups(class_to_idx, num_forget_classes=2, seed=42)

    assert forget == ["n0001", "n0004"]
    assert retain == ["n0002", "n0003"]
    assert sorted(forget + retain) == sorted(class_to_idx)


def test_classwise_groups_reject_invalid_forget_count(tmp_path):
    for class_name in ("n0001", "n0002"):
        class_dir = tmp_path / class_name
        class_dir.mkdir()
        _write_image(class_dir / "0.JPEG", 10)

    _, class_to_idx = discover_samples(tmp_path)

    try:
        build_classwise_groups(class_to_idx, num_forget_classes=2, seed=42)
    except ValueError as error:
        assert "at least one retain class" in str(error)
    else:
        raise AssertionError("Expected invalid class-wise split to raise ValueError")
