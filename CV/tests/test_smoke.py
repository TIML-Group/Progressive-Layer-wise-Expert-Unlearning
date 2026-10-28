"""CPU smoke test of the full CV pipeline on a tiny synthetic ImageNet-style dataset.

    cd CV && python -m pytest tests -q
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from pleu_cv.curve import BezierCurve
from pleu_cv.masks import sepo_mask
from pleu_cv.routing import assign_samples_to_layers, routing_distance

CV_DIR = Path(__file__).resolve().parent.parent
TINY = ["--image-size", "32", "--patch-size", "8", "--embed-dim", "32", "--depth", "4", "--num-heads", "2"]


def test_bezier_endpoints_and_mask():
    start = {"w": torch.zeros(4)}
    end = {"w": torch.ones(4)}
    curve = BezierCurve(start, end, entry_mask={"w": torch.tensor([True, False, True, False])})
    assert torch.equal(curve.export(torch.tensor(0.0))["w"], torch.zeros(4))
    assert torch.equal(curve.export(torch.tensor(1.0))["w"], torch.tensor([1.0, 0.0, 1.0, 0.0]))
    assert curve.num_trainable() == 2


def test_idff_assigns_first_matching_layer():
    weights = {
        1: torch.tensor([[0.9, 0.1], [0.2, 0.8], [0.3, 0.7]]),
        3: torch.tensor([[0.1, 0.9], [0.6, 0.4], [0.2, 0.8]]),
    }
    targets = {1: 0, 3: 0}
    assignment, unassigned = assign_samples_to_layers(weights, targets, {1: 1, 3: 1})
    assert assignment[1].tolist() == [0]
    assert assignment[3].tolist() == [1]
    assert unassigned.tolist() == [2]
    shared, _ = assign_samples_to_layers(weights, targets, {1: 1, 3: 1}, exclusive=False)
    assert shared[3].tolist() == [1]


def test_sepo_mask_protects_retain_entries():
    retain = {"a": torch.tensor([10.0, 0.0, 0.0, 0.0])}
    forget = {"a": torch.tensor([9.0, 8.0, 1.0, 0.5])}
    mask, stats = sepo_mask(retain, forget, k_r=0.25, k_f=0.34)
    assert mask["a"].tolist() == [False, True, False, False]
    assert stats["num_active"] == 1


def test_routing_distance_zero_at_target():
    probs = torch.tensor([[[0.0, 1.0, 0.0]]])
    assert routing_distance(probs, [1]).item() == 0.0
    assert routing_distance(probs, [0]).item() == pytest.approx(2.0)


def run(script, *args, tmp):
    command = [sys.executable, str(CV_DIR / script), "--dataset", "imagenet100",
               "--data-root", str(tmp / "data"), "--output-root", str(tmp / "out"), "--num-workers", "0", *args]
    subprocess.run(command, cwd=CV_DIR, check=True)


@pytest.mark.slow
def test_pipeline(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    for root, classes in ((tmp_path / "data" / "imagenet100", 6), (tmp_path / "data" / "data_re", 3)):
        for c in range(classes):
            folder = root / f"n{c:08d}"
            folder.mkdir(parents=True)
            for i in range(12):
                Image.fromarray(rng.integers(0, 255, (40, 40, 3), dtype=np.uint8)).save(folder / f"{i}.jpg")

    # Patch the dataset defaults to a tiny model through the CLI overrides below.
    monkeypatch.setenv("PYTHONPATH", str(CV_DIR))
    subprocess.run([sys.executable, str(CV_DIR / "prepare_splits.py"), "--dataset", "imagenet100",
                    "--data-root", str(tmp_path / "data"), "--output-root", str(tmp_path / "out"),
                    "--num-forget-classes", "2"], cwd=CV_DIR, check=True)
    tiny = TINY + ["--num-classes", "6", "--batch-size", "8", "--eval-batch-size", "16"]
    run("train.py", "--epochs", "1", "--warmup-epochs", "0", *tiny, tmp=tmp_path)
    for split in ("class", "random"):
        common = ["--split-mode", split, *tiny]
        run("unlearn_baseline.py", "--method", "neggrad_plus", "--epochs", "1", *common, tmp=tmp_path)
        run("unlearn_baseline.py", "--method", "ga", "--seuf", "--epochs", "1", *common, tmp=tmp_path)
        run("unlearn_mcu.py", "--epochs", "1", "--eval-num-points", "2", *common, tmp=tmp_path)
        run("unlearn_pleu.py", "--epochs", "1", "--eval-num-points", "2", "--unassigned", "last", *common,
            tmp=tmp_path)
        for ablation in ("no_lteu", "no_idff", "no_sepo"):
            run("unlearn_pleu.py", "--epochs", "1", "--eval-num-points", "2", "--ablation", ablation, *common,
                tmp=tmp_path)
        run("relearn.py", "--method", "pleu", "--steps", "2", "--trace", *common, tmp=tmp_path)

    result = json.loads((tmp_path / "out" / "imagenet100" / "results" / "pleu_class_s42.json").read_text())
    assert set(result["metrics"]) >= {"forget_train_acc", "retain_test_acc"}
    assert len(json.loads((tmp_path / "out" / "imagenet100" / "results" /
                           "relearn_pleu_class_s42.json").read_text())["fa_train_trace"]) == 3
