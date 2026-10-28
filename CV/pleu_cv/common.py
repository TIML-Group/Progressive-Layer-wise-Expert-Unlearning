"""Shared helpers: arguments, seeding, checkpoints, evaluation and result files."""

import argparse
import json
import random
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .data import DATASET_CONFIGS, DEFAULT_DATA_ROOT, DEFAULT_OUTPUT_ROOT, SPLIT_MODES
from .model import MoEVisionTransformer


def add_common_args(parser, split_mode_default="class"):
    parser.add_argument("--dataset", default="cifar100", choices=sorted(DATASET_CONFIGS))
    parser.add_argument("--split-mode", default=split_mode_default, choices=SPLIT_MODES)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--origin-path", type=Path, default=None,
                        help="Original model checkpoint (default: <output-root>/<dataset>/models/origin.pth)")
    parser.add_argument("--batch-size", type=int, default=None, help="Training batch size (dataset default)")
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", default="", help="Optional suffix for output file names")
    arch = parser.add_argument_group("architecture overrides (default: dataset config)")
    for flag in ("--num-classes", "--image-size", "--patch-size", "--embed-dim", "--depth", "--num-heads"):
        arch.add_argument(flag, type=int, default=None)
    return parser


def finalize_args(args):
    config = DATASET_CONFIGS[args.dataset]
    for key in ("num_classes", "image_size", "patch_size", "embed_dim", "depth", "num_heads"):
        if getattr(args, key, None) is None:
            setattr(args, key, config[key])
    if getattr(args, "batch_size", None) is None:
        args.batch_size = config["train"]["batch_size"]
    if args.origin_path is None:
        args.origin_path = models_dir(args) / "origin.pth"
    return args


def parse_args(parser):
    return finalize_args(parser.parse_args())


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------- paths

def dataset_dir(args):
    return Path(args.output_root) / args.dataset


def models_dir(args):
    return dataset_dir(args) / "models"


def results_dir(args):
    return dataset_dir(args) / "results"


def run_name(method, args):
    name = f"{method}_{args.split_mode}_s{args.seed}"
    return f"{name}_{args.tag}" if args.tag else name


def model_path(method, args):
    return models_dir(args) / f"{run_name(method, args)}.pth"


def write_results(method, args, payload):
    path = results_dir(args) / f"{run_name(method, args)}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable_args = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    payload = {"method": method, "dataset": args.dataset, "split_mode": args.split_mode,
               "seed": args.seed, **payload, "args": serializable_args}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote results to {path}")
    return path


# ---------------------------------------------------------------- models

def build_model(args, device):
    return MoEVisionTransformer(
        img_size=args.image_size,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        num_classes=args.num_classes,
    ).to(device)


def extract_state_dict(checkpoint):
    state = checkpoint
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model_state"):
            if key in checkpoint:
                state = checkpoint[key]
                break
    return OrderedDict((k[len("module."):] if k.startswith("module.") else k, v) for k, v in state.items())


def load_model(args, path, device):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    model = build_model(args, device)
    model.load_state_dict(extract_state_dict(torch.load(path, map_location=device)), strict=True)
    return model


def save_model(path, state_dict, **extra):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state_dict = OrderedDict((k, v.detach().cpu()) for k, v in state_dict.items())
    torch.save({"model_state_dict": state_dict, **extra}, path)
    print(f"Saved model to {path}")


def clone_state(model):
    return OrderedDict((name, value.detach().clone()) for name, value in model.state_dict().items())


# ---------------------------------------------------------------- evaluation

@torch.no_grad()
def evaluate(model, loader, device, desc="eval", max_batches=None):
    model.eval()
    correct = total = 0
    for batch_idx, (inputs, targets) in enumerate(tqdm(loader, desc=desc, leave=False)):
        if max_batches is not None and batch_idx >= max_batches:
            break
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(inputs)[0]
        correct += logits.argmax(dim=1).eq(targets).sum().item()
        total += targets.size(0)
    return 100.0 * correct / max(total, 1)


def compute_metrics(model, eval_loaders, device, max_batches=None):
    metrics = {
        "forget_train_acc": evaluate(model, eval_loaders["forget_train_eval"], device, "FA train", max_batches),
        "retain_train_acc": evaluate(model, eval_loaders["retain_train_eval"], device, "RA train", max_batches),
        "forget_test_acc": evaluate(model, eval_loaders["forget_test"], device, "FA test", max_batches),
        "retain_test_acc": evaluate(model, eval_loaders["retain_test"], device, "RA test", max_batches),
        "full_test_acc": evaluate(model, eval_loaders["full_test"], device, "TA", max_batches),
    }
    return metrics


def selection_score(metrics, args):
    """Lower is better. Class-wise: maximize RA - FA. Random: match the RT forget accuracy."""
    if args.split_mode == "random":
        target = getattr(args, "target_forget_acc", None)
        if target is None:
            target = DATASET_CONFIGS[args.dataset]["rt_forget_train_acc"]
        return abs(metrics["forget_train_acc"] - target)
    return -(metrics["retain_train_acc"] + metrics["retain_test_acc"]
             - metrics["forget_train_acc"] - metrics["forget_test_acc"])


def add_selection_args(parser):
    parser.add_argument("--target-forget-acc", type=float, default=None,
                        help="Random split: train FA of the retrained model to aim for (default: Table 2 RT)")


def format_metrics(metrics):
    return (f"FA(train) {metrics['forget_train_acc']:.2f} | RA(train) {metrics['retain_train_acc']:.2f} | "
            f"FA(test) {metrics['forget_test_acc']:.2f} | RA(test) {metrics['retain_test_acc']:.2f} | "
            f"TA {metrics['full_test_acc']:.2f}")


class Timer:
    def __init__(self):
        self.start = time.time()

    def minutes(self):
        return (time.time() - self.start) / 60.0


def new_parser(description):
    return argparse.ArgumentParser(description=description)
