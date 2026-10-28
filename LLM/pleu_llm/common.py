"""Arguments, run directories, result files and unlearned-model checkpoints.

An unlearned model is stored as a directory that refers to the untouched base model:
``delta.safetensors`` holds the tensors that changed (every tensor for full-model GA /
NegGrad+ and MCU, the target experts and routers for SEUF / PLEU). Single tensors can be
read lazily, so PLEU only loads the endpoint experts it needs.
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from .data import DEFAULT_DATA_ROOT, DEFAULT_OUTPUT_ROOT, FORGET_DATASETS
from .modeling import DEFAULT_MODEL_ROOT, load_model_and_tokenizer, resolve_model_path

CHECKPOINT_FILE = "pleu_checkpoint.json"


def add_common_args(parser):
    parser.add_argument("--model", default="deepseek", help="deepseek | qwen | HF id | local path")
    parser.add_argument("--forget-dataset", default="wmdp", choices=FORGET_DATASETS)
    parser.add_argument("--wmdp-subset", default="cyber", choices=["cyber", "bio"])
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--forget-file", type=Path, default=None, help="JSONL forget corpus (offline override)")
    parser.add_argument("--retain-file", type=Path, default=None, help="JSONL retain corpus (offline override)")
    parser.add_argument("--max-forget-samples", type=int, default=512)
    parser.add_argument("--max-retain-samples", type=int, default=512)
    parser.add_argument("--cutoff-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--tag", default="")
    return parser


def new_parser(description):
    return argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def model_key(args):
    return Path(args.model).name if Path(args.model).is_dir() else args.model.split("/")[-1]


def run_dir(method, args):
    name = f"{method}_s{args.seed}" + (f"_{args.tag}" if args.tag else "")
    return Path(args.output_root) / model_key(args) / args.forget_dataset / name


def write_results(path, method, args, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serial = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    payload = {"method": method, "model": args.model, "forget_dataset": args.forget_dataset,
               "seed": args.seed, **payload, "args": serial}
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"Wrote results to {path}")


class Timer:
    def __init__(self):
        self.start = time.time()

    def minutes(self):
        return (time.time() - self.start) / 60.0


# ---------------------------------------------------------------- checkpoints

def save_delta(out_dir, base_model, tensors, **meta):
    from safetensors.torch import save_file

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # copy=True: tied weights on CPU must not share storage in the file
    save_file({k: v.detach().to("cpu", copy=True).contiguous() for k, v in tensors.items()},
              str(out_dir / "delta.safetensors"))
    (out_dir / CHECKPOINT_FILE).write_text(
        json.dumps({"kind": "delta", "base_model": str(base_model), **meta}, indent=2, default=str), encoding="utf-8")
    print(f"Saved {len(tensors)} changed tensors to {out_dir}")


def read_checkpoint_meta(path):
    meta_path = Path(path) / CHECKPOINT_FILE
    return json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else None


def load_tensors(path, names=None, device="cpu"):
    """Tensors stored in a delta checkpoint (all, or the requested names)."""
    from safetensors import safe_open

    out = {}
    with safe_open(str(Path(path) / "delta.safetensors"), framework="pt", device=device) as handle:
        for key in handle.keys():
            if names is None or key in names:
                out[key] = handle.get_tensor(key)
    return out


@torch.no_grad()
def apply_delta(model, tensors):
    params = dict(model.named_parameters())
    for name, value in tensors.items():
        params[name].copy_(value.to(params[name].device, params[name].dtype))


def load_any(path_or_model, args, device_map=None):
    """Base model (short name / path) or an unlearned checkpoint directory, ready to use."""
    meta = read_checkpoint_meta(path_or_model) if Path(str(path_or_model)).is_dir() else None
    device_map = args.device_map if device_map is None else device_map
    if meta is None:
        return load_model_and_tokenizer(resolve_model_path(path_or_model, args.model_root), device_map)
    model, tokenizer = load_model_and_tokenizer(meta["base_model"], device_map)
    apply_delta(model, load_tensors(path_or_model))
    return model, tokenizer


def endpoint_tensors(path, names=None):
    """Tensors of a pre-unlearned endpoint on CPU: the requested names (all if None).
    Tensors the endpoint did not change come from its base model."""
    meta = read_checkpoint_meta(path)
    if meta is None:
        raise FileNotFoundError(f"{path} is not an unlearned checkpoint directory")
    stored = load_tensors(path, None if names is None else set(names))
    missing = [] if names is None else [n for n in names if n not in stored]
    if missing:
        from safetensors import safe_open

        wanted = set(missing)
        for shard in sorted(Path(meta["base_model"]).glob("*.safetensors")):
            with safe_open(str(shard), framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    if key in wanted:
                        stored[key] = handle.get_tensor(key)
        still = [n for n in missing if n not in stored]
        if still:
            raise KeyError(f"Tensors not found in endpoint or base model: {still[:5]}")
    return stored
