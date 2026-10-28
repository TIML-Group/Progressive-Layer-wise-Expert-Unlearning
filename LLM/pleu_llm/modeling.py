"""Model loading, transformers compatibility shims and MoE introspection for
DeepSeek-V2-Lite and Qwen1.5-MoE-A2.7B."""

import importlib.util
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

LLM_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_ROOT = Path(os.environ.get("PLEU_LLM_MODELS", LLM_DIR / "models"))
MODEL_IDS = {
    "deepseek": "deepseek-ai/DeepSeek-V2-Lite-Chat",
    "qwen": "Qwen/Qwen1.5-MoE-A2.7B-Chat",
}


def resolve_model_path(model, model_root=DEFAULT_MODEL_ROOT):
    """``model`` is a short name (deepseek / qwen), a HF id or a local directory."""
    if Path(model).is_dir():
        return str(Path(model))
    repo_id = MODEL_IDS.get(model, model)
    local = Path(model_root) / repo_id.split("/")[-1]
    if not (local / "config.json").is_file():
        raise FileNotFoundError(f"{local} not found; run download_model.py --model {model}")
    return str(local)


def patch_transformers_compat():
    """DeepSeek-V2 remote code targets transformers 4.36; restore the APIs it still uses."""
    from transformers.cache_utils import DynamicCache
    from transformers.utils import import_utils

    if not hasattr(import_utils, "is_torch_fx_available"):
        import_utils.is_torch_fx_available = lambda: hasattr(torch, "fx")
    if not hasattr(DynamicCache, "seen_tokens"):
        DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())
    if not hasattr(DynamicCache, "get_max_length") and hasattr(DynamicCache, "get_max_cache_shape"):
        DynamicCache.get_max_length = DynamicCache.get_max_cache_shape
    if not hasattr(DynamicCache, "get_usable_length"):
        def get_usable_length(self, new_seq_length=0, layer_idx=None):
            max_length = self.get_max_cache_shape() if hasattr(self, "get_max_cache_shape") else None
            current = self.get_seq_length()
            return current if max_length is None else min(current, max_length - new_seq_length)
        DynamicCache.get_usable_length = get_usable_length


def load_model_and_tokenizer(model_path, device_map="auto", dtype=torch.bfloat16):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    patch_transformers_compat()
    if device_map == "auto" and importlib.util.find_spec("accelerate") is None:
        device_map = None
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=dtype if torch.cuda.is_available() else torch.float32,
        device_map=device_map,
    )
    if device_map is None and torch.cuda.is_available():
        model.to("cuda")
    model.config.use_cache = False
    return model, tokenizer


def input_device(model):
    return model.get_input_embeddings().weight.device


# ---------------------------------------------------------------- MoE structure

@dataclass
class MoELayer:
    layer_idx: int
    prefix: str            # e.g. "model.layers.3.mlp."
    block: nn.Module
    router: nn.Module      # module with ``weight`` [num_experts, hidden]
    num_experts: int
    top_k: int

    def expert_prefix(self, expert_idx):
        return f"{self.prefix}experts.{expert_idx}."

    @property
    def router_prefix(self):
        return f"{self.prefix}gate."


def decoder_layers(model):
    base = model
    for attr in ("base_model", "model"):
        while hasattr(base, attr) and not hasattr(base, "layers"):
            base = getattr(base, attr)
    if not hasattr(base, "layers"):
        raise RuntimeError("Could not locate decoder layers.")
    return base.layers


def moe_layers(model) -> List[MoELayer]:
    names = {module: name for name, module in model.named_modules()}
    layers = []
    for layer_idx, layer in enumerate(decoder_layers(model)):
        block = getattr(layer, "mlp", None)
        if block is None or not hasattr(block, "experts") or not hasattr(block, "gate"):
            continue
        top_k = getattr(block, "top_k", None) or getattr(block, "num_experts_per_tok", None) or block.gate.top_k
        layers.append(MoELayer(layer_idx, names[block] + ".", block, block.gate, len(block.experts), int(top_k)))
    if not layers:
        raise RuntimeError("Model has no routed MoE layer.")
    return layers


def expert_param_names(model, layer: MoELayer, expert_idx) -> List[str]:
    prefix = layer.expert_prefix(expert_idx)
    return [name for name, _ in model.named_parameters() if name.startswith(prefix)]


def router_param_names(model, layer: MoELayer) -> List[str]:
    return [name for name, _ in model.named_parameters() if name.startswith(layer.router_prefix)]


class RouterRecorder:
    """Captures softmax routing weights of selected MoE layers.

    The router input can be [B, T, H] (DeepSeek) or [B*T, H] (Qwen2-MoE); the recorded
    tensor is always [B*T, E] in float32 and keeps its graph for the routing loss.
    """

    def __init__(self, model, layer_indices: Optional[List[int]] = None):
        self.probs = {}
        self.handles = []
        wanted = None if layer_indices is None else set(layer_indices)
        for layer in moe_layers(model):
            if wanted is None or layer.layer_idx in wanted:
                self.handles.append(layer.router.register_forward_hook(self._hook(layer.layer_idx)))

    def _hook(self, layer_idx):
        def hook(module, inputs, output):
            hidden = inputs[0]
            hidden = hidden.reshape(-1, hidden.shape[-1])
            logits = F.linear(hidden.float(), module.weight.float())
            self.probs[layer_idx] = logits.softmax(dim=-1)
        return hook

    def clear(self):
        self.probs.clear()

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()


def token_mask(attention_mask):
    return attention_mask.reshape(-1).bool()


def sample_average(probs, attention_mask):
    """Token-averaged routing weights per sample: [B*T, E] -> [B, E] over non-padding tokens."""
    batch, length = attention_mask.shape
    probs = probs.view(batch, length, -1)
    weights = attention_mask.to(probs.device, probs.dtype).unsqueeze(-1)
    return (probs * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def routing_distance(probs, attention_mask, experts):
    """L_dist: squared L2 distance to the one-/multi-hot target over non-padding tokens."""
    keep = token_mask(attention_mask).to(probs.device)
    anchor = torch.zeros_like(probs)
    anchor[:, list(experts)] = 1.0
    per_token = (probs - anchor).pow(2).sum(dim=-1)
    return per_token[keep].mean() if keep.any() else per_token.sum() * 0.0
