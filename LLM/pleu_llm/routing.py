"""Expert relevance (LTEU / SEUF), sample-to-layer assignment (IDFF) and curve-point selection."""

from collections import OrderedDict

import numpy as np
import torch
from tqdm import tqdm

from .data import lm_loss, to_device
from .modeling import RouterRecorder, input_device, moe_layers, sample_average


@torch.no_grad()
def sample_routing_weights(model, loader, desc="routing"):
    """{layer_idx: [N, E]} token-averaged routing weights of every sample (loader must not shuffle)."""
    model.eval()
    device = input_device(model)
    per_layer = OrderedDict((layer.layer_idx, []) for layer in moe_layers(model))
    with RouterRecorder(model) as recorder:
        for batch in tqdm(loader, desc=desc, leave=False):
            batch = to_device(batch, device)
            recorder.clear()
            model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False)
            for layer_idx, probs in recorder.probs.items():
                per_layer[layer_idx].append(sample_average(probs, batch["attention_mask"]).cpu())
    return OrderedDict((k, torch.cat(v)) for k, v in per_layer.items())


def relevance(weights):
    return OrderedDict((layer_idx, w.mean(dim=0)) for layer_idx, w in weights.items())


def layer_targets(weights, strategy="relevance", seed=0):
    rel = relevance(weights)
    rng = np.random.default_rng(seed)
    if strategy == "random":
        return OrderedDict((l, int(rng.integers(r.numel()))) for l, r in rel.items()), rel
    return OrderedDict((l, int(r.argmax())) for l, r in rel.items()), rel


def global_targets(weights, num_experts=1):
    rel = relevance(weights)
    ranked = sorted(((float(s), l, e) for l, r in rel.items() for e, s in enumerate(r.tolist())), reverse=True)
    chosen = OrderedDict()
    for _, l, e in ranked[:num_experts]:
        chosen.setdefault(l, []).append(e)
    return OrderedDict(sorted(chosen.items())), rel


def assign_samples(weights, targets, top_k, exclusive=True):
    """IDFF: each sample goes to the first layer whose top-K (token-averaged) experts contain the target."""
    num = next(iter(weights.values())).size(0)
    remaining = torch.ones(num, dtype=torch.bool)
    assignment = OrderedDict()
    for layer_idx, target in targets.items():
        routed = (weights[layer_idx].topk(top_k[layer_idx], dim=-1).indices == target).any(dim=-1)
        matched = routed & remaining if exclusive else routed
        assignment[layer_idx] = matched.nonzero(as_tuple=True)[0].tolist()
        if exclusive:
            remaining &= ~matched
    unassigned = remaining.nonzero(as_tuple=True)[0].tolist() if exclusive else []
    return assignment, unassigned


@torch.no_grad()
def mean_loss(model, batches):
    device = input_device(model)
    losses = [lm_loss(model, to_device(batch, device)).float().item() for batch in batches]
    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def select_t(model, curve, t_values, forget_batches, retain_batches, retain_tol):
    """Pick t on the curve: the largest held-out forget loss among points whose retain loss
    stays within ``retain_tol`` (relative) of the starting model; otherwise the lowest retain loss."""
    model.eval()
    curve.set_t(0.0)
    base_retain = mean_loss(model, retain_batches)
    table = []
    for t in t_values:
        curve.set_t(t)
        table.append({"t": float(t), "forget_loss": mean_loss(model, forget_batches),
                      "retain_loss": mean_loss(model, retain_batches)})
    ok = [row for row in table if row["retain_loss"] <= base_retain * (1.0 + retain_tol)]
    best = max(ok, key=lambda r: r["forget_loss"]) if ok else min(table, key=lambda r: r["retain_loss"])
    return best["t"], {"base_retain_loss": base_retain, "grid": table}
