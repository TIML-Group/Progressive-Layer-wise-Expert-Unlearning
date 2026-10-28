"""Router introspection: expert relevance (LTEU), sample-to-layer assignment (IDFF)
and the routing distance constraint."""

from collections import OrderedDict

import numpy as np
import torch
from tqdm import tqdm


def moe_layers(model):
    """[(block index, number of experts, top-k)] for every MoE block."""
    layers = []
    for layer_idx, blk in enumerate(model.blocks):
        if getattr(blk, "use_moe", False):
            layers.append((layer_idx, blk.mlp.num_experts, blk.mlp.top_k))
    if not layers:
        raise RuntimeError("Model has no MoE layer.")
    return layers


def expert_param_prefix(layer_idx, expert_idx):
    return f"blocks.{layer_idx}.mlp.experts.{expert_idx}."


def router_param_prefix(layer_idx):
    return f"blocks.{layer_idx}.mlp.gate."


class GateRecorder:
    """Records softmax routing weights [B, T, E] of selected MoE layers via forward hooks.

    Hooks fire under ``torch.func.functional_call`` too, so the recorded tensors
    keep their graph and can be used in the routing distance loss.
    """

    def __init__(self, model, layers=None):
        self.probs = OrderedDict()
        self.handles = []
        wanted = None if layers is None else set(layers)
        for layer_idx, _, _ in moe_layers(model):
            if wanted is not None and layer_idx not in wanted:
                continue
            gate = model.blocks[layer_idx].mlp.gate
            self.handles.append(gate.register_forward_hook(self._hook(layer_idx)))

    def _hook(self, layer_idx):
        def hook(module, inputs, output):
            self.probs[layer_idx] = torch.softmax(output, dim=-1)
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


@torch.no_grad()
def sample_routing_weights(model, loader, device, desc="routing"):
    """Token-averaged routing weights G^(l)(h^(l-1)(x)) for every sample, computed on ``model``.

    ``loader`` must not shuffle; row i corresponds to the i-th sample of its dataset.
    Returns {layer_idx: FloatTensor [N, E]} on CPU.
    """
    model.eval()
    per_layer = OrderedDict((layer_idx, []) for layer_idx, _, _ in moe_layers(model))
    with GateRecorder(model) as recorder:
        for inputs, _ in tqdm(loader, desc=desc, leave=False):
            recorder.clear()
            model(inputs.to(device, non_blocking=True))
            for layer_idx, probs in recorder.probs.items():
                per_layer[layer_idx].append(probs.mean(dim=1).float().cpu())
    return OrderedDict((layer_idx, torch.cat(chunks)) for layer_idx, chunks in per_layer.items())


def expert_relevance(sample_weights):
    """Rel_i^(l): expected routing weight of each expert over the forget set (Eq. 3.2)."""
    return OrderedDict((layer_idx, weights.mean(dim=0)) for layer_idx, weights in sample_weights.items())


def select_layer_targets(sample_weights, strategy="relevance", seed=0):
    """E_*^(l) for every MoE layer (Eq. 3.3). ``random`` is the w/o-LTEU ablation."""
    relevance = expert_relevance(sample_weights)
    rng = np.random.default_rng(seed)
    targets = OrderedDict()
    for layer_idx, rel in relevance.items():
        if strategy == "relevance":
            targets[layer_idx] = int(rel.argmax().item())
        elif strategy == "random":
            targets[layer_idx] = int(rng.integers(rel.numel()))
        else:
            raise ValueError(f"Unknown target strategy: {strategy}")
    return targets, relevance


def select_global_targets(sample_weights, num_experts=1):
    """SEUF: the ``num_experts`` (layer, expert) pairs with the highest relevance over the network."""
    relevance = expert_relevance(sample_weights)
    candidates = [
        (float(score), layer_idx, expert_idx)
        for layer_idx, rel in relevance.items()
        for expert_idx, score in enumerate(rel.tolist())
    ]
    candidates.sort(reverse=True)
    chosen = OrderedDict()
    for _, layer_idx, expert_idx in candidates[:num_experts]:
        chosen.setdefault(layer_idx, []).append(expert_idx)
    return OrderedDict(sorted(chosen.items())), relevance


def routed_to_target(weights, target_idx, top_k):
    """E_* in S^(l)(x): the target is among the top-K experts of the token-averaged weights."""
    topk = weights.topk(top_k, dim=-1).indices
    return (topk == target_idx).any(dim=-1)


def assign_samples_to_layers(sample_weights, layer_targets, top_k, exclusive=True):
    """IDFF (Eq. 3.5): assign each forget sample to the first layer that routes it to its target.

    Returns ({layer_idx: LongTensor of sample positions}, unassigned positions).
    With ``exclusive=False`` (w/o-IDFF ablation) every layer receives all samples
    routed to its target expert, so samples can drive several layers.
    """
    num_samples = next(iter(sample_weights.values())).size(0)
    remaining = torch.ones(num_samples, dtype=torch.bool)
    assignment = OrderedDict()
    for layer_idx, target_idx in layer_targets.items():
        routed = routed_to_target(sample_weights[layer_idx], target_idx, top_k[layer_idx])
        matched = routed & remaining if exclusive else routed
        assignment[layer_idx] = matched.nonzero(as_tuple=True)[0]
        if exclusive:
            remaining &= ~matched
    unassigned = remaining.nonzero(as_tuple=True)[0] if exclusive else torch.empty(0, dtype=torch.long)
    return assignment, unassigned


def routing_distance(gate_probs, target_experts):
    """L_dist (Eq. 3.4): squared L2 distance between token routing weights and the one-hot target,
    averaged over tokens and samples. ``target_experts`` may list several experts (multi-hot)."""
    anchor = torch.zeros_like(gate_probs)
    anchor[..., list(target_experts)] = 1.0
    return (gate_probs - anchor).pow(2).sum(dim=-1).mean()
