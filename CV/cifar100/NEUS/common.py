import os
import sys
from collections import OrderedDict

import torch


NEUS_DIR = os.path.dirname(os.path.abspath(__file__))
CIFAR_DIR = os.path.dirname(NEUS_DIR)
if CIFAR_DIR not in sys.path:
    sys.path.insert(0, CIFAR_DIR)

from MCU.common import (  # noqa: E402
    RANDOM_TARGET_METRICS,
    build_cifar100_datasets,
    build_eval_loaders,
    build_model,
    build_split_subsets,
    compute_metrics,
    compute_selection_score,
    load_checkpoint_state,
    load_split_indices,
    set_seed,
)


FLOW_TARGET_VOTE_THRESHOLD = 0.6
LAST_LAYER_TAKE_ALL = True
ROUTING_UNIT = "patch_majority_threshold_flow"


def collect_moe_gate_probs(model, inputs):
    x = model.patch_embed(inputs)
    layer_gate_probs = OrderedDict()
    for layer_idx, blk in enumerate(model.blocks):
        residual = x
        attn_in = blk.norm1(x)
        attn_out, _ = blk.attn(attn_in, attn_in, attn_in)
        x = residual + attn_out

        residual = x
        mlp_in = blk.norm2(x)
        if getattr(blk, "use_moe", False) and hasattr(blk.mlp, "experts"):
            gate_logits = blk.mlp.gate(mlp_in)
            gate_probs = torch.softmax(gate_logits, dim=-1)
            layer_gate_probs[layer_idx] = gate_probs
            mlp_out, _ = blk.mlp(mlp_in)
        else:
            mlp_out = blk.mlp(mlp_in)
        x = residual + mlp_out
    return layer_gate_probs


def get_moe_layers(model):
    moe_layers = []
    for layer_idx, blk in enumerate(model.blocks):
        if getattr(blk, "use_moe", False) and hasattr(blk.mlp, "experts"):
            moe_layers.append((layer_idx, len(blk.mlp.experts)))
    return moe_layers


def compute_patch_majority_experts(gate_probs):
    patch_probs = gate_probs[:, 1:, :]
    if patch_probs.size(1) == 0:
        patch_probs = gate_probs

    patch_top1 = patch_probs.argmax(dim=-1)
    num_experts = gate_probs.size(-1)
    expert_votes = torch.stack(
        [(patch_top1 == expert_idx).sum(dim=1) for expert_idx in range(num_experts)],
        dim=1,
    )
    return expert_votes.argmax(dim=1), expert_votes


def compute_target_vote_ratio(vote_counts, target_idx):
    total_votes = vote_counts.sum(dim=1).clamp_min(1)
    return vote_counts[:, target_idx].float() / total_votes.float()


def compute_layerwise_target_experts(model, dataloader, device):
    moe_layers = get_moe_layers(model)
    if not moe_layers:
        raise RuntimeError("No MoE layer found in the model.")

    all_sample_experts = {layer_idx: [] for layer_idx, _ in moe_layers}
    all_patch_vote_counts = {layer_idx: [] for layer_idx, _ in moe_layers}

    model.eval()
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            layer_gate_probs = collect_moe_gate_probs(model, inputs)
            for layer_idx, gate_probs in layer_gate_probs.items():
                sample_experts, vote_counts = compute_patch_majority_experts(gate_probs)
                all_sample_experts[layer_idx].append(sample_experts.cpu())
                all_patch_vote_counts[layer_idx].append(vote_counts.cpu())

    sample_experts = {
        layer_idx: torch.cat(expert_batches, dim=0)
        for layer_idx, expert_batches in all_sample_experts.items()
    }
    patch_vote_counts = {
        layer_idx: torch.cat(vote_batches, dim=0)
        for layer_idx, vote_batches in all_patch_vote_counts.items()
    }
    num_samples = next(iter(sample_experts.values())).size(0)
    remaining = torch.ones(num_samples, dtype=torch.bool)

    targets = OrderedDict()
    per_layer_stats = OrderedDict()
    for layer_position, (layer_idx, num_experts) in enumerate(moe_layers):
        current_sample_experts = sample_experts[layer_idx][remaining]
        current_vote_counts = patch_vote_counts[layer_idx][remaining]
        remaining_before = int(remaining.sum().item())
        is_last_layer = layer_position == len(moe_layers) - 1
        if remaining_before == 0:
            target_idx = int(sample_experts[layer_idx].bincount(minlength=num_experts).argmax().item())
            patch_vote_ratio = torch.zeros(num_experts).tolist()
            sample_top1_ratio = torch.zeros(num_experts).tolist()
            matched_count = 0
            strong_matched_count = 0
        else:
            counts = current_sample_experts.bincount(minlength=num_experts)
            target_idx = int(counts.argmax().item())
            patch_vote_ratio = (
                current_vote_counts.float().sum(dim=0)
                / max(current_vote_counts.sum().item(), 1)
            ).tolist()
            sample_top1_ratio = (counts.float() / remaining_before).tolist()

            full_target_vote_ratio = compute_target_vote_ratio(patch_vote_counts[layer_idx], target_idx)
            strong_matched = (
                remaining
                & (sample_experts[layer_idx] == target_idx)
                & (full_target_vote_ratio > FLOW_TARGET_VOTE_THRESHOLD)
            )
            matched = remaining.clone() if is_last_layer and LAST_LAYER_TAKE_ALL else strong_matched
            strong_matched_count = int(strong_matched.sum().item())
            matched_count = int(matched.sum().item())
            remaining &= ~matched

        targets[layer_idx] = target_idx
        per_layer_stats[layer_idx] = {
            "target_idx": target_idx,
            "patch_vote_ratio": patch_vote_ratio,
            "sample_top1_ratio": sample_top1_ratio,
            "routing_unit": ROUTING_UNIT,
            "flow_threshold": FLOW_TARGET_VOTE_THRESHOLD,
            "last_layer_take_all": LAST_LAYER_TAKE_ALL,
            "remaining_before": remaining_before,
            "strong_matched_count": strong_matched_count,
            "matched_count": matched_count,
            "remaining_after": int(remaining.sum().item()),
        }
    return targets, per_layer_stats


def get_trainable_param_names(model, layer_targets):
    trainable = set()
    for layer_idx, target_idx in layer_targets.items():
        blk = model.blocks[layer_idx]
        for name, _ in blk.mlp.gate.named_parameters(prefix=f"blocks.{layer_idx}.mlp.gate"):
            trainable.add(name)
        for name, _ in blk.mlp.experts[target_idx].named_parameters(prefix=f"blocks.{layer_idx}.mlp.experts.{target_idx}"):
            trainable.add(name)
    return trainable


def get_layer_param_names(model, layer_idx, target_idx):
    blk = model.blocks[layer_idx]
    names = set()
    for name, _ in blk.mlp.gate.named_parameters(prefix=f"blocks.{layer_idx}.mlp.gate"):
        names.add(name)
    for name, _ in blk.mlp.experts[target_idx].named_parameters(
        prefix=f"blocks.{layer_idx}.mlp.experts.{target_idx}"
    ):
        names.add(name)
    return names


def freeze_to_layer_targets(model, layer_targets):
    for param in model.parameters():
        param.requires_grad = False
    for layer_idx, target_idx in layer_targets.items():
        blk = model.blocks[layer_idx]
        for param in blk.mlp.gate.parameters():
            param.requires_grad = True
        for param in blk.mlp.experts[target_idx].parameters():
            param.requires_grad = True


def compute_fixed_flow_masks(model, inputs, layer_targets):
    """Assign samples to fixed target experts using thresholded patch-token voting."""
    with torch.no_grad():
        layer_gate_probs = collect_moe_gate_probs(model, inputs)

    remaining = torch.ones(inputs.size(0), dtype=torch.bool, device=inputs.device)
    flow_masks = OrderedDict()
    layer_items = list(layer_targets.items())
    for layer_position, (layer_idx, target_idx) in enumerate(layer_items):
        gate_probs = layer_gate_probs.get(layer_idx)
        if gate_probs is None:
            flow_masks[layer_idx] = torch.zeros_like(remaining)
            continue
        # target_idx is fixed before training; this only checks which samples route to it.
        routed_experts_by_patch_vote, vote_counts = compute_patch_majority_experts(gate_probs)
        target_vote_ratio = compute_target_vote_ratio(vote_counts, target_idx)
        strong_matched = (
            remaining
            & (routed_experts_by_patch_vote == target_idx)
            & (target_vote_ratio > FLOW_TARGET_VOTE_THRESHOLD)
        )
        is_last_layer = layer_position == len(layer_items) - 1
        matched = remaining.clone() if is_last_layer and LAST_LAYER_TAKE_ALL else strong_matched
        flow_masks[layer_idx] = matched
        remaining &= ~matched
    return flow_masks, remaining


def compute_anchor_loss(gate_probs, target_idx):
    anchor = torch.zeros_like(gate_probs)
    anchor[..., target_idx] = 1.0
    return torch.norm(gate_probs - anchor, p=2, dim=-1).mean()


def format_layer_targets(layer_targets):
    return ", ".join(f"L{layer_idx}->E{target_idx}" for layer_idx, target_idx in layer_targets.items())
