"""Gradient-based parameter selection: tensor-level masks for MCU and the
entry-level SEPO masks of PLEU (Eq. 3.11)."""

from collections import OrderedDict

import torch
import torch.nn.functional as F
from tqdm import tqdm


def mean_gradients(model, loader, device, param_names, max_batches=None, desc="grad"):
    """Gradient of the average cross-entropy over ``loader`` w.r.t. ``param_names`` (signed)."""
    params = dict(model.named_parameters())
    names = list(param_names)
    sums = OrderedDict((name, torch.zeros_like(params[name])) for name in names)
    previous = {name: params[name].requires_grad for name in names}
    for name in names:
        params[name].requires_grad_(True)
    total = 0
    model.eval()
    for batch_idx, (inputs, targets) in enumerate(tqdm(loader, desc=desc, leave=False)):
        if max_batches is not None and batch_idx >= max_batches:
            break
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        loss = F.cross_entropy(model(inputs)[0], targets, reduction="sum")
        grads = torch.autograd.grad(loss, [params[name] for name in names], allow_unused=True)
        for name, grad in zip(names, grads):
            if grad is not None:
                sums[name] += grad.detach()
        total += targets.size(0)
    for name, flag in previous.items():
        params[name].requires_grad_(flag)
    if total == 0:
        raise RuntimeError("No sample was processed while computing gradients.")
    return OrderedDict((name, value / total) for name, value in sums.items())


# ---------------------------------------------------------------- MCU

def mcu_tensor_mask(retain_grads, forget_grads, forget_ratio, retain_ratio):
    """MCU parameter selection at tensor granularity.

    Tensors are scored by ||grad||_2 / numel. Tensors with the highest retain score
    are protected until ``retain_ratio`` of all parameters is covered; among the rest,
    the highest forget scores are activated until ``forget_ratio`` of all parameters.
    """
    sizes = {name: grad.numel() for name, grad in forget_grads.items()}
    total = sum(sizes.values())

    def score(grads):
        return {name: grad.norm(p=2).item() / max(grad.numel(), 1) for name, grad in grads.items()}

    retain_score, forget_score = score(retain_grads), score(forget_grads)
    protected, covered = set(), 0
    for name in sorted(retain_score, key=retain_score.get, reverse=True):
        if covered >= retain_ratio * total:
            break
        protected.add(name)
        covered += sizes[name]

    mask, selected = OrderedDict(), 0
    for name in sorted(forget_score, key=forget_score.get, reverse=True):
        take = name not in protected and selected < forget_ratio * total
        mask[name] = take
        selected += sizes[name] if take else 0
    return mask, {"selected_ratio": selected / max(total, 1), "protected_ratio": covered / max(total, 1)}


# ---------------------------------------------------------------- SEPO

def sepo_mask(retain_importance, forget_importance, k_r, k_f):
    """Binary mask over the flattened parameters of one target expert.

    The top ``k_r`` fraction by retain importance forms the protected set P; among
    the entries outside P, the top ``k_f`` fraction by forget importance is activated.
    Inputs are {name: |dL/dθ|}; returns {name: bool tensor} and statistics.
    """
    names = list(forget_importance)
    flat_r = torch.cat([retain_importance[name].flatten() for name in names]).float()
    flat_f = torch.cat([forget_importance[name].flatten() for name in names]).float()
    d = flat_f.numel()

    protected = torch.zeros(d, dtype=torch.bool, device=flat_f.device)
    num_protected = int(d * k_r)
    if num_protected > 0:
        protected[flat_r.topk(num_protected).indices] = True

    active = torch.zeros_like(protected)
    num_active = int((d - num_protected) * k_f)
    if num_active > 0:
        candidates = flat_f.masked_fill(protected, float("-inf"))
        active[candidates.topk(num_active).indices] = True

    mask, offset = OrderedDict(), 0
    for name in names:
        numel = forget_importance[name].numel()
        mask[name] = active[offset:offset + numel].view_as(forget_importance[name])
        offset += numel
    return mask, {"num_params": d, "num_protected": num_protected, "num_active": num_active}
