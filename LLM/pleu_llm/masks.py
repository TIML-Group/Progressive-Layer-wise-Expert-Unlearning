"""Gradient importance and parameter selection (MCU tensor masks, PLEU/SEPO entry masks)."""

from collections import OrderedDict

import torch
from tqdm import tqdm

from .data import lm_loss, to_device
from .modeling import input_device


def mean_gradients(model, loader, names, max_batches=None, desc="grad", dtype=torch.float32):
    """Average LM-loss gradient over ``loader`` for the named parameters (signed)."""
    params = dict(model.named_parameters())
    previous = {n: params[n].requires_grad for n in names}
    for n in names:
        params[n].requires_grad_(True)
    sums = OrderedDict((n, torch.zeros(params[n].shape, dtype=dtype, device=params[n].device)) for n in names)
    device, count = input_device(model), 0
    model.train()
    for idx, batch in enumerate(tqdm(loader, desc=desc, leave=False)):
        if max_batches is not None and idx >= max_batches:
            break
        model.zero_grad(set_to_none=True)
        lm_loss(model, to_device(batch, device)).backward()
        for n in names:
            if params[n].grad is not None:
                sums[n] += params[n].grad.to(dtype)
        count += 1
    model.zero_grad(set_to_none=True)
    for n, flag in previous.items():
        params[n].requires_grad_(flag)
    if count == 0:
        raise RuntimeError("No batch processed while computing gradients.")
    return OrderedDict((n, s / count) for n, s in sums.items())


def mcu_tensor_mask(retain_grads, forget_grads, forget_ratio, retain_ratio):
    sizes = {n: g.numel() for n, g in forget_grads.items()}
    total = sum(sizes.values())
    score = lambda grads: {n: g.float().norm().item() / max(g.numel(), 1) for n, g in grads.items()}  # noqa: E731
    retain_score, forget_score = score(retain_grads), score(forget_grads)
    protected, covered = set(), 0
    for n in sorted(retain_score, key=retain_score.get, reverse=True):
        if covered >= retain_ratio * total:
            break
        protected.add(n)
        covered += sizes[n]
    mask, selected = OrderedDict(), 0
    for n in sorted(forget_score, key=forget_score.get, reverse=True):
        mask[n] = n not in protected and selected < forget_ratio * total
        selected += sizes[n] if mask[n] else 0
    return mask, {"selected_ratio": selected / max(total, 1), "protected_ratio": covered / max(total, 1)}


def sepo_mask(retain_importance, forget_importance, k_r, k_f):
    """Protect the top-k_r entries by retain importance; activate the top-k_f of the rest by
    forget importance. Works across the tensors of one expert, which may sit on one device."""
    names = list(forget_importance)
    device = forget_importance[names[0]].device
    flat_r = torch.cat([retain_importance[n].flatten().to(device) for n in names])
    flat_f = torch.cat([forget_importance[n].flatten().to(device) for n in names])
    d = flat_f.numel()
    protected = torch.zeros(d, dtype=torch.bool, device=device)
    num_protected = int(d * k_r)
    if num_protected:
        protected[flat_r.topk(num_protected).indices] = True
    active = torch.zeros_like(protected)
    num_active = int((d - num_protected) * k_f)
    if num_active:
        active[flat_f.masked_fill(protected, float("-inf")).topk(num_active).indices] = True
    mask, offset = OrderedDict(), 0
    for n in names:
        numel = forget_importance[n].numel()
        mask[n] = active[offset:offset + numel].view(forget_importance[n].shape).to(forget_importance[n].device)
        offset += numel
    return mask, {"num_params": d, "num_protected": num_protected, "num_active": num_active}
