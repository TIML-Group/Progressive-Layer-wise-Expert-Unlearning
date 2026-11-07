"""Bezier curves on model weights via ``torch.nn.utils.parametrize``.

Each curved weight is replaced by phi(t) = (1-t)^2 start + 2t(1-t) mid + t^2 end, where
``mid`` (float32) is the trainable control point and ``start``/``end`` are buffers on the
weight's own device, so the model can stay sharded with ``device_map="auto"``. An optional
boolean mask keeps unselected entries at ``start`` (SEPO).
"""

import torch
import torch.nn as nn
from torch.nn.utils import parametrize


class CurveTime:
    def __init__(self, t=0.5):
        self.t = float(t)


class BezierWeight(nn.Module):
    def __init__(self, start, end, clock, mask=None):
        super().__init__()
        self.register_buffer("start", start.detach())  # shares storage with the frozen original weight
        self.register_buffer("end", end.detach().to(start.device, start.dtype).clone())
        self.register_buffer("mask", None if mask is None else mask.to(start.device, torch.bool).clone())
        self.clock = clock

    def forward(self, mid):
        t = self.clock.t
        start = self.start.float()
        value = (1 - t) ** 2 * start + 2 * t * (1 - t) * mid.float() + t ** 2 * self.end.float()
        if self.mask is not None:
            value = torch.where(self.mask, value, start)
        return value.to(self.start.dtype)


class ParamCurve:
    """Puts a set of named weights of ``model`` on Bezier curves towards ``end_tensors``."""

    def __init__(self, model, end_tensors, trainable=None, masks=None, t=0.5, mid_dtype=torch.float32):
        self.model = model
        self.clock = CurveTime(t)
        self.entries = {}
        modules = dict(model.named_modules())
        trainable = set(end_tensors if trainable is None else trainable)
        masks = masks or {}
        for name, end in end_tensors.items():
            module_name, attr = name.rsplit(".", 1)
            module = modules[module_name]
            original = getattr(module, attr)
            end = end.to(original.device, original.dtype)
            mid = nn.Parameter((0.5 * (original.detach().float() + end.float())).to(mid_dtype),
                               requires_grad=name in trainable)
            setattr(module, attr, mid)
            parametrize.register_parametrization(
                module, attr, BezierWeight(original.detach(), end, self.clock, masks.get(name)), unsafe=True)
            self.entries[name] = (module, attr, original)

    def set_t(self, t):
        self.clock.t = float(t)

    def control_points(self):
        return [getattr(module.parametrizations, attr).original for module, attr, _ in self.entries.values()]

    def trainable_parameters(self):
        return [p for p in self.control_points() if p.requires_grad]

    def num_trainable(self):
        total = 0
        for (module, attr, _), param in zip(self.entries.values(), self.control_points()):
            if param.requires_grad:
                mask = getattr(module.parametrizations, attr)[0].mask
                total += int(mask.sum()) if mask is not None else param.numel()
        return total

    @torch.no_grad()
    def bake(self, t):
        """Write phi(t) into the original weights and remove the parametrizations."""
        self.set_t(t)
        values = {}
        for name, (module, attr, original) in self.entries.items():
            value = getattr(module, attr).detach().clone()
            parametrize.remove_parametrizations(module, attr, leave_parametrized=False)
            original.copy_(value)
            setattr(module, attr, original)
            values[name] = original
        self.entries.clear()
        return values
