"""Quadratic Bezier curves in parameter space (mode connectivity, Eq. 3.6 / 3.12)."""

from collections import OrderedDict

import torch
import torch.nn as nn

try:
    from torch.func import functional_call
except ImportError:  # torch < 2.0
    from torch.nn.utils.stateless import functional_call


def bezier_coeffs(t):
    return (1.0 - t) ** 2, 2.0 * t * (1.0 - t), t ** 2


class BezierCurve(nn.Module):
    """phi(t) = (1-t)^2 start + 2t(1-t) mid + t^2 end over a set of named tensors.

    ``trainable`` names the tensors whose control point is optimized (others keep the
    midpoint initialization but still move along the curve, as in MCU).
    ``entry_mask`` ({name: bool tensor}) restricts the curve to selected entries:
    phi~(t) = start + M * (phi(t) - start), so masked-out entries stay at ``start`` (SEPO).
    """

    def __init__(self, start, end, trainable=None, entry_mask=None):
        super().__init__()
        self.names = list(start)
        trainable = set(self.names if trainable is None else trainable)
        self._safe = {name: name.replace(".", "__") for name in self.names}
        self.mid = nn.ParameterDict()
        for name in self.names:
            s, e = start[name].detach(), end[name].detach()
            self.register_buffer(f"start__{self._safe[name]}", s.clone())
            self.register_buffer(f"end__{self._safe[name]}", e.clone())
            self.mid[self._safe[name]] = nn.Parameter(0.5 * (s + e), requires_grad=name in trainable)
        self.entry_mask = {}
        for name, mask in (entry_mask or {}).items():
            self.register_buffer(f"mask__{self._safe[name]}", mask.to(dtype=start[name].dtype).clone())
            self.entry_mask[name] = f"mask__{self._safe[name]}"

    def start(self, name):
        return getattr(self, f"start__{self._safe[name]}")

    def end(self, name):
        return getattr(self, f"end__{self._safe[name]}")

    def point(self, t):
        """{name: tensor} on the (masked) curve at ``t``; differentiable w.r.t. the control points."""
        c0, c1, c2 = bezier_coeffs(t)
        state = OrderedDict()
        for name in self.names:
            start = self.start(name)
            value = c0 * start + c1 * self.mid[self._safe[name]] + c2 * self.end(name)
            if name in self.entry_mask:
                value = start + getattr(self, self.entry_mask[name]) * (value - start)
            state[name] = value
        return state

    @torch.no_grad()
    def export(self, t):
        return OrderedDict((name, value.detach().clone()) for name, value in self.point(t).items())

    def trainable_parameters(self):
        return [param for param in self.mid.values() if param.requires_grad]

    def num_trainable(self):
        total = 0
        for name in self.names:
            param = self.mid[self._safe[name]]
            if not param.requires_grad:
                continue
            total += int(getattr(self, self.entry_mask[name]).sum().item()) if name in self.entry_mask else param.numel()
        return total


class CurveModel(nn.Module):
    """Runs ``model`` with the curve tensors substituted for the matching parameters."""

    def __init__(self, model, curve):
        super().__init__()
        self.model = model
        self.curve = curve

    def forward(self, inputs, t):
        return functional_call(self.model, self.curve.point(t), (inputs,))


class FixedPointModel(nn.Module):
    """Evaluation wrapper for one point on the curve (used with the shared evaluate helper)."""

    def __init__(self, curve_model, t):
        super().__init__()
        self.curve_model = curve_model
        self.t = t

    def forward(self, inputs):
        return self.curve_model(inputs, self.t)


def t_grid(start, end, num_points):
    return torch.linspace(start, end, steps=num_points).tolist()
