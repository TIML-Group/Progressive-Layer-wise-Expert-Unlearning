from collections import OrderedDict

import torch
import torch.nn as nn

from neus_unlearn import BezierCurveModel


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.trainable = nn.Parameter(torch.tensor([1.0, 2.0]))
        self.frozen = nn.Parameter(torch.tensor([3.0, 4.0]))
        self.register_buffer("running", torch.tensor([5.0, 6.0]))

    def forward(self, x):
        return x + self.trainable.sum() + self.frozen.sum() + self.running.sum()


def test_neus_curve_endpoints_use_complete_start_and_target_states():
    model = TinyModel()
    start_state = OrderedDict(
        {
            "trainable": torch.tensor([1.0, 2.0]),
            "frozen": torch.tensor([3.0, 4.0]),
            "running": torch.tensor([5.0, 6.0]),
        }
    )
    end_state = OrderedDict(
        {
            "trainable": torch.tensor([10.0, 20.0]),
            "frozen": torch.tensor([30.0, 40.0]),
            "running": torch.tensor([50.0, 60.0]),
        }
    )

    curve = BezierCurveModel(model, start_state, end_state, trainable_names={"trainable"})

    state_t0 = curve.export_state_dict(torch.tensor([0.0]))
    state_t1 = curve.export_state_dict(torch.tensor([1.0]))

    for name, expected in start_state.items():
        assert torch.equal(state_t0[name], expected)
    for name, expected in end_state.items():
        assert torch.equal(state_t1[name], expected)

    assert curve.mid_params["trainable"].requires_grad
    assert curve.mid_params["frozen"].requires_grad


def test_neus_curve_mask_controls_midpoint_training_like_mcu():
    model = TinyModel()
    start_state = OrderedDict(
        {
            "trainable": torch.tensor([1.0, 2.0]),
            "frozen": torch.tensor([3.0, 4.0]),
            "running": torch.tensor([5.0, 6.0]),
        }
    )
    end_state = OrderedDict(
        {
            "trainable": torch.tensor([10.0, 20.0]),
            "frozen": torch.tensor([30.0, 40.0]),
            "running": torch.tensor([50.0, 60.0]),
        }
    )

    curve = BezierCurveModel(
        model,
        start_state,
        end_state,
        trainable_names={"trainable", "frozen"},
        mask={"frozen": 0},
    )

    assert curve.mid_params["trainable"].requires_grad
    assert not curve.mid_params["frozen"].requires_grad
