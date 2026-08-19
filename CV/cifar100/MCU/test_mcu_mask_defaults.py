import os

from mcu_unlearn import resolve_default_mask_path


def test_default_mask_path_random():
    path = resolve_default_mask_path("random")
    assert path.endswith(os.path.join("MCU", "masks", "mask_random_k0.5_kr0.0.pt"))


def test_default_mask_path_class():
    path = resolve_default_mask_path("class")
    assert path.endswith(os.path.join("MCU", "masks", "mask_class_k0.5_kr0.0.pt"))
