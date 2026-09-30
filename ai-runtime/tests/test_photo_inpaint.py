"""LaMa output range (found by the VFX Phase 5 clean plate: a 0..1 output read as 0..255 was black)."""

import numpy as np

from aiv_runtime.photo_inpaint import to_uint8


def test_a_zero_to_one_output_is_scaled():
    assert to_uint8(np.array([[[0.0, 0.5, 1.0]]], np.float32)).tolist() == [[[0, 128, 255]]]


def test_a_zero_to_255_output_is_kept():
    assert to_uint8(np.array([[[0.0, 127.6, 300.0]]], np.float32)).tolist() == [[[0, 128, 255]]]
