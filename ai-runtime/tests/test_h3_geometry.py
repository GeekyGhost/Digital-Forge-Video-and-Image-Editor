"""Pins the H3 geometry contract. Three layers implement it and only this file checks them."""

from __future__ import annotations

import pytest

from aiv_runtime.h3_geometry import (
    LOW_VRAM_SHORT_EDGE,
    MAX_FRAMES,
    MIN_FRAMES,
    align_num_frames,
    clamp_num_frames,
    resolve_canvas,
    validate_canvas,
    validate_num_frames,
)


def test_alignment_lands_on_the_vae_grid():
    for requested in range(1, 400):
        assert align_num_frames(requested) % 17 == 5
        assert align_num_frames(requested) >= requested


def test_alignment_leaves_a_valid_count_untouched():
    for valid in (5, 22, 124, 141, 345):
        assert align_num_frames(valid) == valid


def test_duration_window_bounds():
    # 124 frames is 5.167 s and 345 is 14.375 s. The next count up, 362, is 15.083 s -- over the
    # checkpoint's 15 s ceiling -- which is why MAX_FRAMES is not align_num_frames(15 * 24).
    assert MIN_FRAMES == 124
    assert MAX_FRAMES == 345
    assert MIN_FRAMES / 24 >= 5.0
    assert MAX_FRAMES / 24 <= 15.0
    assert align_num_frames(15 * 24) == 362
    assert 362 / 24 > 15.0


def test_clamp_holds_the_window():
    assert clamp_num_frames(1) == MIN_FRAMES
    assert clamp_num_frames(10_000) == MAX_FRAMES
    for requested in range(1, 500):
        clamped = clamp_num_frames(requested)
        assert MIN_FRAMES <= clamped <= MAX_FRAMES
        validate_num_frames(clamped)


@pytest.mark.parametrize(
    ("aspect", "expected"),
    [
        ((16, 9), (960, 544)),
        ((9, 16), (544, 960)),
        ((1, 1), (544, 544)),
        ((4, 3), (736, 544)),
        ((3, 4), (544, 736)),
        ((21, 9), (1280, 544)),
    ],
)
def test_low_vram_canvases_are_on_the_grid(aspect, expected):
    resolved = resolve_canvas(*aspect, short_edge=LOW_VRAM_SHORT_EDGE)
    assert resolved == expected
    validate_canvas(*resolved)


def test_default_short_edge_stays_inside_the_area_budget():
    for aspect in ((21, 9), (16, 9), (4, 3), (1, 1), (3, 4), (9, 16)):
        width, height = resolve_canvas(*aspect)
        validate_canvas(width, height)


def test_source_dimensions_can_be_passed_directly():
    # A 1920x1080 keyframe and a 16:9 request must resolve to the same canvas.
    assert resolve_canvas(1920, 1080, short_edge=LOW_VRAM_SHORT_EDGE) == resolve_canvas(
        16, 9, short_edge=LOW_VRAM_SHORT_EDGE
    )


def test_extreme_aspect_ratios_are_rejected():
    with pytest.raises(ValueError):
        resolve_canvas(5, 1)
    with pytest.raises(ValueError):
        validate_canvas(32 * 20, 32)


def test_off_grid_canvases_are_rejected():
    for width, height in ((960, 540), (768, 500), (0, 544), (-32, 544)):
        with pytest.raises(ValueError):
            validate_canvas(width, height)


def test_off_grid_frame_counts_are_rejected():
    for frames in (120, 121, 240, 360):
        with pytest.raises(ValueError):
            validate_num_frames(frames)
