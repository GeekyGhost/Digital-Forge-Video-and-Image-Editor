"""Pins the LTX geometry contract. Three adapters and two layers implement it; only this checks them.

The four disagreements this module exists to end -- axis multiple 64/32/64, frame grid
reject/reject/floor-snap, max frames none/193/none, pixel cap fixed/env/different -- are each
pinned below, so a future edit that reintroduces one fails here rather than after a 27 GB load.
"""

from __future__ import annotations

import pytest

from aiv_runtime.ltx_geometry import (
    AXIS_MULTIPLE,
    DEFAULT_MAX_FRAMES,
    DEFAULT_MAX_PIXELS,
    FRAMES_PER_LATENT,
    MIN_FRAMES,
    DEFAULT_MAX_PIXEL_FRAMES,
    env_max_pixels,
    latent_frames,
    pixel_frames,
    snap_axes,
    snap_axis,
    snap_canvas,
    snap_frames,
    validate_axes,
    validate_num_frames,
    validate_pixel_frames,
)


def test_the_axis_multiple_is_the_one_stage_rule():
    """64 is the TWO-stage divisor. `assert_resolution` computes `64 if is_two_stage else 32`,
    `ti2vid_one_stage.py` passes `is_two_stage=False`, and `retake.py` checks `% 32` directly."""
    assert AXIS_MULTIPLE == 32
    # 832x480 was retakeable but not generatable, purely because generation rounded to 64.
    assert snap_axes(832, 480) == (832, 480)
    assert snap_canvas(832, 480) == (832, 480)


def test_axes_round_up_and_stay_on_the_grid():
    for value in range(1, 2000):
        snapped = snap_axis(value)
        assert snapped % AXIS_MULTIPLE == 0
        assert snapped >= value
        assert snapped - value < AXIS_MULTIPLE
        assert snapped >= AXIS_MULTIPLE


def test_a_canvas_is_always_legal_not_usually_legal():
    for width, height in [
        (1920, 1080), (1080, 1920), (3840, 2160), (768, 512), (832, 480), (1, 1),
        (2000, 33), (33, 2000), (767, 511), (769, 513),
    ]:
        fitted_width, fitted_height = snap_canvas(width, height)
        validate_axes(fitted_width, fitted_height)


def test_the_frame_grid_snaps_rather_than_rejects():
    for requested in range(1, 400):
        snapped = snap_frames(requested)
        assert (snapped - 1) % FRAMES_PER_LATENT == 0
        assert snapped >= MIN_FRAMES
        # Nearest: never further away than half a latent frame, once above the floor.
        if requested >= MIN_FRAMES:
            assert abs(snapped - requested) <= FRAMES_PER_LATENT // 2


def test_a_legal_count_is_left_alone():
    for legal in (9, 17, 49, 121, 193):
        for direction in ("nearest", "up", "down"):
            assert snap_frames(legal, direction) == legal


def test_the_snap_direction_is_available_where_it_matters():
    # Foley floored unconditionally, which planned audio shorter than the clip -- half of why its
    # mux truncated the picture. The direction is now a choice, and nearest is the default.
    assert snap_frames(30) == 33
    assert snap_frames(30, "down") == 25
    assert snap_frames(30, "up") == 33
    assert snap_frames(28) == 25
    with pytest.raises(ValueError):
        snap_frames(30, "sideways")


def test_the_floor_holds_for_counts_below_one_latent_frame():
    for requested in range(1, MIN_FRAMES + 1):
        assert snap_frames(requested) == MIN_FRAMES
    with pytest.raises(ValueError):
        snap_frames(0)


def test_the_cost_metric_is_pixels_times_latent_frames():
    """Retake's own comment has said so since it was written; its guard tested pixels alone."""
    assert latent_frames(49) == 7
    assert latent_frames(121) == 16
    assert latent_frames(9) == 2
    # The measured datapoint: 768x512x49 peaked at 7,905 MiB of 16,376.
    assert pixel_frames(768, 512, 49) == 2_752_512
    # A short window at a higher resolution really is cheaper than a long one at a low resolution.
    assert pixel_frames(768, 768, 49) < pixel_frames(768, 512, 121)


def test_generation_now_has_a_frame_cap_at_all():
    """`--num-frames 481` was accepted by generation and OOMed after loading 27 GB."""
    assert DEFAULT_MAX_FRAMES == 193
    validate_num_frames(193)
    with pytest.raises(ValueError, match="over the 193"):
        validate_num_frames(481)
    # 481 is on the grid, so only the cap catches it.
    assert (481 - 1) % FRAMES_PER_LATENT == 0


def test_validate_is_the_last_resort_not_the_gate():
    validate_axes(768, 512)
    with pytest.raises(ValueError, match="multiple of 32"):
        validate_axes(770, 512)
    with pytest.raises(ValueError, match="budget"):
        validate_axes(1024, 1024)
    with pytest.raises(ValueError, match="8n \\+ 1"):
        validate_num_frames(120)
    with pytest.raises(ValueError, match="at least 9"):
        validate_num_frames(1)


def test_the_pixel_budget_override_keeps_its_documented_name(monkeypatch):
    assert env_max_pixels("AIV_LTX_MAX_PIXEL_FRAMES", "AIV_RETAKE_MAX_PIXELS") == DEFAULT_MAX_PIXELS
    # The old name still works: it is documented and in people's notes.
    monkeypatch.setenv("AIV_RETAKE_MAX_PIXELS", "1000000")
    assert env_max_pixels("AIV_LTX_MAX_PIXEL_FRAMES", "AIV_RETAKE_MAX_PIXELS") == 1_000_000
    # The newer name wins when both are set.
    monkeypatch.setenv("AIV_LTX_MAX_PIXEL_FRAMES", "2000000")
    assert env_max_pixels("AIV_LTX_MAX_PIXEL_FRAMES", "AIV_RETAKE_MAX_PIXELS") == 2_000_000
    monkeypatch.setenv("AIV_LTX_MAX_PIXEL_FRAMES", "not a number")
    with pytest.raises(ValueError, match="integer"):
        env_max_pixels("AIV_LTX_MAX_PIXEL_FRAMES")


def test_the_token_budget_adds_the_diagonal_the_two_caps_miss():
    """The production guard stops at the largest completed local 16-GB retake."""
    assert DEFAULT_MAX_PIXEL_FRAMES == 9_609_216

    # The measured and smaller profiles pass.
    validate_pixel_frames(768, 512, 121)
    validate_pixel_frames(768, 768, 49)
    validate_pixel_frames(736, 544, 185)

    # The unmeasured theoretical corner is no longer advertised as qualified.
    with pytest.raises(ValueError, match="pixel-frames"):
        validate_pixel_frames(768, 768, 193)
    message = str(pytest.raises(ValueError, validate_pixel_frames, 768, 768, 193, 1_000).value)
    assert "shortening the window buys resolution" in message


def test_a_short_window_at_high_resolution_is_cheaper_than_the_reverse():
    """Retake's own comment says so, and its guard tested pixels alone -- so it had this backwards."""
    short_and_wide = pixel_frames(768, 768, 49)
    long_and_narrow = pixel_frames(768, 512, 121)
    assert short_and_wide < long_and_narrow
    # The old pixel-only guard refused the cheaper one and accepted the dearer one.
    assert 768 * 768 > DEFAULT_MAX_PIXELS - 1  # at the pixel cap
    assert 768 * 512 < 768 * 768               # under it
