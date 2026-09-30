from __future__ import annotations

import json
import numpy as np
import pytest
from PIL import Image

from types import SimpleNamespace

import aiv_runtime.mask_effects as mask_effects
from aiv_runtime.mask_effects import (
    EFFECTS, TRANSITIONS, _blend, _extract_clip_matte, _transform_canvas, animated_matte,
    apply_effect, parse_color, parse_gradient, render_gradient, sample_animation, shape_mask,
    transition_alpha,
)


def test_parse_color_accepts_short_and_long_hex() -> None:
    assert parse_color("#00f0ff") == (0, 240, 255)
    assert parse_color("#f80") == (255, 136, 0)
    with pytest.raises(ValueError, match="rrggbb"):
        parse_color("cyan")


def test_mask_shape_can_expand_feather_and_invert() -> None:
    source = np.zeros((25, 25), dtype=np.uint8)
    source[12, 12] = 255
    grown = shape_mask(Image.fromarray(source), feather=0, expand=3, invert=False)
    assert np.count_nonzero(np.asarray(grown)) > 1
    inverted = shape_mask(Image.fromarray(source), feather=0, expand=0, invert=True)
    assert np.asarray(inverted)[12, 12] == 0
    assert np.asarray(inverted)[0, 0] == 255


def test_transient_envelope_keeps_short_clip_frame_count(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mask_effects, "_audio_samples", lambda _args: np.linspace(-1, 1, 400, dtype=np.float32))
    args = SimpleNamespace(audio_reactive="transients", reactive_smoothing=0.4)
    values = mask_effects.audio_envelope(args, 4)
    assert values.shape == (4,)
    assert np.all((0 <= values) & (values <= 1))


@pytest.mark.parametrize("effect", EFFECTS)
def test_every_effect_preserves_frame_geometry(effect: str) -> None:
    frame = Image.new("RGB", (64, 40), (38, 48, 58))
    mask = Image.new("L", frame.size, 0)
    data = np.asarray(mask).copy()
    data[8:34, 20:48] = 255
    rendered = apply_effect(frame, Image.fromarray(data), effect, (0, 240, 255), 1.0)
    assert rendered.mode == "RGBA"
    assert rendered.size == frame.size


@pytest.mark.parametrize("transition", TRANSITIONS)
def test_every_transition_has_full_frame_geometry_and_valid_alpha(transition: str) -> None:
    matte = transition_alpha((80, 48), transition, 0.5, frame_index=7, speed=1.2)
    pixels = np.asarray(matte)
    assert matte.mode == "L"
    assert matte.size == (80, 48)
    assert np.all((0 <= pixels) & (pixels <= 255))


def test_directional_transitions_reveal_from_the_named_side() -> None:
    top = np.asarray(transition_alpha((80, 48), "top-to-bottom", 0.25))
    center = np.asarray(transition_alpha((80, 48), "center-to-edges", 0.25))
    edges = np.asarray(transition_alpha((80, 48), "edges-to-center", 0.25))
    assert top[2, 40] > top[-3, 40]
    assert center[24, 40] > center[2, 2]
    assert edges[2, 2] > edges[24, 40]


def test_procedural_mattes_animate_without_losing_frame_geometry() -> None:
    source = Image.new("L", (96, 64), 0)
    pixels = np.asarray(source).copy()
    pixels[14:58, 34:62] = 255
    source = Image.fromarray(pixels)
    for style in ("subject", "flame", "rings", "burst"):
        first = animated_matte(source, style, 0.1)
        second = animated_matte(source, style, 0.6)
        assert first.mode == "L"
        assert first.size == source.size
        if style != "subject":
            assert not np.array_equal(np.asarray(first), np.asarray(second))


def test_effect_transform_moves_and_scales_a_matte_on_its_canvas() -> None:
    source = Image.new("L", (100, 80), 0)
    pixels = np.asarray(source).copy()
    pixels[30:50, 40:60] = 255
    moved = _transform_canvas(Image.fromarray(pixels), (.75, .25), 50, 30)
    assert moved.size == source.size
    assert moved.getbbox() is not None
    left, top, right, bottom = moved.getbbox()
    assert left > 55 and top < 20
    assert right - left < 20 and bottom - top < 20


def test_gradient_contract_renders_color_and_alpha() -> None:
    gradient = parse_gradient("")
    rendered = render_gradient((80, 48), gradient, animation_phase=.25)
    pixels = np.asarray(rendered)
    assert rendered.mode == "RGBA"
    assert rendered.size == (80, 48)
    assert np.ptp(pixels[..., :3]) > 0
    assert pixels[..., 3].min() < pixels[..., 3].max()


def test_behind_placement_restores_clean_subject_layer(tmp_path) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    plate = Image.new("RGB", (80, 48), (24, 36, 52))
    plate.save(frames / "00000.jpg")
    subject = np.zeros((48, 80), dtype=np.uint8)
    subject[12:42, 28:53] = 255
    args = SimpleNamespace(
        color="#00f0ff", secondary_color="#ff6b1a", feather=0, expand=2, invert=False,
        intensity=1.0, reactive_amount=0.0, effect="power-up", effect_speed=1.0,
        particle_amount=1.0, effect_x=.5, effect_y=.5, effect_scale=100.0,
        effect_rotation=0.0, pulse_amount=.2, pulse_rate=1.0, loop=True,
        matte_style="flame", placement="behind", effect_opacity=1.0,
        effect_clip_color="gradient", gradient_enabled=True, gradient_json="",
        transition_in="none", transition_out="none", transition_in_duration=.5,
        transition_out_duration=.5, transition_origin="subject", occlusion_feather=0,
        occlusion_expand=0, keyframes_json="[]", displacement=0, motion_blur=0,
        key_mode="luma", key_threshold=.05, key_softness=.1, key_color="#00ff00",
        color_spill=0, blend_mode="screen", grain=0, rear_amount=1,
        subject_amount=0, front_amount=0, light_wrap=0,
    )
    count = mask_effects.render_frames(
        args, [Image.fromarray(subject)], np.zeros(1, dtype=np.float32), tmp_path,
        {"fps": 24.0},
    )
    assert count == 1
    rendered = np.asarray(Image.open(tmp_path / "rendered" / "00000.png"))
    decoded_plate = np.asarray(Image.open(frames / "00000.jpg"))
    assert np.array_equal(rendered[24, 40], decoded_plate[24, 40])
    assert not np.array_equal(rendered[8, 40], decoded_plate[8, 40])


def test_radial_transition_uses_the_requested_anchor() -> None:
    left = np.asarray(transition_alpha((100, 60), "center-to-edges", .12, anchor=(.1, .5)))
    assert left[30, 10] > left[30, 80]


@pytest.mark.parametrize("mode", ("normal", "screen", "add", "multiply", "overlay", "soft-light"))
def test_blend_modes_respect_layer_alpha(mode: str) -> None:
    base = Image.new("RGBA", (8, 8), (80, 100, 120, 255))
    layer = Image.new("RGBA", (8, 8), (220, 80, 20, 0))
    assert np.array_equal(np.asarray(_blend(base, layer, mode)), np.asarray(base))


def test_alpha_luma_and_chroma_extraction_are_distinct() -> None:
    pixels = np.zeros((2, 3, 4), dtype=np.uint8)
    pixels[:] = (0, 255, 0, 128)
    pixels[:, 1] = (255, 255, 255, 255)
    layer = Image.fromarray(pixels, mode="RGBA")
    base = dict(key_threshold=.1, key_softness=.1, key_color="#00ff00")
    alpha = np.asarray(_extract_clip_matte(layer, SimpleNamespace(**base, key_mode="alpha")))
    luma = np.asarray(_extract_clip_matte(layer, SimpleNamespace(**base, key_mode="luma")))
    chroma = np.asarray(_extract_clip_matte(layer, SimpleNamespace(**base, key_mode="chroma")))
    assert alpha[0, 0] == 128
    assert luma[0, 0] > 0
    assert chroma[0, 0] == 0 and chroma[0, 1] > 0


def test_transform_keyframes_interpolate_with_easing() -> None:
    values = lambda x: {"effectX": x, "effectY": .5, "effectScale": 100,
                        "effectRotation": 0, "effectOpacity": 1, "intensity": 1}
    args = SimpleNamespace(effect_x=0, effect_y=.5, effect_scale=100, effect_rotation=0,
                           effect_opacity=1, intensity=1, keyframes_json=json.dumps([
                               {"at": 0, "easing": "linear", "values": values(0)},
                               {"at": 1, "easing": "smooth", "values": values(1)},
                           ]))
    assert sample_animation(args, .5)["effectX"] == pytest.approx(.5)
