"""Contract tests for the three LTX adapters.

These files had no Python coverage at all, which is how four different geometry rules and a mode
that could not have run ended up living in them side by side. Everything here is parser- and
argument-level: none of it loads a checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import struct
import subprocess
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

from aiv_runtime import ltx_foley_v2a, ltx_single_stage

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")


def _modes(parser: argparse.ArgumentParser) -> tuple[str, ...]:
    for action in parser._actions:
        if action.dest == "mode":
            return tuple(action.choices)
    raise AssertionError("the parser has no --mode argument")


def test_the_single_pass_worker_no_longer_offers_a_mode_it_could_not_run():
    """`video-audio` built T2AOneStagePipeline with no `quantization=` and `audio_guider_params=None`.

    Every other pipeline in the project passes a quantization policy; without one the meta model's
    linears stay BF16 while the FP8 checkpoint supplies `.weight_scale` keys, and the build fails
    on a missing parameter. It was never reached: `generate_bridge` routes a `video-audio` request
    to `ltx_foley_v2a.py`, which is picture-synced where this was prompt-only. `--dry-run` returns
    before loading, so nothing ever caught it.
    """
    assert _modes(ltx_single_stage.build_parser()) == ("first-last", "text", "image")
    assert "video-audio" not in ltx_single_stage.MODE_REQUIREMENTS
    assert not hasattr(ltx_single_stage, "_generate_audio_for_video")
    # The probe existed only to length the audio for that mode.
    assert not hasattr(ltx_single_stage, "_probe_video")


def test_the_single_pass_worker_takes_no_source_clip():
    """`--video-path` was that mode's only input. Rust stops sending it on this route too."""
    parser = ltx_single_stage.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            "--checkpoint-path", "c", "--gemma-root", "g", "--output-path", "o.mp4",
            "--prompt", "p", "--video-path", "clip.mp4",
        ])


def test_every_remaining_mode_still_declares_what_it_needs():
    assert set(_modes(ltx_single_stage.build_parser())) == set(ltx_single_stage.MODE_REQUIREMENTS)
    assert ltx_single_stage.MODE_REQUIREMENTS["first-last"] == ("first_frame", "last_frame")
    assert ltx_single_stage.MODE_REQUIREMENTS["image"] == ("first_frame",)
    assert ltx_single_stage.MODE_REQUIREMENTS["text"] == ()


# ------------------------------------------------------------------ the audio mux


def _make_clip(path: Path, frames: int, fps: int) -> None:
    subprocess.run(
        [
            FFMPEG or "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-f", "lavfi", "-i", f"testsrc=size=160x96:rate={fps}",
            "-frames:v", str(frames), "-pix_fmt", "yuv420p", str(path),
        ],
        check=True,
    )


def _make_silence(path: Path, seconds: float, rate: int = 48000) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(struct.pack("<h", 0) * 2 * int(seconds * rate))


def _video_frames(path: Path) -> int:
    result = subprocess.run(
        [
            FFPROBE or "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
            "-show_entries", "stream=nb_read_frames", "-of", "json", str(path),
        ],
        capture_output=True, text=True, check=True,
    )
    return int(json.loads(result.stdout)["streams"][0]["nb_read_frames"])


@pytest.mark.skipif(FFMPEG is None or FFPROBE is None, reason="FFmpeg is unavailable")
def test_the_foley_mux_returns_the_users_picture_whole(tmp_path: Path):
    """`-shortest` reconciled the picture and the soundtrack by discarding whichever ran out first.

    30 frames is deliberately not 8n+1. Planning used to floor it to 25, so the audio was always
    shorter than the clip and the mux cut the copied video stream to match -- up to 8 frames, a
    third of a second at 24 fps, removed from footage the user never asked to edit. It now snaps
    to the nearest count, 33, so the audio runs slightly LONG and the explicit `-t` trims it.
    Either way the picture must come back whole, which is what this asserts.
    """
    source_path = tmp_path / "source.mp4"
    _make_clip(source_path, frames=30, fps=24)

    source = ltx_foley_v2a.probe(str(source_path))
    layout = ltx_foley_v2a.plan(source)
    assert layout["frames"] == 33, "30 frames must snap to the nearest 8n+1, not the one below"

    wav = tmp_path / "foley.wav"
    _make_silence(wav, seconds=layout["frames"] / layout["fps"])

    output = tmp_path / "scored.mp4"
    ltx_foley_v2a.mux(
        SimpleNamespace(video_path=str(source_path), output_path=str(output)), wav, source
    )

    assert _video_frames(output) == 30, "the mux edited the user's picture"
    # And the audio was padded up rather than the picture cut down.
    scored = ltx_foley_v2a.probe(str(output))
    assert math.isclose(scored["duration"], source["duration"], abs_tol=0.01)
    assert scored["hasAudio"]


# ------------------------------------------------------------------ negative prompts


def _args(**overrides) -> argparse.Namespace:
    # `nag_scale` is part of the refusal's input now: asking for NAG is the exemption, because it
    # carries a negative prompt without an unconditional branch.
    base = dict(negative_prompt="", cfg_scale=1.0, audio_cfg_scale=1.0, nag_scale=0.0)
    base.update(overrides)
    return argparse.Namespace(**base)


def test_a_negative_prompt_without_guidance_is_refused_not_dropped():
    """`guided` is False at CFG 1.0, so the negative prompt was never encoded and never mentioned.

    The user waited out a multi-minute generation to find their instruction had no effect, with
    nothing in the log to say why.
    """
    with pytest.raises(ValueError, match="silently ignored"):
        ltx_single_stage._refuse_a_negative_prompt_that_cannot_be_used(
            _args(negative_prompt="blurry, watermark")
        )


@pytest.mark.parametrize("scales", [(1.5, 1.0), (1.0, 1.5), (2.5, 2.5)])
def test_either_scale_above_one_is_enough_to_use_it(scales):
    video, audio = scales
    ltx_single_stage._refuse_a_negative_prompt_that_cannot_be_used(
        _args(negative_prompt="blurry", cfg_scale=video, audio_cfg_scale=audio)
    )


@pytest.mark.parametrize("empty", ["", "   ", "\n"])
def test_no_negative_prompt_is_never_an_error(empty):
    ltx_single_stage._refuse_a_negative_prompt_that_cannot_be_used(_args(negative_prompt=empty))


def test_asking_for_nag_is_the_exemption():
    """NAG carries the negative prompt without an unconditional branch, so nothing is dropped."""
    ltx_single_stage._refuse_a_negative_prompt_that_cannot_be_used(
        _args(negative_prompt="blurry, watermark", nag_scale=3.0)
    )
    # And with NAG off it is still refused, so the exemption is the only thing that changed.
    with pytest.raises(ValueError, match="silently ignored"):
        ltx_single_stage._refuse_a_negative_prompt_that_cannot_be_used(
            _args(negative_prompt="blurry, watermark", nag_scale=0.0)
        )


# ------------------------------------------------------------------ IC-LoRA factors


def test_both_ic_lora_reference_factors_are_read_and_passed():
    """The spatial factor was read; the temporal one was left to default to 1.

    `append_ic_lora_reference_video_conditionings` takes both, and `temporal_subsample` implements
    the temporal one -- keep frame 0, then every Nth. Reading one and not the other would feed the
    reference video at the wrong temporal density relative to training the moment a LoRA declared
    a factor, and that presents as bad sync rather than as a missing argument.
    """
    source = Path(ltx_foley_v2a.__file__).read_text(encoding="utf-8")
    generate = source.split("def generate_audio(")[1].split("\ndef ")[0]

    assert "read_lora_reference_downscale_factor(args.lora_path)" in generate
    assert "read_lora_reference_temporal_scale_factor(args.lora_path)" in generate
    assert "reference_downscale_factor=downscale_factor," in generate
    assert "reference_temporal_scale_factor=temporal_scale_factor," in generate


def test_the_conditioning_stage_scale_is_a_dial_not_a_constant():
    """`layout["width"] // 2` was a fossil of the two-stage DubIt design this adapter does not have.

    Still 0.5 by default, so behaviour is unchanged until the peak-VRAM print has been read at 1.0
    on a real clip. What changed is that it is now nameable, and that the stage size is snapped
    back onto the grid rather than divided blindly -- half of a 32-aligned axis is not always
    32-aligned.
    """
    defaults = {action.dest: action.default for action in ltx_foley_v2a.build_parser()._actions}
    assert defaults["stage_scale"] == ltx_foley_v2a.DEFAULT_STAGE_SCALE == 0.5

    source = Path(ltx_foley_v2a.__file__).read_text(encoding="utf-8")
    assert 'layout["width"] // 2' not in source
    assert "snap_axes(" in source

    # 480 // 2 is 240, which is off the 32 grid; the snap is what stops that reaching the encoder.
    from aiv_runtime.ltx_geometry import snap_axes
    assert snap_axes(round(832 * 0.5), round(480 * 0.5)) == (416, 256)


def test_the_foley_grid_is_the_one_stage_grid():
    from aiv_runtime import ltx_geometry
    assert ltx_foley_v2a.SPATIAL_MULTIPLE == ltx_geometry.AXIS_MULTIPLE == 32
    assert ltx_foley_v2a.FRAME_MULTIPLE == ltx_geometry.FRAMES_PER_LATENT == 8


# ------------------------------------------------------------------ teardown


def test_the_denoiser_is_released_before_the_decoders_run():
    """`teardown()` is not the memory reclaim; `to("meta")` is.

    Without it the non-block weights and the GPU buffer slot stayed resident while the video
    decoder, the audio decoder and the vocoder were built and run -- the moment of peak decode
    pressure. Foley gets the correct teardown for free by going through the pinned tree's own
    `_streaming_model` context manager; this adapter builds its wrapper by hand and had to be
    told.
    """
    source = Path(ltx_single_stage.__file__).read_text(encoding="utf-8")
    finally_block = source.split("    finally:\n")[1].split("    if torch.cuda.is_available():")[0]

    assert "streaming_wrapper.teardown()" in finally_block
    assert 'streaming_wrapper.to("meta")' in finally_block
    assert "cleanup_memory()" in finally_block


def test_the_teardown_matches_the_pinned_trees_own_context_manager():
    """Checked against upstream rather than against a remembered list of three calls."""
    blocks = (
        Path(ltx_single_stage.__file__).resolve().parents[2]
        / "vendor/LTX-2/packages/ltx-pipelines/src/ltx_pipelines/utils/blocks.py"
    )
    if not blocks.is_file():
        pytest.skip("the pinned LTX tree is not installed on this machine")

    upstream = blocks.read_text(encoding="utf-8")
    manager = upstream.split("def _streaming_model(")[1].split("\ndef ")[0]
    # What upstream does on the way out, in order.
    assert "wrapped.teardown()" in manager
    assert 'wrapped.to("meta")' in manager
    assert "cleanup_memory()" in manager

    ours = Path(ltx_single_stage.__file__).read_text(encoding="utf-8")
    for call in ("teardown()", '.to("meta")', "cleanup_memory()"):
        assert call in ours, f"upstream releases with {call} and this adapter does not"


# ------------------------------------------------------------------ retake guidance


def test_the_guidance_levers_live_in_the_preset_not_in_the_parser():
    """They used to be argparse defaults; now the parser holds sentinels and the preset holds
    the numbers, so there is one place to read what a retake will actually run.

    The reference values themselves are unchanged -- what changed is which preset ships.
    """
    from aiv_runtime import ltx_retake_16gb

    defaults = {action.dest: action.default for action in ltx_retake_16gb.build_parser()._actions}
    for axis in ("cfg_scale", "audio_cfg_scale", "stg_scale", "modality_scale", "rescale_scale"):
        assert defaults[axis] is None, f"{axis} must defer to --guidance"
    assert defaults["stg_blocks"] == "28"
    assert defaults["guidance"] == "fast"

    guided = ltx_retake_16gb.GUIDANCE_PRESETS["experimental-guided-distilled"]
    assert guided["stg_scale"] == 1.0
    assert guided["modality_scale"] == 3.0
    assert guided["rescale_scale"] == 0.7
    assert guided["cfg_scale"] == 2.5
    assert guided["audio_cfg_scale"] == 1.5


def test_the_shipped_preset_costs_one_pass_per_step():
    """Deferring to WanGP: 8 transformer evaluations per retake instead of 32.

    Each enabled axis is one more transformer pass, and `max_batch_size=1` makes that time rather
    than VRAM -- which is why the reference configuration was four times the wall-clock. It stays
    one flag away; it is no longer what a retake costs by default.
    """
    from aiv_runtime import ltx_retake_16gb

    def passes(preset: dict[str, float]) -> int:
        # cond + uncond, then one more for each extra axis that is off its inert value.
        return (
            2
            + (1 if preset["stg_scale"] > 0.0 else 0)
            + (1 if abs(preset["modality_scale"] - 1.0) > 1e-6 else 0)
        )

    fast = ltx_retake_16gb.GUIDANCE_PRESETS["fast"]
    guided = ltx_retake_16gb.GUIDANCE_PRESETS["experimental-guided-distilled"]
    # `fast` never reaches the guided denoiser at all, so its cost is one pass, not the two the
    # arithmetic above would give a guided run with both extra axes off.
    assert not any(
        (fast["cfg_scale"] > 1.0, fast["audio_cfg_scale"] > 1.0, fast["stg_scale"] > 0.0,
         abs(fast["modality_scale"] - 1.0) > 1e-6)
    )
    assert passes(guided) == 4, "what the reference configuration costs"

    steps = 8  # len(DISTILLED_SIGMAS) - 1, without importing torch to find out
    assert 1 * steps == 8
    assert passes(guided) * steps == 32
