"""The window retake actually loads, around the range being regenerated.

Retake denoises a masked region inside ONE latent, so the clip it is handed has to fit the 16 GB
profile whole. Handing it the source meant a 10.8s clip was refused for a 3s edit, and a
20-minute source could never be retaken at all however short the selection.
"""

from __future__ import annotations

import argparse
import math
from fractions import Fraction
from types import SimpleNamespace

import pytest

from aiv_runtime import ltx_retake_16gb
from aiv_runtime.ltx_geometry import DEFAULT_MAX_FRAMES, FRAMES_PER_LATENT, MIN_FRAMES
from aiv_runtime.ltx_retake_16gb import (
    canonical_frame_rate,
    frame_range,
    max_pixel_frames,
    max_pixels,
    plan_retake_window,
)

BUDGETS = (max_pixels(), max_pixel_frames())


def _source(width=960, height=544, frames=260, fps=24.0):
    return SimpleNamespace(width=width, height=height, frames=frames, fps=fps)


def _plan(source, start, end):
    return plan_retake_window(source, start, end, *BUDGETS)


def test_the_clip_that_started_this_is_now_retakeable():
    """rest42.mp4: 260 frames at 960x544, refused outright for a 3-second edit."""
    window = _plan(_source(), 0.0, 3.0)
    assert DEFAULT_MAX_FRAMES == 193
    assert window["frames"] == 137, "the measured token envelope binds this 960x544 canvas"
    assert window["startFrame"] == 0
    assert (window["rangeIn"], window["rangeOut"]) == (0.0, 3.0)


def test_a_selection_away_from_the_start_is_remapped_into_the_window():
    """The whole point: the worker no longer sees the source's timebase."""
    window = _plan(_source(), 5.0, 8.0)
    assert window["startFrame"] == 88
    assert window["rangeIn"] == pytest.approx(32 / 24)
    assert window["rangeOut"] == pytest.approx(104 / 24)
    # And the selection really is where the window says it is.
    assert window["startFrame"] + window["rangeIn"] * 24.0 == pytest.approx(5.0 * 24.0)


def test_a_twenty_minute_source_can_be_retaken():
    """Impossible before at any selection length; the source was the thing being measured."""
    window = _plan(_source(frames=28_800), 600.0, 603.0)
    assert window["frames"] == 137
    assert window["startFrame"] == 14_368
    assert window["rangeIn"] == pytest.approx(32 / 24)
    assert not window["wholeClip"]


def test_the_window_slides_back_inside_a_selection_at_the_very_end():
    window = _plan(_source(), 9.0, 10.83)
    assert window["startFrame"] + window["frames"] == 260, "flush with the end of the clip"
    assert window["rangeOut"] <= window["seconds"] + 1e-6


def test_a_clip_shorter_than_the_window_uses_what_there_is():
    window = _plan(_source(frames=100), 2.0, 2.5)
    assert window["frames"] == 97, "the largest 8n+1 that fits in 100 frames"
    assert 0 <= window["startFrame"] <= 100 - 97


@pytest.mark.parametrize("frames", [9, 17, 100, 193, 260, 1000, 28_800])
@pytest.mark.parametrize("fps", [24.0, 23.976023976, 29.97002997, 30.0, 25.0])
@pytest.mark.parametrize("position", [0.0, 0.25, 0.5, 0.9])
def test_every_window_is_one_the_worker_would_accept(frames, fps, position):
    """The property that matters: whatever it plans, the profile takes it."""
    source = _source(frames=frames, fps=fps)
    duration = frames / fps
    start = position * duration
    end = min(duration, start + 1.0)
    if end - start < 1.0 / fps:
        pytest.skip("selection shorter than a frame at this clip length")

    window = _plan(source, start, end)

    assert (window["frames"] - 1) % FRAMES_PER_LATENT == 0, "8n+1"
    assert MIN_FRAMES <= window["frames"] <= DEFAULT_MAX_FRAMES
    assert window["frames"] <= frames
    assert 0 <= window["startFrame"]
    assert window["startFrame"] + window["frames"] <= frames
    # The selection has to be inside the window it was planned around.
    assert 0 <= window["rangeIn"] < window["rangeOut"] <= window["seconds"] + 1e-6


def test_a_selection_longer_than_the_profile_is_refused_with_both_numbers():
    with pytest.raises(ValueError, match="the most that can be retaken at once") as failure:
        _plan(_source(), 0.0, 9.0)
    message = str(failure.value)
    assert "216 frames (9.0s)" in message
    assert "137" in message and "5.7s" in message


def test_a_window_cannot_rescue_an_oversized_canvas():
    """A window makes a clip shorter, not smaller, and a downscaled splice shows the seam."""
    with pytest.raises(ValueError, match="downscaled copy"):
        _plan(_source(width=3840, height=2160), 0.0, 3.0)


def test_a_clip_too_short_to_retake_at_all_says_so():
    with pytest.raises(ValueError, match="at least 9"):
        _plan(_source(frames=6), 0.0, 0.2)


# ------------------------------------------------------------------ the extraction itself

import json
import shutil
import subprocess
from pathlib import Path

FFMPEG = shutil.which("ffmpeg")


@pytest.mark.parametrize(
    ("fps", "expected"),
    [(23.976023976, (24_000, 1_001)), (29.97002997, (30_000, 1_001)), (24.0, (24, 1))],
)
def test_frame_rates_are_rational_not_truncated(fps, expected):
    rate = canonical_frame_rate(fps)
    assert (rate.numerator, rate.denominator) == expected


@pytest.mark.parametrize("fps", [23.976023976, 29.97002997])
def test_non_aligned_ranges_cover_every_touched_frame(fps):
    first, last = frame_range(1.001, 2.002, fps, 10_000)
    rate = canonical_frame_rate(fps)
    assert first == math.floor(Fraction("1.001") * rate)
    assert last == math.ceil(Fraction("2.002") * rate)


def _psnr(left: Path, right: Path) -> float:
    """Average PSNR between two stills. Identical content re-encoded lands around 45-50 dB."""
    result = subprocess.run(
        [FFMPEG or "ffmpeg", "-hide_banner", "-i", str(left), "-i", str(right),
         "-filter_complex", "psnr", "-f", "null", "-"],
        capture_output=True, text=True, check=True,
    )
    for token in reversed(result.stderr.replace("\n", " ").split()):
        if token.startswith("average:"):
            value = token.split(":", 1)[1]
            return float("inf") if value == "inf" else float(value)
    raise AssertionError(f"no PSNR in ffmpeg output: {result.stderr[-400:]}")


def _frame(clip: Path, index: int, destination: Path, fps: float = 24.0) -> Path:
    subprocess.run(
        [FFMPEG or "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-i", str(clip), "-ss", f"{index / fps:.6f}", "-frames:v", "1", str(destination)],
        check=True,
    )
    return destination


@pytest.mark.skipif(FFMPEG is None, reason="FFmpeg is unavailable")
def test_the_extracted_window_holds_the_frames_it_planned(tmp_path):
    """The crux of windowing: the selection has to end up where the plan says it is.

    If the extraction is off by a frame, every index computed downstream is wrong by an amount
    nothing can detect -- the retake would simply regenerate slightly the wrong moment.
    """
    from aiv_runtime.ltx_retake_16gb import extract_retake_window

    source = tmp_path / "source.mp4"
    subprocess.run(
        [FFMPEG or "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "testsrc2=size=960x544:rate=24", "-frames:v", "260",
         "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p", str(source)],
        check=True,
    )

    window = _plan(_source(), 5.0, 8.0)
    assert (window["startFrame"], window["frames"]) == (88, 137)
    extracted = Path(extract_retake_window(str(source), window, tmp_path / "window.mp4"))

    probed = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames", "-of", "json", str(extracted)],
        capture_output=True, text=True, check=True,
    )
    assert int(json.loads(probed.stdout)["streams"][0]["nb_read_frames"]) == 137

    # Adjacent frames of this source are genuinely different pictures, so a match means alignment
    # and not a static clip.
    baseline = _psnr(_frame(source, 60, tmp_path / "a.png"), _frame(source, 61, tmp_path / "b.png"))
    assert baseline < 25, f"the fixture is too static to prove anything (PSNR {baseline})"

    for source_frame, window_frame in ((88, 0), (120, 32), (224, 136)):
        aligned = _psnr(
            _frame(source, source_frame, tmp_path / "s.png"),
            _frame(extracted, window_frame, tmp_path / "w.png"),
        )
        assert aligned > 35, f"window frame {window_frame} is not source frame {source_frame} (PSNR {aligned})"

    # And one frame out is obviously wrong, so the check above is not passing on noise.
    misaligned = _psnr(
        _frame(source, 89, tmp_path / "s.png"), _frame(extracted, 0, tmp_path / "w.png")
    )
    assert misaligned < 25


@pytest.mark.skipif(FFMPEG is None, reason="FFmpeg is unavailable")
def test_the_window_keeps_the_canvas_exactly(tmp_path):
    """No scaling: a resampled window spliced beside untouched footage shows the seam."""
    from aiv_runtime.ltx_retake_16gb import extract_retake_window

    source = tmp_path / "source.mp4"
    subprocess.run(
        [FFMPEG or "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "testsrc2=size=960x544:rate=24", "-frames:v", "260",
         "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(source)],
        check=True,
    )
    extracted = extract_retake_window(str(source), _plan(_source(), 5.0, 8.0), tmp_path / "w.mp4")
    probed = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "json", str(extracted)],
        capture_output=True, text=True, check=True,
    )
    stream = json.loads(probed.stdout)["streams"][0]
    assert (stream["width"], stream["height"]) == (960, 544)


def _make_av_fixture(path: Path, fps: str, frames: int, *, audio: bool = True, color: str | None = None):
    picture = f"color={color}:size=160x96:rate={fps}" if color else f"testsrc2=size=160x96:rate={fps}"
    command = [FFMPEG or "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
               "-f", "lavfi", "-i", picture]
    if audio:
        command += ["-f", "lavfi", "-i", "sine=frequency=880:sample_rate=48000"]
    command += ["-frames:v", str(frames), "-c:v", "libx264", "-crf", "0", "-pix_fmt", "yuv420p"]
    if audio:
        numerator, denominator = (int(piece) for piece in fps.split("/"))
        command += ["-c:a", "alac", "-t", f"{frames * denominator / numerator:.9f}"]
    command += [str(path)]
    subprocess.run(command, check=True)


def _stream_info(path: Path) -> dict:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-show_entries",
         "stream=index,codec_type,codec_name,r_frame_rate,nb_read_frames", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    return json.loads(result.stdout)


def _frame_hashes(path: Path, select: str | None = None) -> list[str]:
    command = [FFMPEG or "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path)]
    if select:
        command += ["-vf", select, "-vsync", "0"]
    command += ["-map", "0:v:0", "-f", "framemd5", "-"]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    return [line.rsplit(",", 1)[-1].strip() for line in result.stdout.splitlines() if line and not line.startswith("#")]


def _pcm(path: Path, start: float | None = None, duration: float | None = None) -> bytes:
    command = [FFMPEG or "ffmpeg", "-hide_banner", "-loglevel", "error"]
    if start is not None:
        command += ["-ss", f"{start:.9f}"]
    command += ["-i", str(path)]
    if duration is not None:
        command += ["-t", f"{duration:.9f}"]
    command += ["-map", "0:a:0", "-c:a", "pcm_s16le", "-f", "s16le", "-"]
    return subprocess.run(command, capture_output=True, check=True).stdout


@pytest.mark.skipif(FFMPEG is None, reason="FFmpeg is unavailable")
@pytest.mark.parametrize("mode", ["video-audio", "video", "audio"])
def test_every_regeneration_mode_obeys_the_frame_contract(tmp_path, mode):
    from aiv_runtime.ltx_retake_16gb import finalize_retake_output

    fps = "24000/1001"
    source = tmp_path / "source.mp4"
    generated = tmp_path / f"generated-{mode}.mp4"
    output = tmp_path / f"output-{mode}.mp4"
    _make_av_fixture(source, fps, 72)
    metadata = _source(width=160, height=96, frames=72, fps=24000 / 1001)
    window = _plan(metadata, 0.501, 1.502)
    _make_av_fixture(generated, fps, window["selectionFrames"], color="red")

    finalize_retake_output(mode, str(source), generated, output, window)
    streams = _stream_info(output)["streams"]
    video = next(stream for stream in streams if stream["codec_type"] == "video")
    assert int(video["nb_read_frames"]) == window["selectionFrames"]
    assert video["r_frame_rate"] == fps
    assert any(stream["codec_type"] == "audio" for stream in streams)

    if mode == "audio":
        expected = _frame_hashes(
            source,
            f"select='between(n,{window['selectionStartFrame']},{window['selectionEndFrame'] - 1})'",
        )
        assert _frame_hashes(output) == expected, "audio-only must retain the decoded source frames exactly"
    if mode == "video":
        duration = window["effectiveEndTime"] - window["effectiveStartTime"]
        expected = _pcm(source, window["effectiveStartTime"], duration)
        actual = _pcm(output)
        assert actual == expected, "video-only must retain source PCM samples exactly"


@pytest.mark.skipif(FFMPEG is None, reason="FFmpeg is unavailable")
@pytest.mark.parametrize(("mode", "expects_audio"), [("video", False), ("audio", True)])
def test_single_modality_modes_accept_a_silent_source(tmp_path, mode, expects_audio):
    from aiv_runtime.ltx_retake_16gb import finalize_retake_output

    fps = "30000/1001"
    source = tmp_path / "silent.mp4"
    generated = tmp_path / "generated.mp4"
    output = tmp_path / "output.mp4"
    _make_av_fixture(source, fps, 90, audio=False)
    metadata = _source(width=160, height=96, frames=90, fps=30000 / 1001)
    window = _plan(metadata, 0.337, 1.341)
    _make_av_fixture(generated, fps, window["selectionFrames"], color="blue")
    finalize_retake_output(mode, str(source), generated, output, window)
    streams = _stream_info(output)["streams"]
    assert any(stream["codec_type"] == "audio" for stream in streams) is expects_audio


@pytest.mark.skipif(FFMPEG is None, reason="FFmpeg is unavailable")
def test_context_preview_contains_before_range_and_after(tmp_path):
    from aiv_runtime.ltx_retake_16gb import build_context_preview

    fps = "24000/1001"
    source = tmp_path / "source.mp4"
    retake = tmp_path / "retake.mp4"
    preview = tmp_path / "preview.mp4"
    _make_av_fixture(source, fps, 96)
    metadata = _source(width=160, height=96, frames=96, fps=24000 / 1001)
    window = _plan(metadata, 1.001, 2.002)
    _make_av_fixture(retake, fps, window["selectionFrames"], color="yellow")
    build_context_preview(str(source), str(retake), preview, window, context_seconds=0.5)
    video = next(stream for stream in _stream_info(preview)["streams"] if stream["codec_type"] == "video")
    assert int(video["nb_read_frames"]) > window["selectionFrames"]


@pytest.mark.skipif(FFMPEG is None, reason="FFmpeg is unavailable")
@pytest.mark.parametrize(("start", "end"), [(0.0, 1.001), (2.711, 3.712)])
def test_context_preview_handles_source_boundaries(tmp_path, start, end):
    from aiv_runtime.ltx_retake_16gb import build_context_preview

    fps = "24000/1001"
    source = tmp_path / "source.mp4"
    retake = tmp_path / "retake.mp4"
    preview = tmp_path / "preview.mp4"
    _make_av_fixture(source, fps, 89)
    metadata = _source(width=160, height=96, frames=89, fps=24000 / 1001)
    window = _plan(metadata, start, end)
    _make_av_fixture(retake, fps, window["selectionFrames"], color="yellow")
    build_context_preview(str(source), str(retake), preview, window, context_seconds=0.5)
    count = int(next(stream for stream in _stream_info(preview)["streams"] if stream["codec_type"] == "video")["nb_read_frames"])
    assert count > window["selectionFrames"], "the available side still contributes context"


@pytest.mark.skipif(FFMPEG is None, reason="FFmpeg is unavailable")
def test_whole_source_context_preview_is_just_the_retake(tmp_path):
    from aiv_runtime.ltx_retake_16gb import build_context_preview

    fps = "24000/1001"
    source = tmp_path / "source.mp4"
    retake = tmp_path / "retake.mp4"
    preview = tmp_path / "preview.mp4"
    _make_av_fixture(source, fps, 89)
    metadata = _source(width=160, height=96, frames=89, fps=24000 / 1001)
    window = _plan(metadata, 0.0, 89 * 1001 / 24000)
    _make_av_fixture(retake, fps, window["selectionFrames"], color="yellow")
    build_context_preview(str(source), str(retake), preview, window)
    count = int(next(stream for stream in _stream_info(preview)["streams"] if stream["codec_type"] == "video")["nb_read_frames"])
    assert count == window["selectionFrames"]


# ---------------------------------------------------------------------------
# Guidance presets
# ---------------------------------------------------------------------------


def _guidance_args(**overrides) -> argparse.Namespace:
    base = dict(
        guidance="fast",
        cfg_scale=None,
        audio_cfg_scale=None,
        stg_scale=None,
        modality_scale=None,
        rescale_scale=None,
        negative_prompt="",
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_fast_leaves_every_axis_inert_so_the_step_is_one_pass():
    """The whole point of the preset: 8 transformer evaluations per retake instead of 32."""
    args = _guidance_args(guidance="fast")
    ltx_retake_16gb.resolve_guidance(args)
    assert args.cfg_scale == 1.0
    assert args.audio_cfg_scale == 1.0
    assert args.stg_scale == 0.0
    assert args.modality_scale == 1.0
    assert not ltx_retake_16gb.guidance_is_live(args)


def test_experimental_guided_distilled_costs_four_passes():
    args = _guidance_args(guidance="experimental-guided-distilled")
    ltx_retake_16gb.resolve_guidance(args)
    assert (args.stg_scale, args.modality_scale, args.rescale_scale) == (1.0, 3.0, 0.7)
    assert ltx_retake_16gb.guidance_is_live(args)
    passes = 1 + 1 + (args.stg_scale > 0.0) + (abs(args.modality_scale - 1.0) > 1e-6)
    assert passes == 4


def test_an_explicit_flag_beats_the_preset():
    """Otherwise `--guidance fast --stg-scale 1.0` would silently mean `fast`."""
    args = _guidance_args(guidance="fast", stg_scale=1.0)
    ltx_retake_16gb.resolve_guidance(args)
    assert args.stg_scale == 1.0
    # Everything it did not name still comes from the preset.
    assert args.cfg_scale == 1.0
    assert ltx_retake_16gb.guidance_is_live(args)


def test_resolving_twice_changes_nothing():
    """`validate` resolves, and callers may resolve first; neither may clobber the other."""
    args = _guidance_args(guidance="experimental-guided-distilled", cfg_scale=6.0)
    ltx_retake_16gb.resolve_guidance(args)
    ltx_retake_16gb.resolve_guidance(args)
    assert args.cfg_scale == 6.0
    assert args.stg_scale == 1.0


def test_every_preset_sets_every_axis():
    """A preset missing a key would leave a None to be compared with `>` at run time."""
    keys = {"cfg_scale", "audio_cfg_scale", "stg_scale", "modality_scale", "rescale_scale"}
    for name, preset in ltx_retake_16gb.GUIDANCE_PRESETS.items():
        assert set(preset) == keys, name


def test_a_negative_prompt_under_fast_is_refused_not_dropped():
    """SimpleDenoiser never encodes it, so accepting it would waste a whole run."""
    args = _guidance_args(guidance="fast", negative_prompt="blurry, watermark")
    ltx_retake_16gb.resolve_guidance(args)
    with pytest.raises(ValueError, match="silently"):
        ltx_retake_16gb._refuse_a_negative_prompt_that_cannot_be_used(args)


def test_a_negative_prompt_under_experimental_guidance_is_accepted():
    args = _guidance_args(guidance="experimental-guided-distilled", negative_prompt="blurry, watermark")
    ltx_retake_16gb.resolve_guidance(args)
    ltx_retake_16gb._refuse_a_negative_prompt_that_cannot_be_used(args)


def test_the_default_preset_is_the_fast_one():
    assert ltx_retake_16gb.DEFAULT_GUIDANCE == "fast"
