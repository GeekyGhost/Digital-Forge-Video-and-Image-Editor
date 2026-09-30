from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from fractions import Fraction
from pathlib import Path

# VRAM guard for the 16 GB profile, in pixels. 768*768 is what the validated 768x512/121-frame
# profile was extrapolated to, NOT a model limit -- the real cost is tokens, i.e. pixels TIMES
# latent frames, so a short window at a higher resolution can be cheaper than a long one at a low
# resolution. Raise it via AIV_RETAKE_MAX_PIXELS to probe where the real ceiling is; the adapter
# reports peak VRAM at the end so the answer is measured rather than guessed.
#
# The number itself now lives in `ltx_geometry` alongside the axis grid and the frame cap, because
# this file, the generation adapter and the foley adapter had four different answers between them.

# Lightricks' hosted retake endpoint refuses a window shorter than this. The OSS pipeline does NOT
# enforce it, which is exactly why a short selection appears to "run fine and change nothing": the
# region is legal, it is noised, and the surrounding pinned tokens still dominate the attention so
# the model reconstructs what was already there. Advisory, not fatal -- a short window is allowed,
# it just usually will not visibly change.
ADVISED_MIN_WINDOW_SECONDS = 2.0


# The LTX adapters run under the vendored LTX virtualenv, where this package is not installed, and
# Python only puts the *script's own* directory on sys.path -- not the package root. Without this,
# `import aiv_runtime.…` fails with ModuleNotFoundError depending on which interpreter is used.
_PACKAGE_ROOT = str(Path(__file__).resolve().parent.parent)
if _PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, _PACKAGE_ROOT)

from aiv_runtime.run_timing import StageTimer  # noqa: E402
from aiv_runtime.ltx_attention import (  # noqa: E402
    ATTENTION_BACKENDS,
    DEFAULT_ATTENTION_BACKEND,
    describe_attention,
    install_stage_attention,
)
from aiv_runtime.ltx_offload import (  # noqa: E402
    OFFLOAD_PROFILES,
    cpu_slots_for,
    describe_profile,
    offload_mode_for,
)
from aiv_runtime.ltx_geometry import (  # noqa: E402
    AXIS_MULTIPLE,
    MIN_FRAMES,
    DEFAULT_MAX_FRAMES,
    DEFAULT_MAX_PIXEL_FRAMES,
    DEFAULT_MAX_PIXELS,
    FRAMES_PER_LATENT,
    env_max_pixels,
    pixel_frames,
    snap_axis,
    snap_frames,
)

#: The override, renamed for what it now bounds. `AIV_RETAKE_MAX_PIXELS` stays as an alias -- it is
#: documented and it is in people's notes -- and the new name wins when both are set.
#: What the last validated request cost, so the peak-VRAM report can quote it. Module-level
#: because the report happens after the pipeline has been torn down and the source is long gone.
_LAST_PIXEL_FRAMES = 0
_PIPELINE_CACHE: dict[tuple, object] = {}
_DIAGNOSTICS_INSTALLED = False

PIXEL_FRAME_BUDGET_ENV = "AIV_LTX_MAX_PIXEL_FRAMES"
LEGACY_PIXEL_BUDGET_ENV = "AIV_RETAKE_MAX_PIXELS"


def max_pixels() -> int:
    """The per-frame pixel budget in force, honouring the documented override.

    Kept as a function rather than a constant because the override is read per run; an unparseable
    value now raises instead of silently reverting to the default, which is how "I raised the
    budget and nothing changed" happens.
    """
    return env_max_pixels(LEGACY_PIXEL_BUDGET_ENV)


def max_pixel_frames() -> int:
    """The token budget in force: pixels times latent frames."""
    return env_max_pixels(PIXEL_FRAME_BUDGET_ENV, default=DEFAULT_MAX_PIXEL_FRAMES)


def canonical_frame_rate(fps: float) -> Fraction:
    """Recover the source's rational frame rate from the metadata float."""
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"invalid source frame rate {fps!r}")
    return Fraction(str(fps)).limit_denominator(1001)


def frame_range(start_time: float, end_time: float, fps: float, total_frames: int) -> tuple[int, int]:
    """Resolve a half-open time request to the covering half-open source-frame range."""
    rate = canonical_frame_rate(fps)
    first = max(0, min(total_frames, math.floor(Fraction(str(start_time)) * rate)))
    last = max(first, min(total_frames, math.ceil(Fraction(str(end_time)) * rate)))
    if last <= first and first < total_frames:
        last = first + 1
    return first, last


def plan_retake_window(
    source, start_time: float, end_time: float, pixel_budget: int, token_budget: int
) -> dict:
    """Choose the stretch of the clip to actually load, around the range being regenerated.

    Retake denoises a masked region inside ONE latent, so the whole clip it is given has to fit the
    profile. Handing it the whole source meant a 10.8s clip was refused outright for a 3s edit --
    and a 20-minute source could never be retaken at all, however short the selection.

    So the clip it is given is not the source any more: it is a window around the selection, as
    large as the profile allows. The extra frames either side are not waste; the mask only covers
    the selection and the surrounding pinned latents are what the model reconstructs against, so a
    centred window is also the best conditioning available.

    Spatial limits are NOT solved this way, deliberately. A window can be shorter than the source
    but it cannot be smaller without downscaling, and a downscaled retake spliced back beside
    untouched footage shows the seam. An oversized clip still needs a proxy.

    Returns the window in source frames and the selection remapped into the window's own timebase.
    """
    rate = canonical_frame_rate(float(source.fps))
    fps = float(rate)
    total = int(source.frames)
    pixels = source.width * source.height

    if total < MIN_FRAMES:
        raise ValueError(f"the clip is {total} frames; retake needs at least {MIN_FRAMES}")
    if pixels > pixel_budget:
        raise ValueError(
            f"{source.width}x{source.height} is {pixels:,} pixels, over the {pixel_budget:,} "
            f"budget -- {pixels / pixel_budget:.1f}x too large. A window makes a clip shorter, not "
            "smaller: retake still needs a downscaled copy, 768x768 or less"
        )
    off_grid = [
        f"{name} {value}" for name, value in (("width", source.width), ("height", source.height))
        if value % AXIS_MULTIPLE
    ]
    if off_grid:
        raise ValueError(
            f"{' and '.join(off_grid)} is off the {AXIS_MULTIPLE}-pixel grid the VAE decodes on "
            f"(the nearest are {snap_axis(source.width)}x{snap_axis(source.height)}). A window "
            "cannot fix this either -- the canvas is the canvas"
        )

    # The longest window this canvas can afford, on every axis at once.
    affordable = min(DEFAULT_MAX_FRAMES, _affordable_frames(source.width, source.height, token_budget), total)
    window_frames = snap_frames(affordable, "down")
    if window_frames > total:
        window_frames = snap_frames(total, "down")

    first, last = frame_range(start_time, end_time, fps, total)
    selection_frames = last - first

    if selection_frames > window_frames:
        raise ValueError(
            f"the selected range is {selection_frames} frames ({selection_frames / fps:.1f}s), and "
            f"at {source.width}x{source.height} the most that can be retaken at once is "
            f"{window_frames} ({window_frames / fps:.1f}s). Shorten the selection, or use a "
            "smaller canvas -- cost is pixels times latent frames, so one buys the other"
        )

    # Centre the window on the selection, then slide it back inside the clip.
    start_frame = first - (window_frames - selection_frames) // 2
    start_frame = max(0, min(start_frame, total - window_frames))

    return {
        "startFrame": start_frame,
        "frames": window_frames,
        "startSeconds": start_frame / fps,
        "seconds": window_frames / fps,
        "rangeIn": (first - start_frame) / fps,
        "rangeOut": (last - start_frame) / fps,
        "selectionFrames": selection_frames,
        "selectionStartFrame": first,
        "selectionEndFrame": last,
        "effectiveStartTime": float(Fraction(first, 1) / rate),
        "effectiveEndTime": float(Fraction(last, 1) / rate),
        "sourceFrames": total,
        "fps": fps,
        "fpsNumerator": rate.numerator,
        "fpsDenominator": rate.denominator,
        "wholeClip": start_frame == 0 and window_frames == total,
    }


def extract_retake_window(video_path: str, window: dict, destination: Path) -> str:
    """Cut the planned window out of the source, exactly, as a constant-rate clip.

    Output seek (`-ss` after `-i`) rather than input seek: it decodes from the start, which costs
    time on a long source, but it lands on the frame asked for. A keyframe-accurate seek would put
    the selection somewhere near where the user pointed, and every frame index computed from here
    on would be wrong by an amount nothing downstream could detect.

    No scaling. The canvas is already known to fit, and rescaling here would mean splicing a
    resampled window back beside untouched footage.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-i", video_path,
            "-ss", f"{window['startSeconds']:.6f}",
            "-frames:v", str(window["frames"]),
            "-vf", f"fps={window['fpsNumerator']}/{window['fpsDenominator']},setsar=1",
            "-c:v", "libx264", "-preset", "medium", "-crf", "16", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k",
            str(destination),
        ],
        check=True,
    )
    return str(destination)


def finalize_retake_output(
    mode: str, source_path: str, generated_path: str | Path, output_path: str | Path, window: dict
) -> str:
    """Mux generated media with the untouched, source-derived modality.

    A frozen LTX latent is still VAE-decoded. For audio-only, source frames are decoded and stored
    losslessly with x264; for video-only, source audio samples are decoded and stored losslessly as
    ALAC. At an arbitrary frame boundary packet-copy is not generally possible, so exact means
    identical decoded frames/samples rather than identical compressed packets.
    """
    generated_path = Path(generated_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.unlink(missing_ok=True)
    if mode == "video-audio":
        os.replace(generated_path, output_path)
        return str(output_path)

    start = window["effectiveStartTime"]
    duration = window["effectiveEndTime"] - start
    frames = window["selectionFrames"]
    fps = f"{window['fpsNumerator']}/{window['fpsDenominator']}"
    common = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y"]
    if mode == "audio":
        command = common + [
            "-i", source_path, "-i", str(generated_path),
            "-map", "0:v:0", "-map", "1:a:0?", "-frames:v", str(frames),
            "-vf", (
                f"select='between(n\\,{window['selectionStartFrame']}\\,"
                f"{window['selectionEndFrame'] - 1})',"
                f"setpts=N*{window['fpsDenominator']}/({window['fpsNumerator']}*TB),setsar=1"
            ),
            "-c:v", "libx264", "-preset", "medium",
            "-crf", "0", "-c:a", "aac", "-b:a", "192k",
            "-t", f"{duration:.9f}", str(output_path),
        ]
    else:
        command = common + [
            "-i", str(generated_path), "-ss", f"{start:.9f}", "-i", source_path,
            "-map", "0:v:0", "-map", "1:a:0?", "-frames:v", str(frames),
            "-c:v", "copy", "-c:a", "alac", "-t", f"{duration:.9f}", str(output_path),
        ]
    subprocess.run(command, check=True)
    return str(output_path)


def build_context_preview(
    source_path: str, retake_path: str, output_path: str | Path, window: dict, context_seconds: float = 1.0
) -> str:
    """Build a visual before/retake/after seam preview on the same rational frame grid."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    first, last = window["selectionStartFrame"], window["selectionEndFrame"]
    context_frames = max(1, round(context_seconds * window["fps"]))
    setpts = f"setpts=N*{window['fpsDenominator']}/({window['fpsNumerator']}*TB),setsar=1"
    encode = ["-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p"]
    common = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y"]
    with tempfile.TemporaryDirectory(prefix="aiv-seam-") as temporary:
        temp = Path(temporary)
        clips: list[Path] = []

        def source_segment(name: str, low: int, high: int) -> None:
            clip = temp / f"{name}.mp4"
            subprocess.run(
                common + ["-i", source_path, "-vf",
                          f"select='between(n\\,{low}\\,{high - 1})',{setpts}",
                          "-frames:v", str(high - low)] + encode + [str(clip)],
                check=True,
            )
            clips.append(clip)

        if first > 0:
            source_segment("before", max(0, first - context_frames), first)
        middle = temp / "retake.mp4"
        subprocess.run(
            common + ["-i", retake_path, "-vf", setpts, "-frames:v", str(window["selectionFrames"])]
            + encode + [str(middle)],
            check=True,
        )
        clips.append(middle)
        if last < window["sourceFrames"]:
            source_segment("after", last, min(window["sourceFrames"], last + context_frames))

        manifest = temp / "segments.ffconcat"
        manifest.write_text(
            "ffconcat version 1.0\n" + "".join(
                "file '" + clip.as_posix().replace("'", "'\\''") + "'\n" for clip in clips
            ),
            encoding="utf-8",
        )
        subprocess.run(
            common + ["-f", "concat", "-safe", "0", "-i", str(manifest), "-c", "copy", str(output_path)],
            check=True,
        )
    return str(output_path)


def slice_decoded_frames(video, first: int, last: int):
    """Keep only frames `[first, last)` of the decoder's output.

    The video decoder returns a lazy iterator of chunks, so this is not only a trim: the chunks
    past the selection are never decoded at all. A tensor is handled too, because the annotation
    permits one.
    """
    if hasattr(video, "shape") and not hasattr(video, "__next__"):
        return video[first:last]

    def wanted():
        index = 0
        for chunk in video:
            size = int(chunk.shape[0])
            low, high = max(first, index), min(last, index + size)
            if high > low:
                yield chunk[low - index : high - index]
            index += size
            if index >= last:
                break

    return wanted()


def slice_decoded_audio(audio, start_seconds: float, end_seconds: float):
    """The same cut through the waveform, in samples rather than frames."""
    from dataclasses import replace

    waveform = getattr(audio, "waveform", None)
    rate = int(getattr(audio, "sampling_rate", 0) or 0)
    if audio is None or waveform is None or rate <= 0:
        return audio

    first = max(0, int(round(start_seconds * rate)))
    last = max(first, int(round(end_seconds * rate)))
    # `encode_video` accepts stereo as (2, N) or (N, 2); slice whichever axis is time.
    channels_first = len(waveform.shape) == 2 and waveform.shape[0] <= 2 <= waveform.shape[1]
    sliced = waveform[:, first:last] if channels_first else waveform[first:last]
    return replace(audio, waveform=sliced)


def _affordable_frames(width: int, height: int, token_budget: int) -> int:
    """The longest 8n+1 count this canvas can afford, so the message can name one."""
    pixels = max(1, width * height)
    latents = max(1, token_budget // pixels)
    return min(DEFAULT_MAX_FRAMES, snap_frames((latents - 1) * FRAMES_PER_LATENT + 1, "down"))


def parse_blocks(raw: str) -> list[int]:
    if not raw.strip():
        return []
    return [int(piece) for piece in raw.replace(" ", "").split(",") if piece]


#: What each guidance axis is set to, per preset. `None` nowhere -- every key is resolved.
#:
#: `fast` is the WanGP-equivalent configuration and the default: every axis inert, so the
#: pipeline takes `SimpleDenoiser` and runs **one transformer pass per step**. Against the
#: experimental guided-distilled preset
#: that is 8 evaluations per retake instead of 32.
#:
#: `experimental-guided-distilled` deliberately runs the distilled checkpoint through the guided
#: denoiser while retaining its eight-step distilled sigma schedule. LTX documents distilled=True
#: for this checkpoint; this hybrid is useful to test guidance response, but is not a reference or
#: officially qualified LTX 2.3 mode. It costs four transformer passes per step.
#:
#: The experimental preset has not been qualified against `fast` on real footage, and four times
#: the wall-clock is a large price for an unmeasured difference. `fast` is what WanGP runs on the
#: same weights. The `[latent]` diagnostic measures whether the requested region actually moved.
GUIDANCE_PRESETS: dict[str, dict[str, float]] = {
    "fast": {
        "cfg_scale": 1.0,
        "audio_cfg_scale": 1.0,
        "stg_scale": 0.0,
        "modality_scale": 1.0,
        "rescale_scale": 0.0,
    },
    "experimental-guided-distilled": {
        "cfg_scale": 2.5,
        "audio_cfg_scale": 1.5,
        "stg_scale": 1.0,
        "modality_scale": 3.0,
        "rescale_scale": 0.7,
    },
}

DEFAULT_GUIDANCE = "fast"


def resolve_guidance(args: argparse.Namespace) -> None:
    """Fill every unset guidance scale from the named preset, in place.

    An explicit flag always wins, which is what makes `--guidance fast --stg-scale 1.0` mean
    what it looks like it means rather than being silently overwritten by the preset.
    """
    preset = GUIDANCE_PRESETS[args.guidance]
    for name, value in preset.items():
        if getattr(args, name) is None:
            setattr(args, name, value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LTX 2.3 retake with scaled-FP8 block streaming for 16 GB GPUs.")
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--gemma-root", required=True)
    parser.add_argument("--video-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--start-time", type=float, required=True)
    parser.add_argument("--end-time", type=float, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--negative-prompt", default="")
    # Classifier-free guidance. WITHOUT this the pipeline runs `SimpleDenoiser`, which carries the
    # prompt as context but nothing pushes the result toward it -- so a masked region, surrounded
    # by pinned original latents, simply reconstructs what was already there. 1.0 = unguided.
    #
    # CAVEAT, and it is the whole reason `--diagnostics` exists: this checkpoint is a GUIDANCE-
    # DISTILLED model. Distillation trains the conditioned prediction to already contain the
    # guidance, which drives (cond - uncond) toward zero -- and CFG's entire contribution is
    # (scale - 1) * (cond - uncond). If that difference is ~0, raising cfg-scale from 2 to 5
    # multiplies zero and changes nothing, which is precisely the reported symptom. The
    # `[ltx-retake][cfg]` lines below measure that difference instead of assuming it.
    #
    # Every scale below defaults to None and is resolved from `--guidance`, so that an explicit
    # flag beats the preset and the preset is the only place the numbers live.
    parser.add_argument("--guidance", choices=tuple(GUIDANCE_PRESETS), default=DEFAULT_GUIDANCE)
    parser.add_argument("--cfg-scale", type=float, default=None)
    parser.add_argument("--audio-cfg-scale", type=float, default=None)
    # The other three guidance axes LTX-2.3's own reference config uses. Unlike CFG these do NOT
    # depend on a cond/uncond gap surviving distillation, so they are the levers that can still
    # move a distilled model:
    #   stg      -- perturbs self-attention in `stg-blocks` and pushes away from the degraded
    #               prediction. Reference LTX-2.3 value 1.0 on block 28.
    #   modality -- pushes away from a prediction with the audio<->video cross-attention severed.
    #               Reference value 3.0.
    #   rescale  -- renormalises the guided prediction back to the conditioned std. Reference 0.7.
    #
    # Each enabled axis adds one transformer pass per step, and `max_batch_size=1` splits those
    # passes sequentially, so the cost is TIME, not VRAM. See `GUIDANCE_PRESETS` for which
    # combination ships and why.
    parser.add_argument("--stg-scale", type=float, default=None)
    parser.add_argument("--stg-blocks", default="28")
    parser.add_argument("--modality-scale", type=float, default=None)
    parser.add_argument("--rescale-scale", type=float, default=None)
    parser.add_argument("--mode", choices=("video-audio", "video", "audio"), default="video-audio")
    parser.add_argument("--enhance-prompt", action="store_true")
    # CPU streaming, not disk. Disk streaming crashes this machine natively: six `0xC0000005`
    # faults on 2026-08-11/12, every one inside `torch_cpu.dll` at the SAME offset 0x8e8b949, and
    # every one on the disk path. The identical clip completes cleanly under `cpu`. The suspect is
    # the pinned-buffer recycling around `sft_loader`'s `non_blocking=True` copies, which only the
    # disk path exercises. `disk` is kept as an option for machines short of RAM.
    parser.add_argument("--offload", choices=OFFLOAD_PROFILES, default="cpu")
    # Sage is not the default here, unlike H3: on H3 the kernels were installed and
    # merely unselected, whereas nothing is installed in the LTX venv until
    # scripts\install-ltx-attention.ps1 has been run. Asking for a backend that is not
    # there costs a log line, not a run.
    parser.add_argument(
        "--attention-backend", choices=ATTENTION_BACKENDS, default=DEFAULT_ATTENTION_BACKEND
    )
    parser.add_argument("--diagnostics", dest="diagnostics", action="store_true", default=True)
    parser.add_argument("--no-diagnostics", dest="diagnostics", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--server", action="store_true", help="Read JSON requests from stdin and keep the LTX 2.3 runtime warm.")
    return parser


def validate(args: argparse.Namespace) -> None:
    # Before any range check: the scales arrive as None and mean "whatever the preset says".
    resolve_guidance(args)
    for value in (args.checkpoint_path, args.gemma_root, args.video_path):
        if not Path(value).exists():
            raise FileNotFoundError(value)
    if not args.prompt.strip():
        raise ValueError("prompt cannot be empty")
    if args.start_time < 0 or args.end_time <= args.start_time:
        raise ValueError("retake requires 0 <= start-time < end-time")
    for name, value in (("cfg-scale", args.cfg_scale), ("audio-cfg-scale", args.audio_cfg_scale)):
        if not 1.0 <= value <= 10.0:
            raise ValueError(f"{name} must be between 1.0 (unguided) and 10.0")
    if not 0.0 <= args.stg_scale <= 5.0:
        raise ValueError("stg-scale must be between 0.0 (off) and 5.0")
    if not 1.0 <= args.modality_scale <= 10.0:
        raise ValueError("modality-scale must be between 1.0 (off) and 10.0")
    if not 0.0 <= args.rescale_scale <= 1.0:
        raise ValueError("rescale-scale must be between 0.0 (off) and 1.0")
    parse_blocks(args.stg_blocks)
    _refuse_a_negative_prompt_that_cannot_be_used(args)
    if Path(args.output_path).suffix.lower() != ".mp4":
        raise ValueError("retake output must be an MP4 file")


def _refuse_a_negative_prompt_that_cannot_be_used(args: argparse.Namespace) -> None:
    """A negative prompt needs an unconditional branch to be subtracted from. `fast` has none.

    `guided` False selects `SimpleDenoiser`, which never encodes the negative prompt -- so the
    text is accepted, the run takes its several minutes, and the instruction has no effect and no
    mention anywhere. The single-pass adapter already refuses this; making `fast` the default
    retake preset opens the same hole here, so it gets the same refusal rather than the same bug.
    """
    if not args.negative_prompt.strip():
        return
    if guidance_is_live(args):
        return
    raise ValueError(
        "a negative prompt cannot be used under --guidance fast: with every guidance axis inert "
        "the pipeline runs SimpleDenoiser, which never encodes it, so it would be silently "
        "ignored. Use --guidance experimental-guided-distilled (4 transformer passes per step "
        "instead of 1), or clear the negative prompt."
    )


def guidance_is_live(args: argparse.Namespace) -> bool:
    """Whether any guidance axis is off its inert value, i.e. whether there is an uncond branch."""
    return (
        args.cfg_scale > 1.0
        or args.audio_cfg_scale > 1.0
        or args.stg_scale > 0.0
        or abs(args.modality_scale - 1.0) > 1e-6
    )


def install_diagnostics() -> None:
    """Instrument the three places a retake can silently become a no-op.

    Every one of these is a measurement, not a guess, and each isolates a different failure:

      [mask] -- the realized `denoise_mask` after `TemporalRegionMask` resolves seconds into
                patchified token bounds. Zero here means nothing is denoisable and the output can
                only ever equal the input, whatever the prompt or the guidance scale says.
      [cfg]  -- the mean |cond - uncond| the guider is handed. CFG contributes exactly
                (scale - 1) * (cond - uncond); on a guidance-distilled checkpoint that gap collapses
                and no scale can revive it. A ratio near zero here IS the answer to "strength 5 does
                nothing".
      [step] -- how far the denoised latent has actually moved from the pinned original, measured
                INSIDE the mask only. Non-zero mask plus a near-zero step delta means the model is
                reconstructing the surrounding context rather than the prompt: a window too short,
                or guidance too weak, not a plumbing bug.
    """
    import torch
    from ltx_core.components import guiders as guiders_module
    from ltx_core.conditioning.types import noise_mask_cond
    from ltx_pipelines.utils import samplers

    original_apply = noise_mask_cond.TemporalRegionMask.apply_to

    def logged_apply(self, latent_state, latent_tools):  # noqa: ANN001
        state = original_apply(self, latent_state, latent_tools)
        mask = state.denoise_mask
        live = float(mask.sum().item())
        total = float(mask.numel())
        shape_name = type(getattr(latent_tools, "target_shape", None)).__name__
        share = (live / total) if total else 0.0
        print(
            f"[ltx-retake][mask] {shape_name}: {live:,.0f} of {total:,.0f} tokens denoisable "
            f"({share:.1%}) for {self.start_time:.3f}-{self.end_time:.3f}s @ {self.fps:.3f} fps",
            file=sys.stderr,
        )
        if live == 0:
            print(
                "[ltx-retake][mask] EMPTY. Nothing in the window is denoisable, so the output is "
                "the input by construction. The window fell between token boundaries or the fps is "
                "wrong -- widen the selection.",
                file=sys.stderr,
            )
        return state

    noise_mask_cond.TemporalRegionMask.apply_to = logged_apply

    original_calculate = guiders_module.MultiModalGuider.calculate
    cfg_reported: set[int] = set()

    def logged_calculate(self, cond, uncond_text, uncond_perturbed, uncond_modality):  # noqa: ANN001
        out = original_calculate(self, cond, uncond_text, uncond_perturbed, uncond_modality)
        key = cond.shape[1] if isinstance(cond, torch.Tensor) and cond.dim() > 1 else 0
        if isinstance(uncond_text, torch.Tensor) and key not in cfg_reported:
            cfg_reported.add(key)
            scale = float(cond.float().abs().mean().item()) + 1e-8
            gap = float((cond.float() - uncond_text.float()).abs().mean().item())
            print(
                f"[ltx-retake][cfg] |cond-uncond| = {gap:.6f} = {gap / scale:.3%} of |cond| "
                f"at cfg {self.params.cfg_scale:.2f} ({key:,} tokens)",
                file=sys.stderr,
            )
            if gap / scale < 0.01:
                print(
                    "[ltx-retake][cfg] The conditioned and unconditioned predictions are within 1% "
                    "of each other. This checkpoint is guidance-distilled: CFG multiplies that "
                    "gap, so raising --cfg-scale cannot change the result. Use --stg-scale / "
                    "--modality-scale instead.",
                    file=sys.stderr,
                )
        return out

    guiders_module.MultiModalGuider.calculate = logged_calculate

    original_post = samplers.post_process_latent
    step_counter = {"n": 0}

    def logged_post(denoised, denoise_mask, clean):  # noqa: ANN001
        out = original_post(denoised, denoise_mask, clean)
        mask = denoise_mask.float()
        live = float(mask.sum().item())
        if live > 0:
            width = out.shape[-1] if out.dim() == 3 else 1
            denominator = live * width
            moved = float(((out.float() - clean.float()).abs() * mask).sum().item()) / denominator
            baseline = float((clean.float().abs() * mask).sum().item()) / denominator + 1e-8
            step_counter["n"] += 1
            print(
                f"[ltx-retake][step {step_counter['n']:02d}] in-mask movement {moved:.5f} "
                f"= {moved / baseline:.2%} of the original latent magnitude",
                file=sys.stderr,
            )
        return out

    samplers.post_process_latent = logged_post


def run_inference(args: argparse.Namespace) -> None:
    # Started before the heavy imports, because `import torch` is itself tens of seconds on a
    # cold process and a resident worker would pay it once instead of every run.
    timer = StageTimer("ltx-retake")
    import torch
    from ltx_core.quantization.fp8_scaled_mm import build_policy as build_fp8_scaled_policy
    from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
    from ltx_core.types import SpatioTemporalScaleFactors
    from ltx_core.components.guiders import MultiModalGuiderParams
    from aiv_runtime.safetensors_reader import describe_backend, install_pread_state_dict_loader
    from ltx_pipelines.retake import RetakePipeline
    from ltx_pipelines.utils.blocks import DiffusionStage
    from ltx_pipelines.utils.constants import DISTILLED_SIGMAS
    timer.stage("import")
    from ltx_pipelines.utils.media_io import encode_video, get_videostream_metadata
    from ltx_pipelines.utils.types import OffloadMode

    install_pread_state_dict_loader()
    print(f"[ltx-retake] checkpoint read: {describe_backend()}", file=sys.stderr)

    from aiv_runtime.ltx_single_stage import _scaled_fp8_streaming_stage

    device = torch.device("cuda")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(device) < (8, 9):
        raise RuntimeError("the 16 GB retake profile requires an RTX GPU with FP8 support")
    torch.cuda.reset_peak_memory_stats(device)
    source = get_videostream_metadata(args.video_path)
    budget = max_pixels()
    token_budget = max_pixel_frames()
    if budget != DEFAULT_MAX_PIXELS:
        print(f"[ltx-retake] pixel budget raised to {budget:,} (default {DEFAULT_MAX_PIXELS:,})", file=sys.stderr)
    print(
        f"[ltx-retake] source {source.width}x{source.height} - {source.frames} frames "
        f"@ {source.fps:.2f} fps",
        file=sys.stderr,
    )
    window = plan_retake_window(source, args.start_time, args.end_time, budget, token_budget)
    global _LAST_PIXEL_FRAMES
    _LAST_PIXEL_FRAMES = pixel_frames(source.width, source.height, window["frames"])
    cost = _LAST_PIXEL_FRAMES
    print(
        f"[ltx-retake] window {window['frames']} frames from {window['startSeconds']:.2f}s "
        f"({window['seconds']:.2f}s) around a {window['selectionFrames']}-frame selection "
        f"- {cost:,} pixel-frames ({cost / token_budget:.0%} of the {token_budget:,} budget)",
        file=sys.stderr,
    )

    # The clip the model is handed is the window, not the source. Everything downstream works in
    # the window's timebase, which is why the selection is remapped here and only here.
    scratch = Path(tempfile.mkdtemp(prefix="aiv-retake-"))
    retake_path = args.video_path
    if window["wholeClip"]:
        print("[ltx-retake] the clip already fits the profile; using it whole", file=sys.stderr)
    else:
        retake_path = extract_retake_window(
            args.video_path, window, scratch / "retake-window.mp4"
        )
    time_scale = SpatioTemporalScaleFactors.default().time
    if time_scale != FRAMES_PER_LATENT:
        raise RuntimeError(
            f"the pinned LTX tree reports a temporal scale of {time_scale}, but `ltx_geometry` "
            f"pins {FRAMES_PER_LATENT}. The panel and the Rust validator snap to the pinned value, "
            "so they would now be building requests this worker cannot run."
        )
    # The SOURCE no longer has to be 8n+1 -- the window cut out of it is, by construction.
    latent_frame_seconds = time_scale / float(source.fps)
    selection_seconds = args.end_time - args.start_time
    if selection_seconds < latent_frame_seconds:
        raise ValueError(
            f"retake range is {selection_seconds:.3f}s but one latent frame is "
            f"{latent_frame_seconds:.3f}s ({time_scale} source frames). Select at least that much "
            f"or nothing meaningful can be regenerated."
        )
    if selection_seconds < ADVISED_MIN_WINDOW_SECONDS:
        print(
            f"[ltx-retake] WARNING: the {selection_seconds:.2f}s selection is under the {ADVISED_MIN_WINDOW_SECONDS:.0f}s "
            f"minimum Lightricks' own hosted retake enforces. It will run -- roughly "
            f"{max(1, round(selection_seconds / latent_frame_seconds))} latent frame(s) get noised -- but the "
            f"pinned frames on either side dominate the attention over a window this short, and the "
            f"usual result is that the original comes back essentially unchanged. Widen the "
            f"selection before concluding the prompt is being ignored.",
            file=sys.stderr,
        )
    duration = source.frames / float(source.fps)
    if args.end_time > duration + 0.05:
        raise ValueError(f"retake end time {args.end_time:.3f}s exceeds source duration {duration:.3f}s")

    stg_blocks = parse_blocks(args.stg_blocks)
    guided = guidance_is_live(args)
    if guided:
        passes = 1 + 1 + (1 if args.stg_scale > 0.0 else 0) + (1 if abs(args.modality_scale - 1.0) > 1e-6 else 0)
        print(
            f"[ltx-retake] guidance {args.guidance}: cfg {args.cfg_scale:.2f} video / "
            f"{args.audio_cfg_scale:.2f} audio, stg {args.stg_scale:.2f} on blocks "
            f"{stg_blocks or '-'}, modality {args.modality_scale:.2f}, "
            f"rescale {args.rescale_scale:.2f} -- {passes} transformer passes per step, "
            f"{passes} x {len(DISTILLED_SIGMAS) - 1} = {passes * (len(DISTILLED_SIGMAS) - 1)} "
            f"transformer evaluations for this retake",
            file=sys.stderr,
        )
        if args.stg_scale <= 0.0 and abs(args.modality_scale - 1.0) <= 1e-6:
            print(
                "[ltx-retake] WARNING: stg and modality are both inert, so cfg is the only live "
                "dial -- and this checkpoint is guidance-distilled, which is what drives its "
                "contribution toward zero. This is the configuration that produced the 'retake "
                "strength does nothing' reports; the [cfg] diagnostic below measures whether the "
                "gap has in fact collapsed.",
                file=sys.stderr,
            )
    else:
        print(
            f"[ltx-retake] guidance {args.guidance}: unguided, "
            f"1 transformer pass per step -- {len(DISTILLED_SIGMAS) - 1} evaluations for this "
            f"retake. The prompt conditions but does not steer, which is what WanGP runs on "
            f"these weights. If the [latent] diagnostic below reports the masked region came "
            f"back unchanged, the prompt had nothing pushing it: rerun with "
            f"--guidance experimental-guided-distilled, which costs 4 passes per step.",
            file=sys.stderr,
        )
    global _DIAGNOSTICS_INSTALLED
    if args.diagnostics and not _DIAGNOSTICS_INSTALLED:
        install_diagnostics()
        _DIAGNOSTICS_INSTALLED = True
    offload_mode = offload_mode_for(args.offload)
    cpu_slots = cpu_slots_for(args.offload)
    print(f"[ltx] {describe_profile(args.offload)}", file=sys.stderr)
    cache_key = (
        args.checkpoint_path, args.gemma_root, args.offload, args.attention_backend, guided,
    )
    pipeline = _PIPELINE_CACHE.get(cache_key)

    if pipeline is None:
        quantization = build_fp8_scaled_policy(args.checkpoint_path)
        original_factory = DiffusionStage.__dict__["from_checkpoint"]

        def streaming_factory(
            cls, checkpoint_path, dtype, device, loras=(), quantization=None,
            registry=None, compilation_config=None, alloc_trim_strategy=None,
            offload_mode=OffloadMode.NONE, model_configurator=None, model_sd_ops=None,
        ):
            if loras:
                raise ValueError("the scaled-FP8 16 GB retake adapter does not fuse LoRAs")
            return _scaled_fp8_streaming_stage(
                checkpoint_path, dtype, device, offload_mode, quantization, cpu_slots
            )

        DiffusionStage.from_checkpoint = classmethod(streaming_factory)
        try:
            pipeline = RetakePipeline(
                checkpoint_path=args.checkpoint_path,
                gemma_root=args.gemma_root,
                loras=[],
                quantization=quantization,
                # This hybrid is intentionally exposed as experimental for distilled weights.
                distilled=not guided,
                offload_mode=offload_mode,
            )
        finally:
            DiffusionStage.from_checkpoint = original_factory
        active_attention, attention_swapped, attention_reason = install_stage_attention(
            args.attention_backend,
            pipeline.stage,
            log=lambda message: print(f"[ltx-retake] {message}", file=sys.stderr),
        )
        print(
            f"[ltx] {describe_attention(args.attention_backend, active_attention, attention_swapped, attention_reason)}",
            file=sys.stderr,
        )
        _PIPELINE_CACHE.clear()
        _PIPELINE_CACHE[cache_key] = pipeline
        print("[ltx-retake] resident pipeline created", file=sys.stderr)
    else:
        print("[ltx-retake] resident pipeline reused", file=sys.stderr)
    # Construction is lazy: calling this "load" produced a fictitious sub-second measurement even
    # though the transformer is built when the stage context is entered during pipeline().
    timer.stage("pipeline setup")

    model_load = {"seconds": 0.0}
    original_transformer_context = getattr(
        pipeline.stage, "_aiv_original_transformer_context", pipeline.stage._transformer_ctx
    )
    pipeline.stage._aiv_original_transformer_context = original_transformer_context
    from contextlib import contextmanager

    @contextmanager
    def timed_transformer_context(**kwargs):
        before = time.perf_counter()
        with original_transformer_context(**kwargs) as transformer:
            model_load["seconds"] += time.perf_counter() - before
            yield transformer

    pipeline.stage._transformer_ctx = timed_transformer_context

    regenerate_video = args.mode in ("video-audio", "video")
    regenerate_audio = args.mode in ("video-audio", "audio")
    tiling = TilingConfig.default()
    guider_kwargs = dict(
        stg_scale=args.stg_scale,
        stg_blocks=stg_blocks,
        rescale_scale=args.rescale_scale,
        modality_scale=args.modality_scale,
    )
    video, audio = pipeline(
        video_path=retake_path,
        prompt=args.prompt,
        start_time=window["rangeIn"],
        end_time=window["rangeOut"],
        seed=args.seed,
        regenerate_video=regenerate_video,
        regenerate_audio=regenerate_audio,
        tiling_config=tiling,
        max_batch_size=1,
        sigmas=DISTILLED_SIGMAS,
        negative_prompt=args.negative_prompt,
        enhance_prompt=args.enhance_prompt,
        video_guider_params=MultiModalGuiderParams(cfg_scale=args.cfg_scale, **guider_kwargs) if guided else None,
        audio_guider_params=MultiModalGuiderParams(cfg_scale=args.audio_cfg_scale, **guider_kwargs) if guided else None,
    )
    model_and_inference = timer.stage("model load + inference + decode")
    print(
        f"[ltx-retake] measured transformer context load {model_load['seconds']:.1f}s; "
        f"remaining pipeline work {max(0.0, model_and_inference-model_load['seconds']):.1f}s",
        file=sys.stderr,
    )
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    # The output is the REGENERATED RANGE, not the window and not the clip.
    #
    # `replaceClipRange` takes the returned asset from its own start for the length of the
    # selection, so anything longer put the wrong content on the timeline: a whole-clip output and
    # a selection at 5-8s spliced the clip's untouched opening over 5-8s, silently. Returning
    # exactly what was asked for makes that splice correct by construction, and it keeps the
    # untouched parts of the clip as they were -- they are never re-encoded, because they are
    # never in the output.
    #
    # Slicing the decoder's iterator rather than the encoded file also means the frames outside
    # the selection are never decoded.
    first = window["selectionStartFrame"] - window["startFrame"]
    last = window["selectionEndFrame"] - window["startFrame"]
    print(
        f"[ltx-retake] returning frames {first}-{last} of the window "
        f"({(last - first) / window['fps']:.2f}s, the range that was selected)",
        file=sys.stderr,
    )
    model_output = scratch / "model-range.mp4"
    encode_video(
        video=slice_decoded_frames(video, first, last),
        fps=window["fps"],
        audio=slice_decoded_audio(audio, window["rangeIn"], window["rangeOut"]),
        output_path=str(model_output),
        video_chunks_number=get_video_chunks_number(window["frames"], tiling),
    )
    finalize_retake_output(args.mode, args.video_path, model_output, args.output_path, window)
    preview_path = str(Path(args.output_path).with_name(Path(args.output_path).stem + "-seam-preview.mp4"))
    build_context_preview(args.video_path, args.output_path, preview_path, window)
    del video, audio
    shutil.rmtree(scratch, ignore_errors=True)
    # The number this whole exercise exists to obtain.
    peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
    reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)
    timer.stage("encode + preserve + seam preview")
    timer.report()
    print(f"[ltx-retake] peak VRAM {peak:,.0f} MiB allocated / {reserved:,.0f} MiB reserved", file=sys.stderr)
    # Pair the peak with the token count in one line, so a probe run answers "where is the real
    # ceiling" without anyone having to cross-reference two messages.
    print(
        f"[ltx-retake] calibration: {_LAST_PIXEL_FRAMES:,} pixel-frames -> {peak:,.0f} MiB peak "
        f"({peak / max(1, _LAST_PIXEL_FRAMES) * 1_000_000:.1f} MiB per megatoken)",
        file=sys.stderr,
    )
    torch.cuda.empty_cache()


def result_payload(args: argparse.Namespace) -> dict:
    from ltx_pipelines.utils.media_io import get_videostream_metadata
    source = get_videostream_metadata(args.video_path)
    window = plan_retake_window(source, args.start_time, args.end_time, max_pixels(), max_pixel_frames())
    return {
        "ok": True,
        "outputPath": args.output_path,
        "previewPath": str(Path(args.output_path).with_name(Path(args.output_path).stem + "-seam-preview.mp4")),
        "mode": args.mode,
        "effectiveStartTime": window["effectiveStartTime"],
        "effectiveEndTime": window["effectiveEndTime"],
        "startFrame": window["selectionStartFrame"],
        "endFrame": window["selectionEndFrame"],
        "fpsNumerator": window["fpsNumerator"],
        "fpsDenominator": window["fpsDenominator"],
    }


def serve(args: argparse.Namespace) -> int:
    """Serve sequential retakes while keeping imports and the compatible pipeline resident."""
    import traceback
    torch = __import__("torch")
    fields = {
        "videoPath": "video_path", "outputPath": "output_path", "prompt": "prompt",
        "negativePrompt": "negative_prompt", "startTime": "start_time", "endTime": "end_time",
        "seed": "seed", "mode": "mode", "enhancePrompt": "enhance_prompt",
    }
    print(json.dumps({"ready": True, "protocol": "aiv-ltx-retake-jsonl-v1"}), flush=True)
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            for external, internal in fields.items():
                if external in request:
                    setattr(args, internal, request[external])
            validate(args)
            if args.dry_run:
                payload = {"ok": True, "mode": args.mode, "offload": args.offload}
            else:
                with torch.inference_mode():
                    run_inference(args)
                payload = result_payload(args)
            print(json.dumps(payload), flush=True)
        except Exception as error:  # keep the resident process available after a bad request
            traceback.print_exc(file=sys.stderr)
            print(json.dumps({"ok": False, "error": str(error)}), flush=True)
    return 0


def main() -> int:
    args = build_parser().parse_args()
    if args.server:
        return serve(args)
    validate(args)
    if args.dry_run:
        print(json.dumps({"ok": True, "mode": args.mode, "offload": args.offload, "profile": "scaled-fp8-streaming-16gb"}))
        return 0
    with __import__("torch").inference_mode():
        run_inference(args)
    print(json.dumps(result_payload(args)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
