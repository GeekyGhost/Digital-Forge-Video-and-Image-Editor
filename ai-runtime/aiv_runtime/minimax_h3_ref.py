"""Isolated MiniMax-H3 Ref2VA worker for the optional reference-model pack.

The FL2VA worker stays untouched: this entry point loads ``transformer_ref`` and the dedicated
LightX2V Ref2VA four-step adapter only.  It keeps the same two-process conditioner/denoiser split
used by the 16 GB FL2VA path, while imposing the editor's deliberately conservative first-release
limits in the Rust command boundary.
"""

from __future__ import annotations

import argparse
import contextlib
import faulthandler
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import NamedTuple

faulthandler.enable()

_PACKAGE_ROOT = str(Path(__file__).resolve().parent.parent)
if _PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, _PACKAGE_ROOT)

from aiv_runtime import minimax_h3 as base  # noqa: E402
from aiv_runtime.minimax_h3_first_block_cache import (  # noqa: E402
    DEFAULT_FIRST_BLOCK_CACHE_THRESHOLD,
    install_first_block_cache,
    validate_first_block_cache_threshold,
)

H3_AUDIO_SAMPLE_RATE = 32_000

#: MiniMax's published reference recipe is 50 evaluations. The range is wider at both ends because
#: Ref2VA's only alternative is a v0.1 *preview* adapter at four, so the useful ground between the
#: two is worth reaching: ComfyUI's own R2V workflow defaults to 20.
BASE_DEFAULT_STEPS = 50
BASE_STEPS_RANGE = (10, 60)

# Ref2VA's released reference-video canvas is 768 x 1344. Qwen3-VL samples the video at
# 2 fps, but its vision tower attends over the spatial patches before they are merged into
# the presentation. A minimum-length 768p reference therefore asks native SDPA for more
# than 21 GiB in one allocation. Keep the generated canvas at the requested 544p while
# using a deliberately smaller, shared canvas for reference conditioning on the 16 GB pack.
# 384 is the conservative tier: 448 still left only about 25 MiB free and thrashed on a
# 16 GiB card in a full 5.2-second smoke. At 384 the pre-merge attention matrix is one
# sixteenth the stock 768p area, while generation itself remains at the requested 544p.
H3_REF_VIDEO_SHORT_EDGE = 384
#: The released reference-video budget is the generation's own: a 768 short edge at 768 x 1344,
#: i.e. a long edge 1.75x the short one. Deriving the area from whichever short edge is in force
#: keeps every tier the same shape as upstream's -- 384 -> 384*672, 768 -> 768*1344 exactly -- so
#: raising the tier actually raises the budget instead of being silently re-capped here.
H3_REF_VIDEO_ASPECT_BUDGET = 1344 / 768


def _ref_video_max_pixels(short_edge: int) -> int:
    return short_edge * round(short_edge * H3_REF_VIDEO_ASPECT_BUDGET)


H3_REF_VIDEO_MAX_PIXELS = _ref_video_max_pixels(H3_REF_VIDEO_SHORT_EDGE)

# Reference *images* are a different problem from reference videos and were wrongly held to the
# same number. The released checkpoint encodes them at a 2048 short edge with no area cap, because
# they never bind the generated geometry -- they are the identity signal, and their token cost is
# one still, not a 2 fps temporal stack. ComfyUI exposes the choice as `ref_image_size`: `match`
# (scale the reference to the generation's own pixel budget) or `max` (2048, documented as
# "stronger identity fidelity", several times slower because reference tokens ride through every
# sampling step).
#
# `None` here means `match`: the generated canvas's own short edge. At 544p that is 544 rather
# than the 384 this used to be pinned to, i.e. two thirds more linear detail on every identity
# reference for no extra risk on the path that actually OOMed. AIV_H3_REF_IMAGE_SHORT_EDGE takes
# an absolute value; set it to 2048 for `max`.
H3_REF_IMAGE_SHORT_EDGE: int | None = None

# MiniMax-H3's own clock -- `MINIMAX_H3_FPS`, i.e. `components.fps`. Every reference is resampled
# onto it by dropping and duplicating whole frames, so it is the rate a source frame budget is
# expressed against.
H3_FPS = 24.0


def _geometry_override(name: str, fallback: int) -> int:
    """An `AIV_H3_*` reference-geometry override, or `fallback`.

    These exist so the conditioning canvas can be moved without a rebuild: the values below are a
    conservative 16 GB profile, not model limits, and the encode and denoise stages are separate
    processes that both inherit this environment, so an override cannot desynchronise them.
    """
    value = os.environ.get(name)
    if not value:
        return fallback
    resolved = int(value)
    if resolved < 32:
        raise ValueError(f"{name} must be at least 32, got {resolved}")
    return resolved


def _rebuild_reference_stubs(descriptors: list[dict]):
    """Rebuild lightweight reference descriptors for the denoiser layout stage."""
    from diffusers.modular_pipelines.minimax_h3.references import (
        MiniMaxH3AudioReference,
        MiniMaxH3ImageReference,
        MiniMaxH3VideoReference,
    )
    import torch

    stubs = []
    for d in descriptors:
        kind = d["kind"]
        if kind == "image":
            stubs.append(MiniMaxH3ImageReference(image=None))
        elif kind == "video":
            audio_stub = torch.empty(2, 1) if d.get("has_audio") else None
            stubs.append(MiniMaxH3VideoReference(frames=None, fps=float(d.get("fps", 24.0)), audio=audio_stub))
        elif kind == "audio":
            stubs.append(MiniMaxH3AudioReference(audio=torch.empty(2, 1), sample_rate=H3_AUDIO_SAMPLE_RATE))
    return stubs


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="MiniMax-H3 Ref2VA Base / four-step Turbo worker")
    value.add_argument("--repo-root", required=True)
    value.add_argument("--transformer-path", required=True)
    value.add_argument("--text-encoder-path", required=True)
    # Optional, because Base needs no adapter at all. `validate` still demands it for Turbo.
    value.add_argument("--turbo-lora-path", default=None)
    value.add_argument(
        "--sampling-mode",
        choices=("base", "turbo-4"),
        default="base",
        help="LightX2V's four-evaluation Ref2V adapter, or Base at its own step count. The adapter "
        "published for Ref2V is v0.1, a preview; Base is the quality reference.",
    )
    value.add_argument("--references-json", required=True)
    value.add_argument("--prompt", required=True)
    value.add_argument("--output-path", required=True)
    value.add_argument("--audio-path", required=True)
    value.add_argument("--width", type=int, required=True)
    value.add_argument("--height", type=int, required=True)
    value.add_argument("--num-frames", type=int, required=True)
    value.add_argument("--seed", type=int, default=42)
    value.add_argument(
        "--attention-backend",
        # Native, unlike the generation adapter: see `default_attention` in h3_ref.rs.
        choices=("native", "sage", "sage2_patched", "sage1", "sage2_stock"),
        default="native",
    )
    value.add_argument("--offload", choices=tuple(base.OFFLOAD_PROFILES), default="balanced")
    value.add_argument("--vae-tile", type=int, default=base.VAE_TILE_SIZE)
    value.add_argument(
        "--first-block-cache",
        action="store_true",
        help="Experimental step skipping. Default-off until a user approves the Ref2VA baseline.",
    )
    value.add_argument(
        "--first-block-cache-threshold",
        type=float,
        default=DEFAULT_FIRST_BLOCK_CACHE_THRESHOLD,
    )
    value.add_argument(
        "--profile-denoise",
        action="store_true",
        help="Write opt-in per-step H3 timing, memory, GPU, and offload diagnostics as JSONL.",
    )
    value.add_argument(
        "--profile-path",
        default=None,
        help="Diagnostic JSONL path; defaults beside the output MP4. Also enables profiling.",
    )
    # 0 means "whatever the mode's own default is": 4 for Turbo, the released 50 for Base.
    value.add_argument("--steps", type=int, default=0)
    value.add_argument("--video-shift", type=float, default=None)
    value.add_argument("--ref-image-short-edge", type=int, default=None)
    value.add_argument("--ref-video-short-edge", type=int, default=None)
    value.add_argument("--stage", choices=("all", "encode", "denoise", "decode"), default="all")
    value.add_argument("--embeds-path")
    value.add_argument("--latents-path")
    return value


def validate(args: argparse.Namespace) -> None:
    base.validate_canvas(args.width, args.height)
    base.validate_num_frames(args.num_frames)
    validate_first_block_cache_threshold(
        getattr(args, "first_block_cache_threshold", DEFAULT_FIRST_BLOCK_CACHE_THRESHOLD)
    )
    if not args.prompt.strip():
        raise ValueError("prompt cannot be empty")
    transformer_path = Path(args.transformer_path)
    if not transformer_path.is_file():
        raise FileNotFoundError(f"transformer path not found: {transformer_path}")
    if transformer_path.suffix.lower() != ".safetensors":
        raise ValueError("Ref2VA denoisers must be safetensors files; .pt checkpoints are unsupported")
    if args.sampling_mode in base.TURBO_NFE:
        expected = base.TURBO_NFE[args.sampling_mode]
        if args.steps != expected:
            raise ValueError(
                f"{args.sampling_mode} requires exactly {expected} transformer evaluations, got "
                f"{args.steps}. The adapter is distilled to that count; it is not a speed dial."
            )
        if not args.turbo_lora_path:
            raise ValueError(f"{args.sampling_mode} requires --turbo-lora-path")
        turbo_lora_path = Path(args.turbo_lora_path)
        if not turbo_lora_path.is_file():
            is_embedded = False
            if transformer_path.suffix == ".safetensors":
                # Header only. This runs during argument validation, so it used to charge 20 GiB
                # of Windows commit before the worker had decided whether it could run at all.
                from aiv_runtime.safetensors_header import metadata as read_metadata

                is_embedded = read_metadata(transformer_path).get("is_turbo_baked") == "True"
            if not is_embedded:
                raise FileNotFoundError(f"turbo lora path not found: {turbo_lora_path}")
    elif not BASE_STEPS_RANGE[0] <= args.steps <= BASE_STEPS_RANGE[1]:
        raise ValueError(
            f"base sampling takes {BASE_STEPS_RANGE[0]}-{BASE_STEPS_RANGE[1]} transformer "
            f"evaluations, got {args.steps}. MiniMax's published reference recipe is "
            f"{BASE_DEFAULT_STEPS}."
        )
    conditioner_path = Path(args.text_encoder_path)
    if not conditioner_path.is_file():
        raise FileNotFoundError(f"published text encoder not found: {conditioner_path}")
    if conditioner_path.suffix.lower() != ".safetensors":
        raise ValueError("Ref2VA conditioner weights must be safetensors files; .pt/.bin is unsupported")
    if args.vae_tile < base.VAE_TILE_SIZE or args.vae_tile % 16:
        raise ValueError(
            f"vae tile must be at least {base.VAE_TILE_SIZE} pixels and divisible by 16, got {args.vae_tile}"
        )
    if args.stage in ("encode", "denoise") and not args.embeds_path:
        raise ValueError(f"the internal {args.stage} stage requires --embeds-path")
    if args.stage in ("denoise", "decode") and not args.latents_path:
        raise ValueError(f"the internal {args.stage} stage requires --latents-path")
    references = json.loads(args.references_json)
    if not isinstance(references, list) or not references:
        raise ValueError("Ref2VA requires at least one reference")


def _audio_reference_at_h3_rate(path: str, audio_reference_type):
    """Decode one soundtrack through FFmpeg at the audio VAE's native rate.

    Diffusers delegates non-native resampling to torchaudio. The isolated H3 environment does not
    otherwise need that large binary companion package, while FFmpeg is already a hard runtime
    dependency of the editor. Decode to a temporary PCM WAV so Diffusers receives an exact 32 kHz
    waveform and never enters its optional torchaudio branch.
    """
    handle, normalized_path = tempfile.mkstemp(suffix=".h3-ref-32k.wav")
    os.close(handle)
    try:
        completed = subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                "-i", str(path), "-map", "0:a:0", "-vn", "-ar", str(H3_AUDIO_SAMPLE_RATE),
                "-c:a", "pcm_s16le", normalized_path,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode:
            detail = completed.stderr.strip() or completed.stdout.strip() or "unknown FFmpeg error"
            raise RuntimeError(f"could not normalize the Ref2VA soundtrack to 32 kHz: {detail}")
        normalized = audio_reference_type.from_file(normalized_path)
        if normalized.sample_rate != H3_AUDIO_SAMPLE_RATE:
            raise RuntimeError(
                f"normalized Ref2VA soundtrack reported {normalized.sample_rate} Hz, "
                f"expected {H3_AUDIO_SAMPLE_RATE} Hz"
            )
        return normalized
    finally:
        Path(normalized_path).unlink(missing_ok=True)


class ReferenceDecodeBudget(NamedTuple):
    """What a reference video decode may throw away *before* it allocates it.

    The setup step normalizes every video reference in one fixed order: resample onto 24 fps,
    truncate to the generated frame count, rescale onto the shared conditioning canvas. Everything
    the first two steps drop is decoded, held in host RAM and then discarded, and everything the
    third one shrinks is held at source resolution until it gets there. A five-second 1080p
    reference is over a gigabyte of frames the request never reads.

    That mattered because ``load_references`` used to be evaluated as an *argument* to the
    ``pipeline(...)`` call in :func:`run_denoise`, so the decode ran with the 18.9 GiB denoiser
    already resident -- a hard process kill with no Python traceback, which is what exit code -1
    was.

    Both truncation and rescale commute with the fps resample, because the resample only drops and
    duplicates whole frames. Doing them at decode time is therefore not an approximation -- as long
    as the canvas matches the one the setup step would have resolved. :meth:`canvas_for` proves
    that per file and declines when it cannot.
    """

    # The *aligned* generated frame count, of the form 17 * n + 5, since that is what the setup
    # step truncates to.
    num_frames: int
    canvas_multiple: int
    canvas_short_edge: int
    canvas_max_pixels: int

    @classmethod
    def for_pipeline(cls, pipeline, num_frames: int) -> "ReferenceDecodeBudget":
        """The budget a `ref_pipeline` will hold its references to.

        Safe to call before `load_components`: the VAE-derived properties fall back to the released
        checkpoint's own values (17 frames per chunk, 5 latents, a canvas multiple of 32), which is
        exactly what the setup step reads at encode time, where no VAE is loaded either.
        """
        from diffusers.modular_pipelines.minimax_h3.modular_pipeline import align_num_frames

        return cls(
            align_num_frames(
                num_frames, pipeline.vae_frames_per_chunk, pipeline.vae_latents_per_chunk
            ),
            pipeline.canvas_multiple,
            pipeline.config.canvas_short_edge,
            pipeline.config.canvas_max_pixels,
        )

    def source_frames(self, fps: float) -> int:
        """How many frames of an `fps` source the setup step can still read after its resample."""
        if fps <= 0:
            raise ValueError(f"A reference video must have a positive frame rate, got {fps}.")
        # `_normalize_video_condition` turns n source frames into floor(n * 24/fps + 0.5) frames
        # and then slices `[:num_frames]`. Inverting that gives the exact bound; the two slack
        # frames make a rounding disagreement cost one duplicated tail frame instead of a
        # reference that is short, which would change the request silently.
        return math.ceil((self.num_frames + 1) * fps / H3_FPS) + 1

    def canvas_for(self, width: int, height: int) -> tuple[int, int] | None:
        """The setup step's `(height, width)` canvas for this source, or `None` to leave it alone.

        The setup step resolves the canvas from whatever geometry it is handed, so pre-scaling is
        only transparent where the canvas is a fixed point of that resolution. It is for every
        ratio tried so far, but a canvas that resolved to something else would condition the
        request on a twice-rescaled reference with nothing raised about it, so this returns `None`
        rather than assume.
        """
        from diffusers.modular_pipelines.minimax_h3.modular_pipeline import resolve_canvas_size

        rule = (self.canvas_multiple, self.canvas_short_edge, self.canvas_max_pixels)
        try:
            canvas = resolve_canvas_size(width, height, *rule)
            if resolve_canvas_size(canvas[1], canvas[0], *rule) != canvas:
                return None
        except ValueError:
            # An aspect ratio outside 1:4 to 4:1. The setup step raises a better-worded version of
            # this once it holds the frames; do not pre-empt it with one raised at decode time.
            return None
        return canvas


def _decode_video_reference(
    path: str,
    budget: ReferenceDecodeBudget,
    video_reference_type,
    audio_reference_type,
    want_audio: bool,
):
    """`MiniMaxH3VideoReference.from_file` without the pixels the request cannot read.

    Mirrors `references._decode_video_file`: the same PyAV decode, the same
    `average_rate or guessed_rate`, the same display-matrix rotation snapped to a quarter turn, the
    same soundtrack in its own pass over the container. It differs in three ways, all of them
    memory:

    - it stops at `budget.source_frames(fps)`;
    - it rescales onto the conditioning canvas while decoding, when `budget.canvas_for` allows it;
    - it writes into one preallocated array rather than stacking a list, which otherwise holds the
      frames twice at the moment of the stack.
    """
    import av
    import numpy as np
    from PIL import Image

    with av.open(path) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate or stream.guessed_rate)
        has_soundtrack = bool(container.streams.audio)
        limit = budget.source_frames(fps)
        buffer, canvas, turns, count, rows, columns = None, None, 0, 0, 0, 0
        for frame in container.decode(stream):
            if count >= limit:
                break
            pixels = frame.to_ndarray(format="rgb24")
            if buffer is None:
                # The display matrix rotation belongs to the stream, and PyAV surfaces it on every
                # frame of it, so the first frame carries the whole answer.
                turns = round(frame.rotation / 90.0) % 4
                height, width = pixels.shape[:2]
                if turns % 2:
                    height, width = width, height
                canvas = budget.canvas_for(width, height)
                rows, columns = canvas if canvas is not None else (height, width)
                # `nb_frames` is exact for MP4 and MOV, and absent for a stream that was never
                # indexed. Trust it only as a reason to allocate *less* than the budget: a source
                # shorter than the budget is the common case, and this is Windows, where an
                # untouched allocation is still charged against the commit limit.
                declared = stream.frames if stream.frames > 0 else limit
                buffer = np.empty((min(limit, max(declared, 1)), rows, columns, 3), dtype=np.uint8)
                base.log(
                    f"reference video: reading at most {limit} of its frames at {fps:g} fps for a "
                    f"{budget.num_frames}-frame request, "
                    + (
                        f"onto the {columns}x{rows} conditioning canvas"
                        if canvas is not None
                        else f"at its own {width}x{height}"
                    )
                )
            elif count == len(buffer):
                # The hint under-reported. Grow to the budget in one step rather than shorten the
                # reference, which would be a different request with nothing raised about it.
                buffer.resize((limit, rows, columns, 3), refcheck=False)
            if turns:
                # `ffmpeg` displays a frame upright by undoing the counterclockwise rotation the
                # display matrix carries.
                pixels = np.rot90(pixels, k=-turns, axes=(0, 1))
            if canvas is not None:
                pixels = np.asarray(
                    Image.fromarray(np.ascontiguousarray(pixels)).resize(
                        (canvas[1], canvas[0]), Image.Resampling.LANCZOS
                    )
                )
            buffer[count] = pixels
            count += 1

    if not count:
        raise ValueError(f"No video frames to decode in {path}.")
    audio, sample_rate = None, None
    if want_audio and has_soundtrack:
        # The public loader over the same container's audio stream, i.e. the second pass
        # `_decode_video_file` makes. Skipped outright when the caller excluded the soundtrack.
        soundtrack = audio_reference_type.from_file(path)
        audio, sample_rate = soundtrack.audio, soundtrack.sample_rate
    # A view of the preallocated buffer, not a copy: a source shorter than the budget leaves the
    # tail rows unread rather than paying to trim them.
    return video_reference_type(frames=buffer[:count], fps=fps, audio=audio, sample_rate=sample_rate)


def load_references(document: str, budget: ReferenceDecodeBudget | None = None):
    """Decode the reference media a Ref2VA request conditions on, in packed order.

    `budget` bounds a video reference to the frames and the canvas the request can actually read;
    without one every video is decoded whole, which is the stock behaviour and what the reference
    loader tests pin.
    """
    from diffusers.modular_pipelines.minimax_h3.references import (
        MiniMaxH3AudioReference,
        MiniMaxH3ImageReference,
        MiniMaxH3VideoReference,
    )

    values = []
    for entry in json.loads(document):
        kind = entry["kind"]
        path = entry["path"]
        if kind == "image":
            value = MiniMaxH3ImageReference.from_file(path)
        elif kind == "video":
            base.log(f"reference video: decoding {Path(path).name} to RGB frames with its container FPS")
            value = None
            # A URL or a missing path is the stock loader's business: it downloads the one and
            # raises a worded error for the other, and neither is worth reproducing here.
            if budget is not None and os.path.isfile(path):
                try:
                    value = _decode_video_reference(
                        path,
                        budget,
                        MiniMaxH3VideoReference,
                        MiniMaxH3AudioReference,
                        want_audio=bool(entry.get("includeEmbeddedAudio", True)),
                    )
                except ImportError as error:
                    # PyAV is what the stock loader needs too, so falling through does not work
                    # around anything -- it just lets that loader raise its own install
                    # instructions instead of this one's traceback.
                    base.log(f"reference video: bounded decode unavailable ({error})")
            if value is None:
                value = MiniMaxH3VideoReference.from_file(path)
            frame_shape = tuple(value.frames.shape)
            base.log(
                "reference video decoded: "
                f"{frame_shape[0]} frames · {frame_shape[2]}x{frame_shape[1]} · {value.fps:g} fps; "
                "Ref2VA normalizes to 24 fps, then the Qwen conditioner samples at 2 fps"
            )
            if not entry.get("includeEmbeddedAudio", True):
                value.audio = None
                value.sample_rate = None
                base.log("reference video soundtrack: excluded")
            elif value.audio is not None and value.sample_rate != H3_AUDIO_SAMPLE_RATE:
                base.log(
                    f"reference video audio: {value.sample_rate or 'unknown'} -> "
                    f"{H3_AUDIO_SAMPLE_RATE} Hz"
                )
                normalized = _audio_reference_at_h3_rate(path, MiniMaxH3AudioReference)
                value.audio = normalized.audio
                value.sample_rate = normalized.sample_rate
            elif value.audio is not None:
                base.log(f"reference video soundtrack: included at {value.sample_rate} Hz")
            else:
                base.log("reference video soundtrack: none present")
        elif kind == "audio":
            value = MiniMaxH3AudioReference.from_file(path)
            if value.sample_rate != H3_AUDIO_SAMPLE_RATE:
                base.log(
                    f"reference audio: {value.sample_rate or 'unknown'} -> "
                    f"{H3_AUDIO_SAMPLE_RATE} Hz"
                )
                value = _audio_reference_at_h3_rate(path, MiniMaxH3AudioReference)
        else:
            raise ValueError(f"unsupported reference kind: {kind}")
        values.append(value)
    return values


def qwen_conditioner_attention_context(torch):
    """Prefer cuDNN SDPA for Ref2VA's long grouped-query presentation.

    The Windows Torch build used by the optional pack exposes the Flash flag but was compiled
    without Flash Attention, and its efficient-attention kernel is runtime-disabled. Native
    selection consequently falls back to the quadratic math kernel for Qwen's 64-query/8-KV-head
    GQA tensors. A modest mixed request then tries to allocate 42+ GiB. cuDNN SDPA supports both
    Qwen3-VL shapes used here and keeps the attention workspace linear-sized.

    Keep this scope local to the conditioner. It does not alter MiniMax's denoiser backend or the
    independent Sage toggle, and CPU/non-cuDNN environments retain their normal native behavior.
    """
    if not torch.cuda.is_available():
        return contextlib.nullcontext()
    attention = getattr(torch.nn, "attention", None)
    cudnn_enabled = getattr(torch.backends.cuda, "cudnn_sdp_enabled", lambda: False)
    if attention is None or not cudnn_enabled():
        return contextlib.nullcontext()
    return attention.sdpa_kernel([attention.SDPBackend.CUDNN_ATTENTION])


def ref_pipeline(args: argparse.Namespace, keep: set[str]):
    from diffusers import ModularPipeline

    blocks = ModularPipeline.from_pretrained(args.repo_root).blocks.get_workflow("ref2va")
    for name in tuple(blocks.sub_blocks):
        if name not in keep:
            blocks.sub_blocks.pop(name)
    pipeline = blocks.init_pipeline(args.repo_root)
    # The released stock workflow expands every image reference to a 2048 px short edge and every
    # video reference to a 768 px short edge. Those are server-GPU defaults. Qwen's vision tower
    # attends over the unmerged video patches, so even a five-second 768p reference can request a
    # single 21+ GiB SDPA allocation. Use one conservative reference layout in both worker stages:
    # prompt embeddings and VAE reference latents must never be prepared from different geometry.
    image_short_edge = getattr(args, "ref_image_short_edge", None) or _geometry_override(
        "AIV_H3_REF_IMAGE_SHORT_EDGE",
        H3_REF_IMAGE_SHORT_EDGE or min(args.width, args.height),
    )
    # An explicit choice -- the panel's tier, or the environment override -- is honoured as given.
    # Only the fallback is held under the generated canvas. Upstream does not clamp a reference
    # video to the generation at all (ComfyUI fits it under the same 768x1344 budget either way),
    # so clamping an explicit 768 back down to a 544 canvas would be one more control that quietly
    # does nothing, which is the class of fault this pass exists to remove.
    requested_video_edge = getattr(args, "ref_video_short_edge", None) or os.environ.get(
        "AIV_H3_REF_VIDEO_SHORT_EDGE"
    )
    video_short_edge = _geometry_override(
        "AIV_H3_REF_VIDEO_SHORT_EDGE", int(requested_video_edge or H3_REF_VIDEO_SHORT_EDGE)
    )
    # Derived from the short edge actually in force, not from the module default -- otherwise
    # choosing a 768 px reference video tier left the area pinned at the 384 tier's 258k and the
    # setting did nothing.
    video_max_pixels = _geometry_override(
        "AIV_H3_REF_VIDEO_MAX_PIXELS", _ref_video_max_pixels(video_short_edge)
    )
    # Deliberately NOT clamped to the generated canvas. A reference image never binds the
    # generated geometry, and the released default is 2048 with no area cap at all, so clamping
    # would make `max` mode unreachable -- which is the one lever that buys identity fidelity.
    pipeline.config.reference_image_short_edge = image_short_edge
    if requested_video_edge:
        pipeline.config.canvas_short_edge = video_short_edge
        pipeline.config.canvas_max_pixels = video_max_pixels
    else:
        pipeline.config.canvas_short_edge = min(args.width, args.height, video_short_edge)
        pipeline.config.canvas_max_pixels = min(args.width * args.height, video_max_pixels)
    base.log(
        "reference conditioning canvas: "
        f"image short edge <= {pipeline.config.reference_image_short_edge}, "
        f"video short edge <= {pipeline.config.canvas_short_edge}"
        f"{' (requested)' if requested_video_edge else ' (held under the generated canvas)'}, "
        f"video pixels <= {pipeline.config.canvas_max_pixels:,}"
    )
    return pipeline


def run_encode(args: argparse.Namespace) -> None:
    import torch
    from diffusers.hooks import apply_group_offloading

    base.log(f"Ref2VA encode · {args.width}x{args.height} · {args.num_frames} frames")
    pipeline = ref_pipeline(args, {"before_encode", "text_encoder", "vae_encoder"})
    # Before the conditioner, for the same reason the denoise stage does it -- see
    # `ReferenceDecodeBudget`. Never as an argument to the `pipeline(...)` call below.
    references = load_references(
        args.references_json, ReferenceDecodeBudget.for_pipeline(pipeline, args.num_frames)
    )
    pipeline.update_components(text_encoder=base.load_text_encoder(args))
    pipeline.load_components(
        names=["tokenizer", "processor"],
        pretrained_model_name_or_path=args.repo_root,
        dtype=torch.bfloat16,
    )
    pipeline.load_components(
        names=["vae"], pretrained_model_name_or_path=args.repo_root, dtype=torch.float16
    )
    pipeline.load_components(
        names=["audio_vae"], pretrained_model_name_or_path=args.repo_root, dtype=torch.float32
    )
    if base.ensure_mm_token_type_ids(pipeline.processor):
        base.log("conditioner: installed Transformers 4.57 multimodal-token compatibility shim")
    pipeline.text_encoder.requires_grad_(False)
    apply_group_offloading(
        pipeline.text_encoder.model,
        offload_type="leaf_level",
        onload_device=torch.device("cuda"),
        offload_device=torch.device("cpu"),
    )
    if hasattr(pipeline.vae, "enable_tiling"):
        pipeline.vae.enable_tiling(
            tile_sample_min_height=args.vae_tile, tile_sample_min_width=args.vae_tile
        )
    pipeline.vae.requires_grad_(False)
    # The `dtype=torch.float16` this was loaded with is a no-op -- `_keep_in_fp32_modules` covers
    # the whole model -- so the released FP32 is what arrives here unless AIV_H3_VAE_DTYPE says
    # otherwise. Cast before the offload hooks are attached, never after.
    base.cast_video_vae(pipeline.vae)
    apply_group_offloading(
        pipeline.vae,
        offload_type=base.VIDEO_VAE_OFFLOAD_TYPE,
        onload_device=torch.device("cuda"),
        offload_device=torch.device("cpu"),
    )
    pipeline.audio_vae.to("cuda")
    if base.install_qwen3vl_pos_embed_device_shim(pipeline.text_encoder.model, "cuda"):
        base.log("conditioner: aligned Qwen3-VL positional interpolation with CUDA leaf onload")
    attention_context = qwen_conditioner_attention_context(torch)
    if not isinstance(attention_context, contextlib.nullcontext):
        base.log("conditioner attention: forced cuDNN SDPA for long mixed-reference GQA")
    # No `generator=` here, deliberately, and it would change nothing if there were one. The
    # reference posteriors are sampled by `encode_vae_condition` under a fresh
    # `torch.Generator().manual_seed(components.keyframe_encode_seed)`, fixed at 42 in the
    # released recipe and independent of the request. Reference latents are therefore already
    # deterministic across runs with identical inputs, whatever the user seed, so the embeds cache
    # is not freezing an arbitrary draw. Read against the installed diffusers on 2026-09-08.
    with attention_context:
        state = pipeline(
            prompt=args.prompt,
            references=references,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
        )
    embeds = state.get("prompt_embeds")
    tags = state.get("text_token_tags")
    condition_latents = state.get("condition_latents")
    audio_condition_latents = state.get("audio_condition_latents")
    normalized_refs = state.get("normalized_references", [])

    if embeds is None or tags is None:
        raise RuntimeError("the Ref2VA conditioner produced no prompt embeds")

    ref_descriptors = [
        {"kind": r.kind, "has_audio": r.has_audio, "fps": getattr(r, "fps", 24.0)}
        for r in normalized_refs
    ]

    torch.save(
        {
            "prompt_embeds": embeds.to("cpu"),
            "text_token_tags": tags.to("cpu"),
            "condition_latents": [c.to("cpu") if isinstance(c, torch.Tensor) else c for c in (condition_latents or [])],
            "audio_condition_latents": [a.to("cpu") if isinstance(a, torch.Tensor) else a for a in (audio_condition_latents or [])],
            "reference_descriptors": ref_descriptors,
        },
        args.embeds_path,
    )


def run_denoise(args: argparse.Namespace) -> None:
    import torch

    mode_label = (
        f"{base.TURBO_NFE[args.sampling_mode]}-step Turbo"
        if args.sampling_mode in base.TURBO_NFE
        else f"Base · {args.steps} NFE"
    )
    base.log(f"Ref2VA denoise · {mode_label} · {args.width}x{args.height}")
    pipeline = ref_pipeline(args, {
        "denoise.prepare_layout",
        "denoise.prepare_condition_latents", "denoise.prepare_latents",
        "denoise.prepare_latents_ref2va", "denoise.set_timesteps", "denoise.denoise",
        "denoise.after_denoise",
    })
    transformer = base.load_transformer(args, subfolder="transformer_ref")
    if args.sampling_mode not in base.TURBO_NFE:
        base.log(f"sampling: Base · {args.steps} NFE · no acceleration adapter")
    elif getattr(transformer, "_is_turbo_baked", False):
        base.log("turbo LoRA: residual already attached by the ConvRot bridge · skipping PEFT")
    else:
        base.apply_turbo_lora(transformer, args.turbo_lora_path, base.TURBO_NFE[args.sampling_mode])
    pipeline.update_components(transformer_ref=transformer)
    pipeline.load_components(
        names=["scheduler", "audio_scheduler"],
        pretrained_model_name_or_path=args.repo_root,
        dtype=torch.bfloat16,
    )
    base.instrument_scheduler_grid(pipeline)
    # The shifts belong to the adapter. Base has none, so it keeps the scheduler's released
    # defaults -- exactly what FL2VA does, where `set_shift` is inside the Turbo branch too.
    # Overriding them for Base would be inventing a recipe MiniMax never published.
    if args.sampling_mode in base.TURBO_NFE:
        # Precedence lives in `turbo_shifts`: published pair, then this argument, then the
        # environment. It used to be resolved here, with the argument last -- and because the Rust
        # command builder always sent one, the published value was only ever reached by coincidence
        # and `AIV_TURBO_VIDEO_SHIFT` could not be reached at all.
        video_shift, audio_shift = base.turbo_shifts(
            args.turbo_lora_path, getattr(args, "video_shift", None)
        )
        base.assert_schedules_agree(video_shift, audio_shift, base.scheduler_grid_points(args))
        pipeline.scheduler.set_shift(video_shift)
        pipeline.audio_scheduler.set_shift(audio_shift)
        base.log(
            f"Turbo scheduler shifts · video={video_shift:g} · audio={audio_shift:g} · "
            f"grid={base.scheduler_grid_points(args)} points"
        )
    else:
        base.log(
            f"Base sampling · released scheduler shifts · "
            f"grid={base.scheduler_grid_points(args)} points"
        )
    base.configure_attention_backend(pipeline.transformer_ref, args.attention_backend)

    profile = base.effective_offload_profile(
        args.offload, getattr(pipeline.transformer_ref, "_aiv_fp8_mode", None)
    )
    if profile.get("is_resident") or args.offload in ("resident", "none"):
        base.log("denoiser: resident VRAM mode enabled · zero PCIe swapping")
        pipeline.transformer_ref.to("cuda")
    else:
        devices = {"onload_device": torch.device("cuda"), "offload_device": torch.device("cpu")}
        pipeline.transformer_ref.enable_group_offload(
            offload_type="block_level",
            num_blocks_per_group=profile["blocks_per_group"],
            use_stream=profile["use_stream"],
            record_stream=profile["record_stream"],
            low_cpu_mem_usage=profile["low_cpu_mem_usage"],
            **devices,
        )
    first_block_cache = install_first_block_cache(
        pipeline.transformer_ref,
        enabled=getattr(args, "first_block_cache", False),
        threshold=getattr(
            args, "first_block_cache_threshold", DEFAULT_FIRST_BLOCK_CACHE_THRESHOLD
        ),
        total_evaluations=args.steps,
    )
    if first_block_cache is not None:
        base.log(
            f"first block cache enabled · threshold={first_block_cache.threshold:.3f} · "
            f"warmup={first_block_cache.warmup_evaluations}/{first_block_cache.total_evaluations} evaluations"
        )
    saved = torch.load(args.embeds_path, map_location="cpu", weights_only=False)
    # Moved, NOT cast. These used to arrive as bfloat16 and it destroyed the conditioning
    # augmentation outright.
    #
    # The anchors are held just short of clean: `scale_noise(condition, 0.999, noise)` computes
    # `t*x0 + (1-t)*noise`, and it casts `t` to the *sample's* dtype first. bfloat16 has 7 mantissa
    # bits, so bf16(0.999) rounds to exactly 1.0 and `1.0 - t` is exactly 0.0 -- the noise term
    # vanishes and every anchor is conditioned at t = 1.0, which the released model's own
    # documentation calls off-distribution. Not "roughly ten ulps of rounding": the augmentation
    # does not happen at all. In float32 the same expression keeps 0.001 intact.
    #
    # `encode_vae_condition` hands these over as float32 already, so this is a round trip that was
    # throwing precision away for nothing. They are small next to an 18.9 GiB denoiser, and the
    # pipeline casts to the transformer's dtype after the mix.
    condition_latents = [
        c.to("cuda") if isinstance(c, torch.Tensor) else c
        for c in saved.get("condition_latents", [])
    ]
    audio_condition_latents = [
        a.to("cuda") if isinstance(a, torch.Tensor) else a
        for a in saved.get("audio_condition_latents", [])
    ]
    normalized_refs = _rebuild_reference_stubs(saved.get("reference_descriptors", []))

    # No `prompt=` here. The denoise block set has no text encoder, so the modular pipeline
    # reports it as an unexpected input and ignores it -- a warning that reads like the cause of
    # any failure that follows it. The conditioner's output is what this stage consumes.
    with base.denoise_profile_context(
        args, pipeline.transformer_ref, "ref2va", profile
    ), torch.inference_mode():
        result = pipeline(
            normalized_references=normalized_refs,
            condition_latents=condition_latents,
            audio_condition_latents=audio_condition_latents,
            prompt_embeds=saved["prompt_embeds"].to("cuda", dtype=torch.bfloat16),
            text_token_tags=saved["text_token_tags"],
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            # MiniMaxH3Scheduler counts the terminal zero as a grid point, so an N-evaluation
            # adapter needs N+1 points. Passing 4 here ran the four-step Ref2V LoRA at three
            # transformer evaluations. main() sets sampling_mode/steps precisely so this helper
            # can resolve it; tests/test_minimax_h3_runtime.py pins the +1 for turbo-4.
            num_inference_steps=base.scheduler_grid_points(args),
            generator=torch.Generator().manual_seed(args.seed),
        )
    base.log_memory("Ref2VA denoise complete")
    if first_block_cache is not None:
        base.log(first_block_cache.summary())
    base.save_denoised_latents(result, args.latents_path, args.num_frames)


def run_decode(args: argparse.Namespace) -> None:
    base.run_decode(args, workflow="ref2va")


#: Everything the stage-1 output depends on. Geometry belongs here: the conditioner is handed
#: explicit height/width/num_frames and both reference canvases, so a key that omits them can
#: replay embeddings built for a different canvas against a request that asked for another one.
_EMBEDDING_KEY_FIELDS = (
    "prompt", "references_json", "width", "height", "num_frames",
    "ref_image_short_edge", "ref_video_short_edge",
)


def _compute_embedding_hash(args: argparse.Namespace) -> str:
    import hashlib

    parts = [f"{name}={getattr(args, name, None)!r}" for name in _EMBEDDING_KEY_FIELDS]
    # The geometry environment overrides bypass the argument namespace entirely, so they have to be
    # keyed directly or an override would not invalidate a cached embedding.
    parts += [
        f"{name}={os.environ.get(name, '')}"
        for name in (
            "AIV_H3_REF_IMAGE_SHORT_EDGE",
            "AIV_H3_REF_VIDEO_SHORT_EDGE",
            "AIV_H3_REF_VIDEO_MAX_PIXELS",
        )
    ]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


def run_all(args: argparse.Namespace) -> None:
    import shutil
    # Reference embeddings depend on the conditioner, both VAEs, media bytes, resize policy and
    # Diffusers preprocessing code. The old home-directory cache did not version all of those and
    # could silently replay stale/broken conditioning after a model update. Correctness is the
    # default; developers may explicitly opt into a model-local cache while benchmarking.
    use_cache = os.environ.get("AIV_H3_CACHE_EMBEDS") == "1"
    cache_key = _compute_embedding_hash(args) if use_cache else None
    cache_file = (
        Path(args.repo_root) / "cache" / "ref-embeddings" / f"h3_embeds_{cache_key}.pt"
        if use_cache else None
    )
    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)

    embeds_handle, embeds_path = tempfile.mkstemp(suffix=".h3-ref-embeds.pt")
    latents_handle, latents_path = tempfile.mkstemp(suffix=".h3-ref-latents.pt")
    os.close(embeds_handle)
    os.close(latents_handle)
    try:
        common = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]
        if cache_file is not None and cache_file.is_file() and cache_file.stat().st_size > 1024:
            base.log(f"Ref2VA: found cached reference embeddings ({cache_key}) · skipping Stage 1 vision encoding!")
            shutil.copyfile(str(cache_file), embeds_path)
        else:
            command = [
                *common, "--stage", "encode", "--embeds-path", embeds_path,
                "--latents-path", latents_path,
            ]
            base.log("Ref2VA encode stage starting")
            completed = subprocess.run(command, check=False)
            if completed.returncode:
                base.log(f"the encode stage exited with code {completed.returncode}; its traceback is directly above")
                raise SystemExit(completed.returncode)
            try:
                if cache_file is not None:
                    shutil.copyfile(embeds_path, str(cache_file))
                else:
                    raise RuntimeError("embedding cache is disabled")
                base.log(f"Ref2VA: cached reference embeddings -> {cache_key}")
            except Exception:
                pass

        for stage in ("denoise", "decode"):
            command = [
                *common, "--stage", stage, "--embeds-path", embeds_path,
                "--latents-path", latents_path,
            ]
            base.log(f"Ref2VA {stage} stage starting")
            completed = subprocess.run(command, check=False)
            if completed.returncode:
                base.log(
                    f"the {stage} stage exited with code {completed.returncode}; "
                    "its traceback is directly above"
                )
                raise SystemExit(completed.returncode)
    finally:
        Path(embeds_path).unlink(missing_ok=True)
        Path(latents_path).unlink(missing_ok=True)


def main() -> int:
    args = parser().parse_args()
    # `--steps 0` means "the mode's own default", so a caller that does not care never has to know
    # that Turbo's count is fixed by the distillation and Base's is a choice.
    if not args.steps:
        args.steps = base.TURBO_NFE.get(args.sampling_mode, BASE_DEFAULT_STEPS)
    validate(args)
    if args.stage == "encode":
        run_encode(args)
    elif args.stage == "denoise":
        run_denoise(args)
    elif args.stage == "decode":
        run_decode(args)
    else:
        run_all(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
