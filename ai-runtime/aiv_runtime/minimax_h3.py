"""MiniMax-H3 joint video + audio generation, split across three processes for 16 GB / 64 GB.

Why three processes
-------------------
H3 conditions on Qwen3-VL-32B. Quantized to int8 that conditioner is about 25 GiB of host weights
and the INT8 ConvRot transformer is about 19.5 GiB; the VAEs add 5.8 GiB. Diffusers' own 12-16 GB recipe holds
all of it resident and asks for roughly 75 GB of host RAM, which this machine (63.7 GB) does not
have -- and the LTX work already proved that overcommitting here does not degrade, it dies at
`safe_open` with `OSError 1455` and takes a day to diagnose.

But the two big models never run at the same time. The conditioner emits `prompt_embeds` -- a
`(1, num_text_tokens, 5120)` tensor of a few megabytes -- and is then finished for the rest of the
job. So this adapter re-spawns itself:

    stage `encode`   loads the conditioner only, writes the embeds to a temp file, exits
    stage `denoise`  loads the transformer and keyframe VAE only, writes joint latents, exits
    stage `decode`   loads audio VAE then video VAE, one at a time, and writes the two outputs

Process exit is what makes this work. `del model; gc.collect()` returns pages to the allocator but
not reliably to Windows, and the commit charge is what the limit is measured against. Peak host RAM
becomes `max(conditioner, transformer)` rather than their sum, which is the difference between
fitting in 64 GB and not.

Outputs
-------
Video and audio are written as SEPARATE files -- an mp4 with no audio stream and a wav beside it --
because the editor places them on V1 and A1 as linked clips. Muxing them into one file and *also*
placing the wav would double the audio in every export.
"""

from __future__ import annotations

import argparse
import contextlib
import faulthandler
import gc
import json
import os
import subprocess
import sys
import tempfile
import time
import types
from collections.abc import Mapping
from pathlib import Path

import torch
import torch.nn as nn

faulthandler.enable()

# Same sys.path shim the LTX adapters carry: these run under a vendored virtualenv where
# `aiv_runtime` is not installed, and Python only puts the script's own directory on sys.path.
_PACKAGE_ROOT = str(Path(__file__).resolve().parent.parent)
if _PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, _PACKAGE_ROOT)

from aiv_runtime.h3_geometry import (  # noqa: E402
    FPS,
    LOW_VRAM_SHORT_EDGE,
    MIN_SHORT_EDGE,
    clamp_num_frames,
    resolve_canvas,
    validate_canvas,
    validate_num_frames,
)
from aiv_runtime.minimax_h3_first_block_cache import (  # noqa: E402
    DEFAULT_FIRST_BLOCK_CACHE_THRESHOLD,
    install_first_block_cache,
    validate_first_block_cache_threshold,
)
from aiv_runtime.minimax_h3_profiler import (  # noqa: E402
    MiniMaxH3DenoiseProfiler,
    profile_requested,
    resolve_profile_path,
)

#: Editor mode -> the diffusers workflow that serves it. `first-last` and `image` are the same
#: workflow with a different number of keyframes; H3 accepts a last frame on its own too, which is
#: how `last-only` works without a dedicated pipeline.
# The blocks the encode stage keeps. `text_encoder` is the conditioner itself; `before_encode`
# resolves the canvas and puts the keyframes onto it, which is what gives the conditioner a
# picture to look at. Everything after them belongs to the denoise process.
ENCODE_STAGE_BLOCKS = frozenset({"before_encode", "text_encoder"})

MODE_WORKFLOWS = {
    "text": "t2va",
    "image": "fl2va",
    "first-last": "fl2va",
    "last-only": "fl2va",
}

#: Offload profiles, in descending order of speed.
#:
#: The dials and what they actually cost, measured against diffusers' group-offloading hooks:
#:
#: * `blocks_per_group` is the main VRAM/speed lever and it is CHEAP. The transformer's resident
#:   working set is roughly `(blocks_per_group + 1) * per_block_size`, and an FP8 block is a few
#:   hundred MB -- so going from 1 to 4 costs about a gigabyte of VRAM and removes three quarters
#:   of the device synchronisations. Holding this at 1 is the single most common way to make a
#:   16 GB card look slow for no reason.
#: * `use_stream` overlaps the next group's host->device copy with the current group's compute,
#:   but Diffusers only supports it with one block per group. The four-block profile therefore
#:   trades copy overlap for fewer group transitions; balanced uses one streamed block.
#: * `record_stream` is close to free speed on top of streams, for slightly more memory.
#: * `low_cpu_mem_usage` pins tensors on the fly rather than pre-pinning the whole model. It
#:   roughly halves the host cost of streaming (which is otherwise about 2x the streamed model)
#:   at a modest speed cost. Only meaningful when `use_stream` is on.
#: * `vae_resident` would keep the 5.2 GiB video VAE on the GPU. That is a third of this card, held
#:   for the whole denoise to save time on a decode that happens ONCE. All 16 GB profiles offload
#:   it; the performance profile spends its headroom on four transformer blocks instead.
#:
#: Not included on purpose: `torch.compile`. The correctness baseline uses the published ConvRot
#: checkpoint and kernel exactly as released before layering on graph compilation.
OFFLOAD_PROFILES = {
    "resident": {
        "blocks_per_group": None, "use_stream": False, "record_stream": False,
        "low_cpu_mem_usage": True, "vae_resident": True, "is_resident": True,
    },
    "performance": {
        "blocks_per_group": 4, "use_stream": False, "record_stream": False,
        "low_cpu_mem_usage": True, "vae_resident": False,
    },
    "balanced": {
        "blocks_per_group": 1, "use_stream": True, "record_stream": True,
        "low_cpu_mem_usage": True, "vae_resident": False,
    },
    "minimal": {
        "blocks_per_group": 1, "use_stream": False, "record_stream": False,
        "low_cpu_mem_usage": True, "vae_resident": False,
    },
}
#: Accepted `--offload` spellings that are not profile names. `none` is the old word for
#: `resident`; `cpu` and `disk` are the LTX-shaped names an older frontend bundle still sends.
#:
#: These were two separate assignments to this same name, seventy lines apart, so the second
#: silently deleted the first and `offload_profile("none")` raised KeyError. Nothing hit it because
#: `effective_offload_profile` special-cases `none` before it ever looks the name up -- which is
#: exactly the kind of accident that hides until someone calls the other function.
OFFLOAD_ALIASES = {"none": "resident", "cpu": "balanced", "disk": "minimal"}

# Block-level offload misses the VAE's ungrouped Conv3D and decoder entry/output projections.
VIDEO_VAE_OFFLOAD_TYPE = "leaf_level"

VAE_TILE_SIZE = 256
VAE_TILE_OVERLAP = 64

#: What precision the video VAE runs at. FP32 is the default because it is what the released
#: pipeline does, not because the model requires it.
#:
#: `AutoencoderKLMiniMaxH3._keep_in_fp32_modules` names `encoder`, `decoder`, `quant_conv` and
#: `post_quant_conv` -- effectively the whole model -- and Diffusers enforces that at load time
#: whatever dtype is requested, so `dtype=torch.float16` on this component is silently a no-op.
#: That pin is a Diffusers-side conservative default: `.to(dtype)` only *warns*, and the released
#: shards are 703 FP32 tensors, 10.4 GB.
#:
#: The ecosystem does not agree. ComfyUI ships `minimax_h3_video_vae_fp16.safetensors` and selects
#: it in every official H3 template, so their entire userbase decodes this VAE in FP16 daily, and
#: the one published measurement of it (ComfyUI #15416) reports "fp16 / --bf16-vae / --fp32-vae
#: within 0.1 of each other" on a 0-255 scale, with no NaN or black-frame reports at FP16.
#:
#: So the choice is exposed rather than made. FP32 stays the default: matching the released
#: pipeline is worth more than 5 GB on the path that currently works. FP16 or BF16 halve the model
#: -- 10.4 GB to 5.2 GB, resident on the card during decode -- which is the lever to reach for if a
#: 768p decode does not fit, since 768x1344 is 2.6x the spatial area of the 544p profile. BF16 is
#: the safer of the two: it keeps FP32's exponent range, so it cannot overflow where FP16 would.
VAE_DTYPES = {"fp32": "float32", "bf16": "bfloat16", "fp16": "float16"}


def video_vae_dtype():
    """The dtype `AIV_H3_VAE_DTYPE` asks for, or `None` to leave the released FP32 alone."""
    name = (os.environ.get("AIV_H3_VAE_DTYPE") or "").strip().lower()
    if not name or name == "fp32":
        return None
    if name not in VAE_DTYPES:
        raise ValueError(
            f"AIV_H3_VAE_DTYPE must be one of {', '.join(VAE_DTYPES)}, got {name!r}"
        )
    return getattr(torch, VAE_DTYPES[name])


def cast_video_vae(vae):
    """Apply `AIV_H3_VAE_DTYPE` to a freshly loaded video VAE, before it is placed anywhere.

    Clearing `_keep_in_fp32_modules` first is deliberate and is the whole point: leaving it set
    makes `.to()` log a warning about inconsistent results and casts anyway, which is the worst of
    both. Cleared, the cast is total and `get_parameter_dtype` then reports the real dtype -- so
    `AutoencoderKLMiniMaxH3.encode`, which casts incoming pixels to the encoder's own dtype, stays
    self-consistent instead of handing FP32 pixels to FP16 weights.

    Must run before `.to("cuda")` or `apply_group_offloading`: a dtype cast on a group-offloaded
    module is not what the offload hooks expect.
    """
    requested = video_vae_dtype()
    current = next(vae.parameters()).dtype
    if requested is None or requested == current:
        return current
    vae._keep_in_fp32_modules = None
    vae.to(requested)
    log(
        f"video VAE: cast {current} -> {requested} on AIV_H3_VAE_DTYPE. This overrides Diffusers' "
        f"`_keep_in_fp32_modules` pin; ComfyUI ships this VAE in fp16 and measures the difference "
        f"at about 0.1 of 255, but the released pipeline runs it in fp32."
    )
    return requested
H3_PIXEL_MEAN = (0.485, 0.456, 0.406)
H3_PIXEL_STD = (0.229, 0.224, 0.225)

TURBO_NFE = {"turbo-4": 4, "turbo-8": 8}

# The scheduler shifts belong to the *adapter*, not to Turbo in general. LightX2V publish
# video 12 / audio 3 for the Ref2VA 4-step v0.1 and for the 544p FL2V 8-step v1.0, but video
# *6* for the 768p FL2V 4-step v1.0 -- so a module constant shared by both code paths is a trap
# waiting for the day someone adopts that file. Resolve per adapter, keyed on the filename.
TURBO_VIDEO_SHIFT = 12.0
TURBO_AUDIO_SHIFT = 3.0
#: (filename fragments that must all be present) -> (video shift, audio shift)
TURBO_SHIFT_OVERRIDES: tuple[tuple[tuple[str, ...], tuple[float, float]], ...] = (
    (("fl2v", "768p"), (6.0, 3.0)),
)

# PEFT applies a LoRA at alpha/r. LightX2V ship bare safetensors with no adapter_config.json, so
# alpha was assumed to have to come from outside the file. It does not, for most of them:
#
#   2026-08-25, read from the installed files' safetensors `__metadata__`:
#     minimax_h3_ref2v_turbo_4step_v0.1_bf16   alpha '8'   rank 128
#     minimax_h3_fl2v_turbo_8step_v1.0_bf16    alpha '8'   rank 128
#     minimax_h3_fl2v_turbo_4step_v0.1         (no alpha)  rank 128
#
# So the two adapters that matter state their own alpha, and it agrees with the published 8.
# `turbo_lora_alpha` prefers that declaration and keeps the table below only as the fallback for
# files like the third one, which declare nothing. This matters for the 768p FL2V v1.0 adapter the
# filename table guesses 128 for: if that file declares an alpha, the file wins over the guess.
#
# The published value is corroborated by MiniMax's own MiniMax-H3-Turbo-Lora Space
# (h3_lora.py, commit 456aa27), which carries:
#
#     LIGHTX_ALPHA = 8        # rank 128, alpha 8 -> fold scale 0.0625
#
# together with the note that the resulting 0.0625 fold is exactly what LightX2V's own inference
# script produces at `weights=1.0`. So 8 is the number, and it is an ABSOLUTE alpha -- not a
# stand-in for the rank.
#
# 2026-08-18: this was briefly changed to `alpha = rank` on the theory that a distillation adapter
# must be applied at unity because "it IS the sampler". That reasoning is right for every adapter
# trained under the alpha == rank convention -- ai-toolkit, fal's Ref2VA trainer, larryvrh's FL2VA
# turbo -- and wrong for exactly this family. On the rank-128 Ref2VA adapter it applied the LoRA at
# 16x and the output came back a garbled mess. Do not restore `rank` without a published alpha for
# the adapter in hand; the ecosystem does not agree on one (Kijai's ComfyUI repack guesses 16;
# drbaph's SVD resizes bake alpha/rank into lora_B and strip the alpha entirely).
#
# AIV_TURBO_LORA_ALPHA overrides it for experimentation, and `turbo_lora_alpha` is where that is
# read -- every path resolves its alpha through that function, including ConvRot.
TURBO_LORA_ALPHA: int = 8
TURBO_LORA_TARGETS = (
    "to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2",
)
LORA_A_SUFFIX = ".lora_A.default.weight"
LORA_B_SUFFIX = ".lora_B.default.weight"



def offload_profile(name: str) -> dict:
    return OFFLOAD_PROFILES[OFFLOAD_ALIASES.get(name, name)]


def effective_offload_profile(name: str, fp8_mode: str | None = None) -> dict:
    """Resolve an offload profile with safe streamed prefetching and memory pinning."""
    if name in ("resident", "none"):
        return dict(OFFLOAD_PROFILES["resident"])
    profile = dict(offload_profile(name))
    if fp8_mode in ("weight-only", "int4-resident", "portable-fp8") or (fp8_mode or "").startswith("gemlite-"):
        profile["use_stream"] = False
        profile["record_stream"] = False
    return profile


MODE_REQUIREMENTS = {
    "text": (),
    "image": ("first_frame",),
    "first-last": ("first_frame", "last_frame"),
    "last-only": ("last_frame",),
}


def log(message: str) -> None:
    """Progress goes to stderr; stdout carries the single JSON result line the Rust side parses."""
    print(f"[minimax-h3] {message}", file=sys.stderr, flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MiniMax-H3 joint video and audio generation.")
    parser.add_argument("--repo-root", required=True, help="models/MiniMax-H3, the diffusers layout")
    parser.add_argument(
        "--transformer-path",
        default=None,
        help="Published MiniMax-H3 INT8 ConvRot transformer checkpoint. The original BF16 "
        "partition does not fit this machine's 16 GiB runtime profile.",
    )
    parser.add_argument(
        "--text-encoder-path",
        default=None,
        help="Published Qwen3-VL INT8 ConvRot .safetensors conditioner. PyTorch .pt/.bin "
        "conditioner checkpoints are intentionally unsupported.",
    )
    parser.add_argument("--mode", choices=tuple(MODE_WORKFLOWS), default="text")
    parser.add_argument("--first-frame", default=None)
    parser.add_argument("--last-frame", default=None)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output-path", required=True, help="mp4, written WITHOUT an audio stream")
    parser.add_argument("--audio-path", default=None, help="wav; defaults to the mp4 stem + .wav")
    parser.add_argument("--width", type=int, default=0, help="0 resolves from the aspect ratio")
    parser.add_argument("--height", type=int, default=0)
    parser.add_argument(
        "--short-edge",
        type=int,
        default=LOW_VRAM_SHORT_EDGE,
        help="Only used when width/height are 0. The trained value is 768; the default here is "
        f"{LOW_VRAM_SHORT_EDGE}, which is what a 16 GB card finishes in a sane time.",
    )
    parser.add_argument("--aspect", default="16:9", help="Only used when width/height are 0")
    parser.add_argument("--num-frames", type=int, default=124)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument(
        "--sampling-mode",
        choices=("base", "turbo-4", "turbo-8"),
        default="turbo-8",
        help="Base H3 sampling, or a dedicated LightX2V 4/8-NFE Turbo adapter and schedule.",
    )
    parser.add_argument(
        "--turbo-lora-path",
        default=None,
        help="LightX2V FL2VA Diffusers LoRA matching the selected Turbo sampling mode.",
    )
    parser.add_argument(
        "--attention-backend",
        choices=("native", "sage", "sage2_patched", "sage1", "sage2_stock"),
        default="sage",
        help="H3 transformer attention engine: native, sage (patched 2.0), sage1, or sage2_stock. "
        "Sage falls back to native per call when a call does not meet its constraints.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--offload",
        choices=tuple(OFFLOAD_PROFILES) + tuple(OFFLOAD_ALIASES),
        default="balanced",
        help="performance moves four transformer blocks per group; balanced moves one group "
        "with copy/compute overlap; minimal moves one group without streaming.",
    )
    parser.add_argument(
        "--vae-tile",
        type=int,
        default=VAE_TILE_SIZE,
        help=f"Video VAE spatial tile size. MiniMax-H3's reference value is {VAE_TILE_SIZE} "
        f"pixels with {VAE_TILE_OVERLAP} pixels of overlap.",
    )
    parser.add_argument(
        "--first-block-cache",
        action="store_true",
        help="Experimental step skipping. Default-off until the Ref2VA quality baseline is approved.",
    )
    parser.add_argument(
        "--first-block-cache-threshold",
        type=float,
        default=DEFAULT_FIRST_BLOCK_CACHE_THRESHOLD,
        help="Relative block-0 residual change allowed for a skip (0.06-0.14; default 0.08).",
    )
    parser.add_argument(
        "--profile-denoise",
        action="store_true",
        help="Write opt-in per-step H3 timing, memory, GPU, and offload diagnostics as JSONL.",
    )
    parser.add_argument(
        "--profile-path",
        default=None,
        help="Diagnostic JSONL path; defaults beside the output MP4. Also enables profiling.",
    )
    # Internal. The Rust side never passes these; the adapter sets them on itself when it re-spawns.
    parser.add_argument("--stage", choices=("all", "encode", "denoise", "decode"), default="all")
    parser.add_argument("--embeds-path", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--latents-path", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def resolve_geometry(args: argparse.Namespace) -> tuple[int, int, int]:
    """Settle width, height and frame count before any weight is touched."""
    if args.width and args.height:
        width, height = args.width, args.height
    else:
        try:
            left, right = (float(part) for part in args.aspect.replace("x", ":").split(":", 1))
        except ValueError as error:
            raise ValueError(f"could not read --aspect {args.aspect!r} as w:h") from error
        short_edge = max(MIN_SHORT_EDGE, args.short_edge)
        width, height = resolve_canvas(left, right, short_edge=short_edge)
    validate_canvas(width, height)
    num_frames = clamp_num_frames(args.num_frames)
    validate_num_frames(num_frames)
    return width, height, num_frames


def validate(args: argparse.Namespace) -> None:
    root = Path(args.repo_root)
    validate_first_block_cache_threshold(
        getattr(args, "first_block_cache_threshold", DEFAULT_FIRST_BLOCK_CACHE_THRESHOLD)
    )
    if not root.is_dir():
        raise FileNotFoundError(f"MiniMax-H3 pack not found: {root}")
    if not (root / "modular_model_index.json").is_file() and not (root / "model_index.json").is_file():
        raise FileNotFoundError(
            f"{root} does not look like a MiniMax-H3 diffusers pack (no model index). Install it "
            "with: .venv\\Scripts\\python.exe scripts\\models.py install minimax-h3 --accept-licenses"
        )
    for attribute in MODE_REQUIREMENTS[args.mode]:
        value = getattr(args, attribute)
        if not value:
            raise ValueError(f"mode {args.mode!r} requires --{attribute.replace('_', '-')}")
        if not Path(value).exists():
            raise FileNotFoundError(value)
    if not args.prompt.strip():
        raise ValueError("prompt cannot be empty")
    if args.transformer_path and Path(args.transformer_path).suffix.lower() != ".safetensors":
        raise ValueError("MiniMax-H3 denoisers must be safetensors files; .pt checkpoints are unsupported")
    if args.text_encoder_path:
        conditioner = Path(args.text_encoder_path)
        if conditioner.is_file() and conditioner.suffix.lower() != ".safetensors":
            raise ValueError(
                "MiniMax-H3 conditioner weights must be safetensors files; .pt/.bin checkpoints are unsupported"
            )
        if conditioner.is_dir() and not (conditioner / "model.safetensors.index.json").is_file():
            raise ValueError(
                "MiniMax-H3 conditioner directories must contain model.safetensors.index.json"
            )
    if args.steps < 1 or args.steps > 60:
        raise ValueError("steps must be between 1 and 60")
    if args.vae_tile < VAE_TILE_SIZE or args.vae_tile % 16:
        raise ValueError(
            f"vae tile must be at least {VAE_TILE_SIZE} pixels and divisible by 16, got {args.vae_tile}"
        )
    if args.stage in ("denoise", "decode") and not args.latents_path:
        raise ValueError(f"the internal {args.stage} stage requires --latents-path")
    if args.stage == "denoise" and not args.embeds_path:
        raise ValueError("the internal denoise stage requires --embeds-path")
    if args.sampling_mode in TURBO_NFE:
        expected = TURBO_NFE[args.sampling_mode]
        if args.steps != expected:
            raise ValueError(
                f"{args.sampling_mode} requires exactly {expected} transformer evaluations"
            )
        if not args.turbo_lora_path or not Path(args.turbo_lora_path).is_file():
            raise FileNotFoundError(
                f"{args.sampling_mode} requires its matching LightX2V LoRA; "
                "--turbo-lora-path was not found"
            )
    if Path(args.output_path).suffix.lower() != ".mp4":
        raise ValueError("output must be an .mp4 file")


def audio_path_for(args: argparse.Namespace) -> Path:
    return Path(args.audio_path) if args.audio_path else Path(args.output_path).with_suffix(".wav")


def scheduler_grid_points(args: argparse.Namespace) -> int:
    """Translate the UI's requested work into MiniMaxH3Scheduler grid points.

    The scheduler includes terminal zero in ``num_inference_steps`` and therefore executes one
    fewer transformer evaluation. The application reports transformer evaluations (NFE) for both
    Base and Turbo, so every request needs one terminal grid point in addition to its NFE count.
    """
    return args.steps + 1


def sigma_grid_length(shift: float, points: int) -> int:
    """How many sigmas `MiniMaxH3Scheduler.set_timesteps` will actually realize at this shift.

    Mirrors its construction exactly -- `linspace(1, 0, points)` in float32, through the
    exponential shift, then `unique_consecutive` -- because the collapse this exists to catch IS a
    float32 rounding effect and computing it in float64 would never see it.
    """
    base = torch.linspace(1.0, 0.0, int(points), dtype=torch.float32)
    sigmas = shift * base / (1 + (shift - 1) * base)
    return int(torch.unique_consecutive(sigmas).numel())


def assert_schedules_agree(video_shift: float, audio_shift: float, points: int) -> None:
    """Refuse a shift pair that would give the two tracks different lengths.

    The shift compresses the grid near sigma = 1, and `unique_consecutive` drops the float32
    duplicates that compression creates. Collapse costs an evaluation the adapter was distilled
    for, and -- worse -- if it hits one track and not the other, the audio and the video denoise
    over different numbers of steps and come out desynchronised, silently. Upstream raises here.

    Measured on 2026-09-08: the shipped shifts are nowhere near it. At 51 grid points the first
    collapse is above shift ~212,000; at 9 points, ~2.4 million. Our published pairs are 12/3 and
    6/3. So this guards a typo in `AIV_TURBO_VIDEO_SHIFT` -- a stray zero, or `12e5` for `12` --
    rather than a plausible tuning value, and it costs two `linspace` calls to do it before 20 GB
    of weights are loaded rather than after.
    """
    video = sigma_grid_length(video_shift, points)
    audio = sigma_grid_length(audio_shift, points)
    if video != audio:
        raise ValueError(
            f"video shift {video_shift:g} and audio shift {audio_shift:g} realize different "
            f"schedules at {points} grid points ({video} sigmas against {audio}), so the two "
            "tracks would denoise over different numbers of steps and come out desynchronised. "
            "The shift compresses the grid near sigma = 1 until float32 cannot separate "
            "consecutive points; lower the larger shift."
        )
    if video != points:
        log(
            f"WARNING: the shift collapsed the sigma grid from {points} points to {video}, so "
            f"this run makes {video - 1} transformer evaluations rather than {points - 1}"
        )


def instrument_scheduler_grid(pipeline) -> None:
    """Log the schedule each scheduler actually realizes, once per run.

    `scheduler_grid_points` asserts that ``steps + 1`` grid points buy ``steps`` transformer
    evaluations, and nothing used to check it. Read against the installed scheduler on 2026-09-08
    and CONFIRMED: ``MiniMaxH3Scheduler.set_timesteps`` builds ``linspace(1, 0, n)``, pushes it
    through the exponential shift and publishes ``timesteps = 1 - sigmas[:-1]``, so an ``n``-point
    grid drives ``n - 1`` evaluations. Our convention is right and a turbo-4 run is 4 NFE.

    The identity has exactly one escape hatch: ``torch.unique_consecutive`` collapses the float32
    duplicates the shift creates near sigma = 1. A collapse costs an evaluation the adapter was
    distilled for and, worse, can leave the video and audio tracks at different lengths --
    silently desynchronising them. That is what this reports, and it is the measurement the
    sigma-collapse guard is built on. Instrumentation only: it changes no schedule.
    """
    realized: dict[str, int] = {}
    for name in ("scheduler", "audio_scheduler"):
        scheduler = getattr(pipeline, name, None)
        if scheduler is None or getattr(scheduler, "_aiv_grid_logged", False):
            continue
        original = scheduler.set_timesteps

        def logged(*call_args, _scheduler=scheduler, _original=original, _name=name, **kwargs):
            result = _original(*call_args, **kwargs)
            requested = kwargs.get(
                "num_inference_steps", call_args[0] if call_args else None
            )
            realized[_name] = len(_scheduler.timesteps)
            log(
                f"{_name} grid \u00b7 requested {requested} points \u00b7 "
                f"realized {len(_scheduler.sigmas)} sigmas \u00b7 "
                f"{len(_scheduler.timesteps)} transformer evaluations"
            )
            # The authoritative version of `assert_schedules_agree`: that one predicts from the
            # shifts before loading, this one reads what the schedulers actually built. Both
            # exist because the cheap one cannot see a Base run's released default shift.
            if len(realized) == 2 and len(set(realized.values())) > 1:
                raise RuntimeError(
                    "the video and audio schedules realized different lengths "
                    f"({realized}), so the two tracks would denoise over different numbers of "
                    "steps and come out desynchronised"
                )
            return result

        scheduler.set_timesteps = logged
        scheduler._aiv_grid_logged = True


def _keyframes(args: argparse.Namespace, width: int, height: int) -> dict:
    """The keyframe kwargs for the chosen mode, already opened as PIL images.

    `image` is the frame the video starts from and `last_image` the one it ends on. H3 stretches
    the first onto the canvas and cover-crops the follower, so the canvas is resolved from the
    FIRST frame's aspect ratio when one was given -- passing a portrait keyframe with a landscape
    canvas is a request to distort it, not an error.
    """
    from PIL import Image

    kwargs: dict = {}
    if args.mode in ("image", "first-last") and args.first_frame:
        kwargs["image"] = Image.open(args.first_frame).convert("RGB")
    if args.mode in ("first-last", "last-only") and args.last_frame:
        kwargs["last_image"] = Image.open(args.last_frame).convert("RGB")
    return kwargs


def load_text_encoder(args: argparse.Namespace):
    """Load only a safetensors conditioner, preferring Comfy-Org's published ConvRot file."""
    requested = Path(args.text_encoder_path) if args.text_encoder_path else None
    if requested is not None and requested.is_dir():
        return load_baked_text_encoder(requested)

    checkpoint = requested or (
        Path(args.repo_root)
        / "text_encoders"
        / "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"published INT8 ConvRot conditioner not found: {checkpoint}\n"
            "Install the MiniMax-H3 model pack; legacy .pt/.bin conditioners are not used."
        )
    if checkpoint.suffix.lower() != ".safetensors":
        raise ValueError("MiniMax-H3 conditioner weights must be a .safetensors file")

    config_path = Path(args.repo_root) / "text_encoder"
    if not (config_path / "config.json").is_file():
        raise FileNotFoundError(f"official Qwen3-VL conditioner config not found: {config_path}")
    from aiv_runtime.minimax_h3_convrot_text_encoder import (
        load_text_encoder as load_convrot_conditioner,
    )

    model = load_convrot_conditioner(config_path=config_path, checkpoint_path=checkpoint)
    report = model._aiv_convrot_report
    log(
        "conditioner: official/community INT8 ConvRot safetensor · "
        f"{report['int8_projections']} projections · "
        f"{report['real_decoder_layers']} released decoder layers"
    )
    return model


def _checkpoint_shards(index_path: Path) -> tuple[list[str], set[str]]:
    """Return index-order shard names and the complete parameter-key contract."""
    document = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = document.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"invalid or empty weight_map in {index_path}")
    shards = list(dict.fromkeys(weight_map.values()))
    if not all(isinstance(name, str) and name for name in shards):
        raise ValueError(f"invalid shard name in {index_path}")
    return shards, set(weight_map)


def load_baked_text_encoder(path: Path):
    """Load a legacy experimental conditioner only when it is safetensors-sharded."""
    import torch.nn as nn
    from accelerate import init_empty_weights
    from transformers import AutoConfig, Qwen3VLForConditionalGeneration

    index_path = path / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(
            f"safetensors conditioner index not found: {index_path}; .pt/.bin weights are unsupported"
        )
    shard_names, indexed_keys = _checkpoint_shards(index_path)

    config = AutoConfig.from_pretrained(path)
    with init_empty_weights():
        model = Qwen3VLForConditionalGeneration(config)

    is_int4 = any(k.endswith(".qweight") for k in indexed_keys)
    if is_int4:
        replaced = 0
        for name, module in list(model.named_modules()):
            if isinstance(module, nn.Linear):
                qw_key = f"{name}.weight.qweight"
                if qw_key in indexed_keys:
                    parent_name, _, child_name = name.rpartition(".")
                    parent = model.get_submodule(parent_name) if parent_name else model
                    has_bias = f"{name}.bias" in indexed_keys
                    int4_mod = FastInt4Linear(module.in_features, module.out_features, group_size=128, bias=has_bias)
                    setattr(parent, child_name, int4_mod)
                    replaced += 1
        log(f"conditioner: installed {replaced} FastInt4Linear resident layers in Qwen3-VL")

    log(f"conditioner: baked {'int4' if is_int4 else 'int8'} from {path} · {len(shard_names)} direct-assignment shards")
    loaded: set[str] = set()
    from aiv_runtime.safetensors_reader import describe_backend, load_state_dict

    log(f"conditioner read: {describe_backend()}")
    for position, shard_name in enumerate(shard_names, start=1):
        shard_path = path / shard_name
        if not shard_path.is_file():
            raise FileNotFoundError(f"baked conditioner shard not found: {shard_path}")
        sd = load_state_dict(shard_path, device="cpu")
        state_dict = {}
        for k, v in sd.items():
            if k.endswith(".weight.qweight"):
                state_dict[k.replace(".weight.qweight", ".weight_qweight")] = v
            elif k.endswith(".weight.scales"):
                state_dict[k.replace(".weight.scales", ".weight_scales")] = v
            elif k.endswith(".weight.group_size"):
                state_dict[k.replace(".weight.group_size", ".weight_group_size")] = v
            else:
                state_dict[k] = v

        if not isinstance(state_dict, Mapping):
            raise TypeError(f"expected a state dict in {shard_path}, got {type(state_dict).__name__}")
        shard_keys = set(state_dict)
        incompatible = model.load_state_dict(state_dict, strict=False, assign=True)
        if incompatible.unexpected_keys:
            raise RuntimeError(
                f"baked conditioner shard {shard_name} has unexpected key "
                f"{incompatible.unexpected_keys[0]}"
            )
        loaded.update(shard_keys)
        log(
            f"conditioner: assigned shard {position}/{len(shard_names)} · "
            f"{shard_path.stat().st_size / 1024**3:.2f} GiB"
        )
        del state_dict
        gc.collect()

    model.requires_grad_(False)
    model.eval()
    log(f"conditioner: assigned all {len(loaded)} tensors without the Transformers shard loader")
    return model


def ensure_mm_token_type_ids(processor) -> bool:
    """Backport the Transformers 5 multimodal-token helper required by H3 Diffusers.

    The H3 environment stays on Transformers 4.57.6 because its v5 checkpoint loader could not
    bake Qwen3-VL-32B within the Windows commit limit. Diffusers' H3 branch only needs this small
    processor API from v5; adding it to the loaded processor preserves the known-good model code.
    Returns ``True`` when the compatibility method was installed.
    """
    if hasattr(processor, "create_mm_token_type_ids"):
        return False

    def token_ids(plural: str, singular: str) -> set[int]:
        values = getattr(processor, plural, None)
        if values is None:
            values = [getattr(processor, singular, None)]
        return {int(value) for value in values if value is not None}

    image_ids = token_ids("image_token_ids", "image_token_id")
    video_ids = token_ids("video_token_ids", "video_token_id")
    audio_ids = token_ids("audio_token_ids", "audio_token_id")

    def create_mm_token_type_ids(input_ids) -> list[list[int]]:
        rows: list[list[int]] = []
        for source in input_ids:
            values = source if isinstance(source, list) else source.tolist()
            rows.append([
                1 if value in image_ids else 2 if value in video_ids else 3 if value in audio_ids else 0
                for value in values
            ])
        return rows

    processor.create_mm_token_type_ids = create_mm_token_type_ids
    return True


def install_qwen3vl_pos_embed_device_shim(model, onload_device) -> bool:
    """Keep Qwen3-VL's interpolation tensors with its leaf-offloaded embedding.

    Transformers 4.57 asks ``pos_embed.weight.device`` before calling the embedding. Under
    Diffusers leaf offload that weight is still on CPU at that moment; the embedding hook then
    onloads the parameter and returns CUDA output, which immediately gets multiplied by CPU
    interpolation weights. This scoped replacement is the upstream method with only the final
    index/weight tensor device made explicit.
    """
    visual = getattr(model, "visual", None)
    if visual is None or not hasattr(visual, "fast_pos_embed_interpolate"):
        return False
    if getattr(visual, "_aiv_pos_embed_device_shim", False):
        return False
    target = torch.device(onload_device)

    def fast_pos_embed_interpolate(self, grid_thw):
        grid_ts, grid_hs, grid_ws = grid_thw[:, 0], grid_thw[:, 1], grid_thw[:, 2]
        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]
        for _t, h, w in zip(grid_ts, grid_hs, grid_ws):
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h)
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w)
            h_idxs_floor = h_idxs.int()
            w_idxs_floor = w_idxs.int()
            h_idxs_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            w_idxs_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            dh = h_idxs - h_idxs_floor
            dw = w_idxs - w_idxs_floor
            base_h = h_idxs_floor * self.num_grid_per_side
            base_h_ceil = h_idxs_ceil * self.num_grid_per_side
            indices = [
                (base_h[None].T + w_idxs_floor[None]).flatten(),
                (base_h[None].T + w_idxs_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
            ]
            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]
            for index in range(4):
                idx_list[index].extend(indices[index].tolist())
                weight_list[index].extend(weights[index].tolist())
        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=target)
        weight_tensor = torch.tensor(
            weight_list, dtype=self.pos_embed.weight.dtype, device=target
        )
        pos_embeds = self.pos_embed(idx_tensor) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]
        patch_pos_embeds = patch_pos_embeds.split(
            [int(h * w) for h, w in zip(grid_hs, grid_ws)]
        )
        patch_pos_embeds_permute = []
        merge_size = self.config.spatial_merge_size
        for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
            t, h, w = int(t), int(h), int(w)
            pos_embed = pos_embed.repeat(t, 1)
            pos_embed = (
                pos_embed.view(
                    t, h // merge_size, merge_size, w // merge_size, merge_size, -1
                )
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            patch_pos_embeds_permute.append(pos_embed)
        return torch.cat(patch_pos_embeds_permute)

    visual.fast_pos_embed_interpolate = types.MethodType(fast_pos_embed_interpolate, visual)
    visual._aiv_pos_embed_device_shim = True
    return True


def assert_conditioner_saw_the_keyframes(pipeline, tags, keyframes: dict) -> None:
    """Prove the presentation carries a vision block, rather than trusting that it does.

    `embeds is None or tags is None` was the only guard here, and an all-TEXT tag vector passes it
    happily -- which is exactly what a keyframe request produced while `before_encode` was being
    popped. The tag histogram is the cheapest possible proof: MiniMax-H3 tags a vision block's
    rows as VIDEO, so a keyframe request whose tags are entirely TEXT means the conditioner was
    handed a prompt and nothing else.

    Reported on every request, asserted only on the ones that carry a frame.
    """
    # The released tag values, taken from the pipeline when it exposes them. Defaults are
    # `MINIMAX_H3_TEXT_TAG` / `MINIMAX_H3_VIDEO_TAG`, so a Diffusers build that moves the
    # properties degrades this to a log line rather than taking a working generation down.
    text_tag = int(getattr(pipeline, "text_tag", 1))
    video_tag = int(getattr(pipeline, "video_tag", 0))
    histogram = {int(tag): int(count) for tag, count in zip(*tags.unique(return_counts=True))}
    log(
        f"conditioner presentation \u00b7 {tags.numel()} rows \u00b7 "
        f"{histogram.get(text_tag, 0)} text \u00b7 {histogram.get(video_tag, 0)} video \u00b7 "
        f"{len(keyframes)} keyframe(s) sent"
    )
    if keyframes and not histogram.get(video_tag, 0):
        raise RuntimeError(
            f"the conditioner was sent {len(keyframes)} keyframe(s) and produced a text-only "
            "presentation: no row is tagged VIDEO, so no vision block was built and the prompt "
            "embedding carries no visual grounding. The `before_encode` block that turns "
            "`image`/`last_image` into `keyframes` is missing from the encode stage."
        )


def run_encode(args: argparse.Namespace) -> None:
    """Stage 1. Load the conditioner alone, emit `prompt_embeds`, and exit so the RAM comes back."""
    import torch
    from diffusers import ModularPipeline
    from diffusers.hooks import apply_group_offloading

    width, height, num_frames = resolve_geometry(args)
    workflow = MODE_WORKFLOWS[args.mode]
    log(f"stage encode · workflow {workflow} · {width}x{height} · {num_frames} frames")

    blocks = ModularPipeline.from_pretrained(args.repo_root).blocks.get_workflow(workflow)
    # Everything after the conditioner belongs to the other process. Modular block names are
    # flattened (`denoise.denoise`, `decode.video`, ...), so popping the bare group names does
    # nothing. Retain the exact blocks we need instead.
    #
    # `before_encode` is one of them, and dropping it was a silent quality bug for every keyframe
    # request. It holds `MiniMaxH3ResizeStep`, which is what turns `image` / `last_image` into
    # `keyframes`. `MiniMaxH3FL2VATextEncoderStep` is still selected without it -- the auto block
    # triggers on `image` / `last_image`, not on `keyframes` -- but it guards its whole vision
    # path on `if block_state.keyframes:`, so with the resize step gone it silently built a
    # text-only presentation: no vision features, every row tagged TEXT, no visual grounding in
    # the embedding at all. The picture still anchored the video rows, because the keyframe VAE
    # runs in the denoise process where `before_encode` is retained, which is exactly why this
    # never showed up as a crash. `minimax_h3_ref.py` has always kept the block; this is the base
    # worker catching up. The step's only component is a `VaeImageProcessor` built `from_config`,
    # so it loads no weights.
    for name in tuple(blocks.sub_blocks):
        if name not in ENCODE_STAGE_BLOCKS:
            blocks.sub_blocks.pop(name)
    pipeline = blocks.init_pipeline(args.repo_root)
    pipeline.update_components(text_encoder=load_text_encoder(args))
    # Component specs retain the original Hub repo even in a local snapshot. Override it here or
    # Diffusers re-downloads the pruned 62 GiB bf16 encoder while looking for unrelated pieces.
    pipeline.load_components(
        names=["tokenizer", "processor"],
        pretrained_model_name_or_path=args.repo_root,
        dtype=torch.bfloat16,
    )
    if ensure_mm_token_type_ids(pipeline.processor):
        log("conditioner: installed Transformers 4.57 multimodal-token compatibility shim")

    pipeline.text_encoder.requires_grad_(False)
    is_int4 = any(isinstance(m, FastInt4Linear) for m in pipeline.text_encoder.model.modules())
    if is_int4:
        log("conditioner: 4-bit INT4 resident mode enabled · zero PCIe offloading")
        pipeline.text_encoder.model.to("cuda")
    else:
        apply_group_offloading(
            pipeline.text_encoder.model,
            offload_type="leaf_level",
            onload_device=torch.device("cuda"),
            offload_device=torch.device("cpu"),
        )
        if install_qwen3vl_pos_embed_device_shim(pipeline.text_encoder.model, "cuda"):
            log("conditioner: aligned Qwen3-VL positional interpolation with CUDA leaf onload")

    keyframes = _keyframes(args, width, height)
    state = pipeline(
        prompt=args.prompt,
        height=height,
        width=width,
        num_frames=num_frames,
        **keyframes,
    )
    embeds = state.get("prompt_embeds")
    tags = state.get("text_token_tags")
    if embeds is None or tags is None:
        raise RuntimeError(
            "the conditioner produced no prompt_embeds; the text_encoder block did not run"
        )
    assert_conditioner_saw_the_keyframes(pipeline, tags, keyframes)
    torch.save(
        {"prompt_embeds": embeds.to("cpu"), "text_token_tags": tags.to("cpu")}, args.embeds_path
    )
    log(f"stage encode · {tuple(embeds.shape)} embeds -> {args.embeds_path}")


class FastInt4Linear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, group_size: int = 128, bias: bool = False):
        super().__init__(in_features, out_features, bias=bias, device="meta")
        self.group_size = group_size
        self.register_buffer("weight_qweight", torch.empty((out_features, in_features // 2), dtype=torch.uint8))
        self.register_buffer("weight_scales", torch.empty((out_features, in_features // group_size), dtype=torch.bfloat16))
        self.register_buffer("weight_group_size", torch.tensor(group_size, dtype=torch.int32))
        self.weight = nn.Parameter(torch.empty(0, dtype=torch.bfloat16), requires_grad=False)
        self._unpacked = False

    def unpack_weights(self):
        if not self._unpacked and self.weight_qweight.numel() > 0:
            q = self.weight_qweight
            dtype = self.weight_scales.dtype
            low = (q & 0x0F).to(dtype) - 8.0
            high = ((q >> 4) & 0x0F).to(dtype) - 8.0
            w = torch.stack([low, high], dim=-1).view(self.out_features, self.in_features // self.group_size, self.group_size)
            w = (w * self.weight_scales.unsqueeze(-1)).view(self.out_features, self.in_features)
            self.weight = nn.Parameter(w, requires_grad=False)
            self.weight_qweight = torch.empty(0, dtype=torch.uint8, device=self.weight_qweight.device)
            self.weight_scales = torch.empty(0, dtype=dtype, device=self.weight_scales.device)
            self._unpacked = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._unpacked:
            self.unpack_weights()
        return torch.nn.functional.linear(x, self.weight, self.bias)


class Int8Linear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, group_size: int = 128, bias: bool = False):
        super().__init__(in_features, out_features, bias=bias, device="meta")
        self.group_size = group_size
        self.register_buffer("weight_qweight", torch.empty((out_features, in_features), dtype=torch.int8))
        self.register_buffer("weight_scales", torch.empty((out_features, in_features // group_size), dtype=torch.bfloat16))
        self.register_buffer("weight_group_size", torch.tensor(group_size, dtype=torch.int32))
        self.register_buffer("weight_is_int8", torch.tensor(1, dtype=torch.int32))
        self.weight = nn.Parameter(torch.empty(0, dtype=torch.bfloat16), requires_grad=False)
        self._unpacked = False

    def unpack_weights(self):
        if not self._unpacked and self.weight_qweight.numel() > 0:
            dtype = self.weight_scales.dtype
            w = self.weight_qweight.view(self.out_features, self.in_features // self.group_size, self.group_size).to(dtype)
            w = (w * self.weight_scales.unsqueeze(-1)).view(self.out_features, self.in_features)
            self.weight = nn.Parameter(w, requires_grad=False)
            self.weight_qweight = torch.empty(0, dtype=torch.int8, device=self.weight_qweight.device)
            self.weight_scales = torch.empty(0, dtype=dtype, device=self.weight_scales.device)
            self._unpacked = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._unpacked:
            self.unpack_weights()
        return torch.nn.functional.linear(x, self.weight, self.bias)


def _prepare_int4_transformer(model, state_dict: Mapping) -> tuple[int, dict]:
    """Replace linear layers with FastInt4Linear / Int8Linear and align state_dict keys."""
    count_int4 = 0
    count_int8 = 0
    remapped_sd = {}
    for k, v in state_dict.items():
        if k.endswith(".weight.qweight"):
            remapped_sd[k.replace(".weight.qweight", ".weight_qweight")] = v
        elif k.endswith(".weight.scales"):
            remapped_sd[k.replace(".weight.scales", ".weight_scales")] = v
        elif k.endswith(".weight.group_size"):
            remapped_sd[k.replace(".weight.group_size", ".weight_group_size")] = v
        elif k.endswith(".weight.is_int8"):
            remapped_sd[k.replace(".weight.is_int8", ".weight_is_int8")] = v
        else:
            remapped_sd[k] = v

    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            qw_key = f"{name}.weight_qweight"
            if qw_key in remapped_sd:
                parent_name, _, child_name = name.rpartition(".")
                parent = model.get_submodule(parent_name) if parent_name else model
                has_bias = f"{name}.bias" in remapped_sd
                is_int8 = f"{name}.weight_is_int8" in remapped_sd
                if is_int8:
                    mod = Int8Linear(module.in_features, module.out_features, group_size=128, bias=has_bias)
                    count_int8 += 1
                else:
                    mod = FastInt4Linear(module.in_features, module.out_features, group_size=128, bias=has_bias)
                    count_int4 += 1
                setattr(parent, child_name, mod)
    return count_int4 + count_int8, remapped_sd


class PortableFP8Linear(nn.Module):
    """Ordinary-tensor FP8 storage with one-layer-at-a-time BF16 dequantization."""

    def __init__(
        self,
        qdata: torch.Tensor,
        scale: torch.Tensor,
        orig_shape: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        out_features, in_features = (int(value) for value in orig_shape.tolist())
        if tuple(qdata.shape) != (out_features, in_features):
            raise ValueError("portable FP8 payload shape does not match its declared shape")
        if scale.numel() not in (1, out_features):
            raise ValueError("portable FP8 scale is neither tensor-wise nor row-wise")
        self.in_features = in_features
        self.out_features = out_features
        self.W_nbits = 8
        self.qdata = nn.Parameter(qdata, requires_grad=False)
        self.scale = nn.Parameter(scale, requires_grad=False)
        self.bias = nn.Parameter(bias, requires_grad=False) if bias is not None else None
        self.register_buffer(
            "_weight_dtype_anchor", torch.empty(0, dtype=torch.bfloat16), persistent=False
        )

    @property
    def weight(self) -> torch.Tensor:
        """Compatibility anchor for H3's explicit ``linear.weight.dtype`` casts."""
        return self._weight_dtype_anchor

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Match TorchAO's source dequantization order: scale in FP32, round once
        # to the compute dtype, then run the ordinary matmul.
        weight = (self.qdata.float() * self.scale.float()).to(inputs.dtype)
        return torch.nn.functional.linear(inputs, weight, self.bias)


class EmbeddedLoraLinear(nn.Module):
    """Low-bit base projection plus an exact embedded BF16 PEFT residual."""

    def __init__(
        self,
        base: nn.Module,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
        scale: torch.Tensor,
    ) -> None:
        super().__init__()
        if lora_a.ndim != 2 or lora_b.ndim != 2:
            raise ValueError("embedded LoRA tensors must be matrices")
        if tuple(lora_a.shape)[1] != base.in_features:
            raise ValueError("embedded LoRA A input width does not match packed projection")
        if tuple(lora_b.shape) != (base.out_features, lora_a.shape[0]):
            raise ValueError("embedded LoRA B shape does not match packed projection")
        self.base = base
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.W_nbits = base.W_nbits
        self.lora_A = nn.Parameter(lora_a, requires_grad=False)
        self.lora_B = nn.Parameter(lora_b, requires_grad=False)
        override = os.environ.get("AIV_TURBO_LORA_ALPHA")
        self.scaling = (
            float(override) / int(lora_a.shape[0]) if override else float(scale.item())
        )

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base_output = self.base(inputs)
        residual = torch.nn.functional.linear(
            torch.nn.functional.linear(inputs.to(self.lora_A.dtype), self.lora_A),
            self.lora_B,
        )
        return base_output.add(residual.to(base_output.dtype), alpha=self.scaling)


def _prepare_portable_fp8_transformer(model, state_dict: Mapping) -> tuple[int, dict]:
    """Install ordinary-tensor FP8 modules that are safe for Diffusers offloading."""
    remapped = dict(state_dict)
    count = 0
    fields = ("qdata", "scale", "orig_shape")
    lora_fields = ("lora_A", "lora_B", "lora_scale")

    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        prefix = f"{name}.aiv_fp8."
        if prefix + "qdata" not in remapped:
            continue
        packed = {field: remapped.pop(prefix + field) for field in fields}
        bias_key = f"{name}.bias"
        if bias_key in remapped:
            packed["bias"] = remapped.pop(bias_key)
        base_module = PortableFP8Linear(**packed)
        lora_keys = {field: prefix + field for field in lora_fields}
        present_lora = [field for field, key in lora_keys.items() if key in remapped]
        if present_lora and len(present_lora) != len(lora_keys):
            raise ValueError(f"incomplete embedded LoRA state for {name}: {present_lora}")
        replacement: nn.Module = base_module
        if present_lora:
            replacement = EmbeddedLoraLinear(
                base_module,
                remapped.pop(lora_keys["lora_A"]),
                remapped.pop(lora_keys["lora_B"]),
                remapped.pop(lora_keys["lora_scale"]),
            )
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, replacement)
        count += 1

    if not count:
        raise ValueError("portable FP8 checkpoint did not install any projections")
    return count, remapped


def _gemlite_module_from_state(state: Mapping[str, torch.Tensor]):
    """Rebuild one pre-packed GemLite module without repacking or BF16 expansion."""
    try:
        from gemlite import DType, GemLiteLinearTriton
    except ImportError as error:
        raise RuntimeError(
            "this checkpoint requires GemLite. Repair the H3 runtime with scripts/setup-h3.ps1"
        ) from error

    metadata = state["metadata"]
    shape_tensor = state["orig_shape"]
    if metadata.numel() != 12 or shape_tensor.numel() != 2:
        raise ValueError(
            f"invalid GemLite checkpoint metadata: metadata={tuple(metadata.shape)}, "
            f"orig_shape={tuple(shape_tensor.shape)}"
        )
    values = [int(value) for value in metadata.tolist()]
    out_features, in_features = (int(value) for value in shape_tensor.tolist())
    (
        scaled_activations,
        weight_bits,
        group_size,
        unpack_mask,
        elements_per_sample,
        input_dtype,
        output_dtype,
        accumulator_dtype,
        metadata_dtype,
        channel_scale_mode,
        weight_group_mode,
        data_contiguous,
    ) = values
    if weight_bits not in (4, 8) or input_dtype not in (
        DType.INT8.value,
        DType.BF16.value,
        DType.FP16.value,
    ):
        raise ValueError(
            f"unsupported GemLite H3 layer: W{weight_bits}, input dtype id {input_dtype}"
        )

    module = GemLiteLinearTriton(
        weight_bits,
        group_size=group_size,
        in_features=in_features,
        out_features=out_features,
        input_dtype=DType(input_dtype),
        output_dtype=DType(output_dtype),
        acc_dtype=DType(accumulator_dtype),
        scaled_activations=bool(scaled_activations),
    )
    module.unpack_mask = unpack_mask
    module.elements_per_sample = elements_per_sample
    module.meta_dtype = DType(metadata_dtype)
    module.channel_scale_mode = channel_scale_mode
    module.W_group_mode = weight_group_mode
    module.data_contiguous = bool(data_contiguous)
    module.W_q = nn.Parameter(state["W_q"], requires_grad=False)
    module.scales = nn.Parameter(state["scales"], requires_grad=False)
    module.zeros = nn.Parameter(state["zeros"], requires_grad=False)
    module.metadata = nn.Parameter(metadata, requires_grad=False)
    module.orig_shape = nn.Parameter(shape_tensor, requires_grad=False)
    bias = state.get("bias")
    module.bias = nn.Parameter(bias, requires_grad=False) if bias is not None else None
    module.device = module.W_q.device
    return module


def _prepare_gemlite_transformer(model, state_dict: Mapping) -> tuple[tuple[int, int], dict]:
    """Install every serialized GemLite W4A8/W8A8 projection into the transformer."""
    remapped = dict(state_dict)
    counts = {4: 0, 8: 0}
    fields = ("W_q", "scales", "zeros", "metadata", "orig_shape")

    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        prefix = f"{name}.gemlite."
        if prefix + "W_q" not in remapped:
            continue
        packed = {}
        for field in fields:
            key = prefix + field
            if key not in remapped:
                raise ValueError(f"incomplete GemLite state for {name}: missing {field}")
            packed[field] = remapped.pop(key)
        lora_keys = {field: prefix + field for field in ("lora_A", "lora_B", "lora_scale")}
        present_lora = [field for field, key in lora_keys.items() if key in remapped]
        if present_lora and len(present_lora) != len(lora_keys):
            raise ValueError(f"incomplete embedded LoRA state for {name}: {present_lora}")
        bias_key = f"{name}.bias"
        if bias_key in remapped:
            packed["bias"] = remapped.pop(bias_key)

        packed_module = _gemlite_module_from_state(packed)
        if (
            packed_module.in_features != module.in_features
            or packed_module.out_features != module.out_features
        ):
            raise ValueError(
                f"GemLite shape mismatch for {name}: checkpoint "
                f"{packed_module.out_features}x{packed_module.in_features}, model "
                f"{module.out_features}x{module.in_features}"
            )
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        counts[packed_module.W_nbits] += 1
        if present_lora:
            packed_module = EmbeddedLoraLinear(
                packed_module,
                remapped.pop(lora_keys["lora_A"]),
                remapped.pop(lora_keys["lora_B"]),
                remapped.pop(lora_keys["lora_scale"]),
            )
        setattr(parent, child_name, packed_module)

    if not counts[4] and not counts[8]:
        raise ValueError(f"GemLite checkpoint did not install any packed projections: {counts}")
    return (counts[4], counts[8]), remapped


def load_transformer(args: argparse.Namespace, subfolder: str = "transformer") -> "MiniMaxH3Transformer3DModel":
    """The denoiser, from the pre-quantized state dict when one is installed."""
    import torch
    from accelerate import init_empty_weights
    from diffusers import MiniMaxH3Transformer3DModel

    checkpoint = Path(args.transformer_path) if args.transformer_path else None
    if checkpoint is not None and not checkpoint.is_file():
        raise FileNotFoundError(f"requested transformer checkpoint not found: {checkpoint}")
    if checkpoint is None:
        name = (
            "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
            if subfolder == "transformer_ref"
            else "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
        )
        checkpoint = Path(args.repo_root) / "diffusion_models" / name
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"published INT8 ConvRot checkpoint not found: {checkpoint}\n"
                "Install the corresponding MiniMax-H3 model pack."
            )

    if checkpoint.suffix.lower() != ".safetensors":
        raise ValueError("MiniMax-H3 denoiser weights must be a .safetensors file; .pt is unsupported")

    log(f"denoiser: pre-quantized {checkpoint.name} ({checkpoint.stat().st_size / 1024**3:.1f} GiB)")
    # The header alone. This probe used to map the whole 20 GiB denoiser to ask one question,
    # and it runs before every load -- see `safetensors_header`.
    from aiv_runtime.safetensors_header import has_key_suffix

    is_convrot = has_key_suffix(checkpoint, ".comfy_quant")
    if is_convrot:
        from aiv_runtime.minimax_h3_convrot import load_transformer as load_convrot_transformer

        turbo_path = None
        turbo_alpha = TURBO_LORA_ALPHA
        if args.sampling_mode in TURBO_NFE:
            if not args.turbo_lora_path:
                raise ValueError(f"{args.sampling_mode} requires --turbo-lora-path")
            turbo_path = Path(args.turbo_lora_path)
            turbo_alpha = turbo_lora_alpha(turbo_path)
        transformer = load_convrot_transformer(
            repo_root=Path(args.repo_root),
            subfolder=subfolder,
            checkpoint_path=checkpoint,
            turbo_lora_path=turbo_path,
            turbo_lora_alpha=turbo_alpha,
        )
        report = transformer._aiv_convrot_report
        log(
            "denoiser: official/community INT8 ConvRot · "
            f"{report['int8_projections']} projections · "
            f"{report['turbo_pairs']} Turbo pairs · ordinary offloadable tensors"
        )
        if report["turbo_pairs"]:
            source = "declared" if declared_lora_alpha(turbo_path) is not None else "filename table"
            log(
                f"turbo LoRA: {turbo_path.name} · rank {report['turbo_rank']} · "
                f"alpha {report['turbo_alpha']} ({source}) · scale {report['turbo_scale']:g} · "
                f"{TURBO_NFE[args.sampling_mode]} NFE · unfused residual"
            )
            if report["turbo_scale"] > 1.0 + 1e-6:
                log(
                    f"WARNING: turbo LoRA scale is {report['turbo_scale']:g}, above unity. A "
                    f"distillation adapter applied above its trained strength burns motion and "
                    f"detail; LightX2V's published alpha is {TURBO_LORA_ALPHA} at any rank."
                )
        return transformer

    config = MiniMaxH3Transformer3DModel.load_config(args.repo_root, subfolder=subfolder)
    with init_empty_weights():
        transformer = MiniMaxH3Transformer3DModel.from_config(config)

    from aiv_runtime.safetensors_header import metadata as read_metadata
    from aiv_runtime.safetensors_reader import describe_backend, load_state_dict

    state_dict = load_state_dict(checkpoint, device="cpu")
    log(f"denoiser read: {describe_backend()}")
    # A second mapping of the same 20 GiB file, for a few hundred bytes of metadata.
    metadata = read_metadata(checkpoint)
    checkpoint_format = metadata.get("format", "unsloth_prequant_transformer_state_dict_v1")

    transformer._is_turbo_baked = metadata.get("is_turbo_baked") == "True" or "Turbo" in checkpoint.name

    if "time_embedder.table" in state_dict:
        _install_curve_adaln(transformer, checkpoint_format, metadata, state_dict)

    is_portable_fp8 = any(".aiv_fp8.qdata" in key for key in state_dict)
    is_gemlite = any(".gemlite.W_q" in key for key in state_dict)
    is_int4 = any(k.endswith(".qweight") or k.endswith(".weight_qweight") for k in state_dict)
    if is_portable_fp8:
        fp8_count, state_dict = _prepare_portable_fp8_transformer(transformer, state_dict)
        log(
            f"denoiser: installed {fp8_count} portable source-FP8 projections; "
            "TorchAO tensor subclasses are not resident"
        )
    elif is_gemlite:
        (w4_count, w8_count), state_dict = _prepare_gemlite_transformer(transformer, state_dict)
        tiers = []
        if w4_count:
            tiers.append(f"{w4_count} packed W4")
        if w8_count:
            tiers.append(f"{w8_count} packed W8")
        log(f"denoiser: installed {', '.join(tiers)} genuine GemLite projections")
    elif is_int4:
        replaced, state_dict = _prepare_int4_transformer(transformer, state_dict)
        log(
            f"denoiser: installed {replaced} legacy packed layers; these expand to BF16 "
            "and are retained for checkpoint compatibility only"
        )

    transformer._aiv_fp8_mode = (
        "portable-fp8"
        if is_portable_fp8
        else (
            f"gemlite-{metadata.get('profile', 'packed')}"
            if is_gemlite
            else _configure_fp8_for_device(state_dict)
        )
    )

    incompatible = transformer.load_state_dict(
        state_dict, strict=not (is_int4 or is_gemlite or is_portable_fp8), assign=True
    )
    if incompatible.unexpected_keys:
        raise RuntimeError(
            "the quantized checkpoint left transformer tensors unused: "
            f"unexpected={incompatible.unexpected_keys[:1]}"
        )
    if "document" in locals():
        del document
    del state_dict
    gc.collect()
    transformer.requires_grad_(False)
    transformer.eval()
    return transformer


def _configure_fp8_for_device(state_dict: Mapping) -> str:
    """Use native FP8 on Hopper+, native INT8 for Int8Tensor, and weight-only dequantization everywhere else."""
    import torch

    if any(k.endswith(".qweight") or k.endswith(".weight_qweight") for k in state_dict):
        log("denoiser: native 4-bit INT4 packed weights resident on CUDA (zero PCIe swaps)")
        return "int4-resident"

    try:
        from torchao.quantization import Float8Tensor, Int8Tensor
    except ImportError:
        return "none"

    int8_tensors = [t for t in state_dict.values() if isinstance(t, Int8Tensor)]
    if int8_tensors:
        log(f"denoiser: native TorchAO INT8 on CUDA · {len(int8_tensors)} weights resident")
        return "int8-native"

    tensors = [tensor for tensor in state_dict.values() if isinstance(tensor, Float8Tensor)]
    if not tensors:
        return "none"
    capability = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
    if capability >= (9, 0):
        log(f"denoiser: native row-wise FP8 on SM{capability[0]}{capability[1]} · {len(tensors)} weights")
        return "native"
    for tensor in tensors:
        tensor.act_quant_kwargs = None
    device = f"SM{capability[0]}{capability[1]}" if torch.cuda.is_available() else "CPU"
    log(
        f"denoiser: {device} has no row-wise FP8 kernel · {len(tensors)} weights use "
        "TorchAO bfloat16 matmul fallback"
    )
    return "weight-only"


def _install_curve_adaln(transformer, checkpoint_format, metadata: Mapping, state_dict: Mapping) -> None:
    """Rebuild the pruned H3 timestep/AdaLN path recorded by Unsloth's checkpoint.

    The Comfy-Org pruned H3 source does not contain Diffusers' dense 256 -> 5376 -> 2688
    timestep MLP. It stores a sampled 1025 x 8 curve and makes every AdaLN projection consume
    those eight interpolated coordinates directly, without SiLU. Constructing the stock model
    therefore makes every AdaLN weight look like a shape mismatch even though the checkpoint is
    valid. This installs that architecture before ``assign=True`` restores the state dict.
    """
    import torch
    import torch.nn as nn

    table = state_dict["time_embedder.table"]
    is_df_format = str(checkpoint_format).startswith(("df_hybrid", "df_gemlite", "df_portable")) or str(
        metadata.get("format", "")
    ).startswith(("df_hybrid", "df_gemlite", "df_portable"))

    if not is_df_format:
        required = {
            "family": "minimax-h3",
            "transformer_class": "MiniMaxH3Transformer3DModel",
            "adaln_form": "curve",
        }
        actual = {"format": checkpoint_format, **dict(metadata)}
        mismatched = [
            f"{name}={actual.get(name)!r} (expected {value!r})"
            for name, value in required.items()
            if actual.get(name) != value
        ]
        if actual.get("scheme") not in ("fp8", "int8", "int4_packed", "int4"):
            mismatched.append(f"scheme={actual.get('scheme')!r} (expected 'fp8', 'int8', or 'int4_packed')")
        if mismatched:
            raise ValueError("unsupported H3 pre-quantized checkpoint metadata: " + ", ".join(mismatched))

    curve_dim = int(metadata.get("curve_dim", table.shape[1]))
    curve_grid = int(metadata.get("curve_grid", table.shape[0]))
    if curve_dim < 1 or curve_grid < 2 or tuple(table.shape) != (curve_grid, curve_dim):
        raise ValueError(
            "invalid H3 AdaLN curve table: "
            f"metadata says ({curve_grid}, {curve_dim}), checkpoint has {tuple(table.shape)}"
        )
    output_dtype_name = str(metadata.get("adaln_out_dtype", "bfloat16"))
    output_dtype = getattr(torch, output_dtype_name, None)
    if not isinstance(output_dtype, torch.dtype):
        raise ValueError(f"unsupported H3 AdaLN output dtype {output_dtype_name!r}")

    class CurveTimestepEmbedding(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer(
                "table",
                torch.empty((curve_grid, curve_dim), device="meta", dtype=table.dtype),
            )

        # The upstream transformer asks for ``time_embedder.linear_1.weight.dtype`` before calling
        # the embedder. These properties preserve that small interface without registering aliases
        # that would create extra state-dict keys.
        @property
        def linear_1(self):
            return self

        @property
        def weight(self):
            return self.table

        def forward(self, timestep):
            values = timestep.reshape(-1).to(dtype=torch.float32)
            position = values.clamp(0.0, 1.0) * (self.table.shape[0] - 1)
            lower = position.floor().long().clamp(max=self.table.shape[0] - 2)
            fraction = (position - lower).unsqueeze(1).to(dtype=self.table.dtype)
            return torch.lerp(self.table[lower], self.table[lower + 1], fraction)

    def curve_block_forward(module, temb):
        projected = module.linear(temb.to(module.linear.weight.dtype)).to(output_dtype)
        projected = projected.view(-1, 6 * module.hidden_size)
        return projected.chunk(6, dim=-1)

    def curve_out_forward(module, hidden_states, temb, timestep_indices):
        shift, scale = module.linear(temb.to(module.linear.weight.dtype)).to(output_dtype).chunk(2, dim=-1)
        hidden_states = module.norm(hidden_states)
        return hidden_states * (1.0 + scale.index_select(0, timestep_indices)) + shift.index_select(
            0, timestep_indices
        )

    transformer.time_proj = nn.Identity()
    transformer.time_embedder = CurveTimestepEmbedding()
    for index, block in enumerate(transformer.transformer_blocks):
        weight_name = f"transformer_blocks.{index}.adaln_proj.linear.weight"
        weight = state_dict.get(weight_name)
        expected_out = 6 * block.adaln_proj.hidden_size * 3
        if weight is None or tuple(weight.shape) != (expected_out, curve_dim):
            raise ValueError(
                f"invalid curve AdaLN weight {weight_name}: expected {(expected_out, curve_dim)}, "
                f"got {None if weight is None else tuple(weight.shape)}"
            )
        block.adaln_proj.linear = nn.Linear(
            curve_dim,
            expected_out,
            bias=True,
            device="meta",
            dtype=weight.dtype,
        )
        block.adaln_proj.forward = types.MethodType(curve_block_forward, block.adaln_proj)

    final_weight = state_dict.get("norm_out.linear.weight")
    final_out = 2 * transformer.norm_out.norm.normalized_shape[0]
    if final_weight is None or tuple(final_weight.shape) != (final_out, curve_dim):
        raise ValueError(
            f"invalid curve output-AdaLN weight: expected {(final_out, curve_dim)}, "
            f"got {None if final_weight is None else tuple(final_weight.shape)}"
        )
    transformer.norm_out.linear = nn.Linear(
        curve_dim,
        final_out,
        bias=True,
        device="meta",
        dtype=final_weight.dtype,
    )
    transformer.norm_out.forward = types.MethodType(curve_out_forward, transformer.norm_out)
    log(
        f"denoiser: restored pruned AdaLN curve · {curve_grid} grid points · "
        f"{curve_dim} coordinates · output {output_dtype_name}"
    )


def _load_turbo_lora_state(path: Path) -> tuple[Mapping[str, object], int]:
    """Load and validate LightX2V's pure-PEFT checkpoint before touching the model."""
    from aiv_runtime.safetensors_reader import load_state_dict

    state_dict = load_state_dict(path, device="cpu")
    lora_a: dict[str, object] = {}
    lora_b: dict[str, object] = {}
    unsupported: list[str] = []
    for key, tensor in state_dict.items():
        if key.endswith(LORA_A_SUFFIX):
            lora_a[key[: -len(LORA_A_SUFFIX)]] = tensor
        elif key.endswith(LORA_B_SUFFIX):
            lora_b[key[: -len(LORA_B_SUFFIX)]] = tensor
        else:
            unsupported.append(key)
    if unsupported:
        raise ValueError(f"Turbo checkpoint contains unsupported key {unsupported[0]!r}")
    if not lora_a or lora_a.keys() != lora_b.keys():
        raise ValueError("Turbo checkpoint does not contain paired PEFT LoRA A/B tensors")
    ranks: set[int] = set()
    for module_name, tensor_a in lora_a.items():
        tensor_b = lora_b[module_name]
        if tensor_a.ndim != 2 or tensor_b.ndim != 2 or tensor_a.shape[0] != tensor_b.shape[1]:
            raise ValueError(f"Turbo LoRA tensors for {module_name!r} have incompatible shapes")
        if not module_name.endswith(TURBO_LORA_TARGETS):
            raise ValueError(f"Turbo checkpoint targets unsupported module {module_name!r}")
        ranks.add(tensor_a.shape[0])
    if len(ranks) != 1:
        raise ValueError(f"Turbo checkpoint has mixed LoRA ranks: {sorted(ranks)}")
    return state_dict, ranks.pop()


def _cast_turbo_lora_parameters_bf16(adapter_parameters: Mapping[str, object]) -> int:
    """Cast only inference-time PEFT adapter parameters to BF16, preserving base FP8 weights."""
    import torch

    recast = 0
    for parameter in adapter_parameters.values():
        if parameter.dtype != torch.bfloat16:
            parameter.data = parameter.data.to(dtype=torch.bfloat16)
            recast += 1
    return recast


def turbo_shifts(adapter_path: str, video_override: float | None = None) -> tuple[float, float]:
    """The (video, audio) scheduler shifts this Turbo adapter was distilled against.

    Precedence, most specific last: the per-adapter published pair, then a caller's
    `video_override` (the Ref2VA worker's `--video-shift`), then `AIV_TURBO_VIDEO_SHIFT` /
    `AIV_TURBO_AUDIO_SHIFT`.

    The environment deliberately wins. It used to lose: the Ref2VA path always received an explicit
    `--video-shift` from the Rust command builder, so `AIV_TURBO_VIDEO_SHIFT` was documented as a
    dial that could not, in fact, be reached from inside the application.

    Raising the audio shift to 4-6 is the community remedy for the audio damage the four-step
    distillation causes; it is exposed rather than adopted because it is a taste dial, not a
    published value.
    """
    name = Path(adapter_path).name.lower()
    video, audio = TURBO_VIDEO_SHIFT, TURBO_AUDIO_SHIFT
    for fragments, shifts in TURBO_SHIFT_OVERRIDES:
        if all(fragment in name for fragment in fragments):
            video, audio = shifts
            break
    if video_override is not None:
        video = float(video_override)
    video = float(os.environ.get("AIV_TURBO_VIDEO_SHIFT") or video)
    audio = float(os.environ.get("AIV_TURBO_AUDIO_SHIFT") or audio)
    return video, audio


def declared_lora_alpha(adapter_path: str | Path) -> int | None:
    """The alpha this adapter states in its own safetensors metadata, if it states one.

    LightX2V's bf16 repacks carry `alpha` as a string in `__metadata__`. A file that describes
    itself outranks any table keyed on its filename, which is why this is consulted first: a
    renamed or newly published adapter then still applies at its trained strength.

    Anything unreadable, absent or not a positive integer answers `None` so the caller falls back
    rather than silently applying a nonsense scale.
    """
    from aiv_runtime.safetensors_header import metadata as read_metadata

    try:
        raw = read_metadata(adapter_path).get("alpha")
    except Exception:  # noqa: BLE001 - a metadata probe must never break loading
        return None
    if raw is None:
        return None
    try:
        value = int(float(raw))
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def turbo_lora_alpha(adapter_path: str | Path) -> int:
    """Return the alpha this LightX2V adapter should be applied at.

    Precedence, most specific last: the filename table for releases that declare nothing, the
    adapter's own declaration, then `AIV_TURBO_LORA_ALPHA`. See the TURBO_LORA_ALPHA comment for
    why this is absolute, not the rank.

    The environment read lives HERE, and that is the whole point. It used to live in
    `apply_turbo_lora` and in `EmbeddedLoraLinear.__init__`, and neither is on the live ConvRot
    path -- which resolves its alpha by calling this function. So the override was documented at
    the TURBO_LORA_ALPHA constant and was unreachable from the configuration that actually ships.
    Exactly the shape of the `AIV_TURBO_VIDEO_SHIFT` bug `turbo_shifts` already carries a note
    about; reading it in the one function every caller goes through fixes both workers at once.
    """
    override = os.environ.get("AIV_TURBO_LORA_ALPHA")
    if override and override.strip():
        return int(override)
    declared = declared_lora_alpha(adapter_path)
    if declared is not None:
        return declared
    name = Path(adapter_path).name.lower()
    return 128 if all(value in name for value in ("4step", "v1.0", "768p")) else TURBO_LORA_ALPHA


def apply_turbo_lora(transformer, path: str, nfe: int) -> None:
    """Inject the selected official LightX2V PEFT adapter into the active H3 transformer."""
    from peft import LoraConfig

    lora_path = Path(path)
    state_dict, rank = _load_turbo_lora_state(lora_path)
    # LightX2V's published alpha is absolute, not the rank. See TURBO_LORA_ALPHA. The environment
    # override is resolved inside `turbo_lora_alpha` so that the ConvRot path gets it too.
    alpha = turbo_lora_alpha(lora_path)
    scale = alpha / rank
    log(f"turbo LoRA: {lora_path.name} · rank {rank} · alpha {alpha} · scale {scale:g} · {nfe} NFE")
    if os.environ.get("AIV_TURBO_LORA_ALPHA"):
        log(f"turbo LoRA alpha overridden by AIV_TURBO_LORA_ALPHA={alpha}")
    if scale > 1.0 + 1e-6:
        log(
            f"WARNING: turbo LoRA scale is {scale:g}, above unity. A distillation adapter applied "
            f"above its trained strength burns motion and detail rather than sharpening them; "
            f"LightX2V's published alpha is {TURBO_LORA_ALPHA} at any rank."
        )
    transformer.add_adapter(LoraConfig(
        r=rank,
        lora_alpha=alpha,
        init_lora_weights=False,
        target_modules=list(TURBO_LORA_TARGETS),
        use_rslora=False,
    ))
    adapter_parameters = {
        name: parameter
        for name, parameter in transformer.named_parameters()
        if ".lora_A." in name or ".lora_B." in name
    }
    missing = sorted(adapter_parameters.keys() - state_dict.keys())
    unexpected = sorted(state_dict.keys() - adapter_parameters.keys())
    mismatched = [
        name for name, parameter in adapter_parameters.items()
        if name in state_dict and state_dict[name].shape != parameter.shape
    ]
    if missing or unexpected or mismatched:
        raise ValueError(
            "Turbo LoRA does not match this H3 transformer: "
            f"missing={missing[:1]}, unexpected={unexpected[:1]}, mismatched={mismatched[:1]}"
        )
    incompatible = transformer.load_state_dict(state_dict, strict=False)
    missing_lora = [
        key for key in incompatible.missing_keys if ".lora_A." in key or ".lora_B." in key
    ]
    if incompatible.unexpected_keys or missing_lora:
        raise RuntimeError(
            "Turbo LoRA loading left adapter tensors unused: "
            f"missing={missing_lora[:1]}, unexpected={incompatible.unexpected_keys[:1]}"
        )
    # PEFT creates inference adapters in its default FP32 dtype. ``load_state_dict`` copies the
    # BF16 checkpoint values into those existing Parameters but intentionally preserves their
    # FP32 dtype. Besides wasting memory, the FP32 LoRA branch promotes H3's Q/K/V activations and
    # sends SageAttention an unsupported dtype. Cast only the inference-only adapter Parameters;
    # converting the whole transformer would destroy the pre-quantized FP8 base weights.
    recast = _cast_turbo_lora_parameters_bf16(adapter_parameters)
    transformer.set_adapters("default", weights=1.0)
    transformer.requires_grad_(False)
    transformer.eval()
    tensor_count = len(state_dict)
    del state_dict
    gc.collect()
    log(
        f"sampling LightX2V Turbo · {nfe} NFE · rank {rank} · alpha "
        f"{alpha} · {tensor_count} adapter tensors · "
        f"BF16 recast {recast}"
    )


def _sage_compatible_call(query, key, value, attn_mask=None) -> bool:
    """Return whether one H3 attention call satisfies SageAttention's hard constraints."""
    import torch

    tensors = (query, key, value)
    return (
        attn_mask is None
        and all(tensor.device.type == "cuda" for tensor in tensors)
        and query.dtype == key.dtype == value.dtype
        and query.dtype in (torch.float16, torch.bfloat16)
    )


def _guard_h3_sage_dispatch(original_dispatch, mode: str = "sage"):
    """Wrap the MiniMax-H3 dispatcher with multi-engine attention acceleration."""
    import torch
    import torch.nn.functional as F

    reported_fallbacks: set[tuple[str, str]] = set()

    def guarded_dispatch(query, key, value, *args, backend=None, **kwargs):
        backend_name = getattr(backend, "value", backend)
        if isinstance(backend_name, str):
            backend_name = backend_name.lower()
        attn_mask = kwargs.get("attn_mask", args[0] if args else None)

        effective_backend = backend_name or mode
        if effective_backend in ("sage", "sage2_patched", "sage1", "sage2_stock"):
            if not _sage_compatible_call(query, key, value, attn_mask):
                reason = (
                    f"dtype={getattr(query, 'dtype', type(query))}"
                    if getattr(query, "dtype", None) == getattr(key, "dtype", None) == getattr(value, "dtype", None)
                    else "mixed dtypes"
                )
                device = getattr(getattr(query, "device", None), "type", "unknown")
                report_key = (reason, device)
                if report_key not in reported_fallbacks:
                    log(f"attention Sage per-call fallback to native SDPA ({reason}, device={device})")
                    reported_fallbacks.add(report_key)
                return original_dispatch(query, key, value, *args, backend="native", **kwargs)

            # Check if tensors are real PyTorch tensors (not mock objects in unit tests)
            if isinstance(query, torch.Tensor) and isinstance(key, torch.Tensor) and isinstance(value, torch.Tensor):
                # Sage 1.0 (INT8 Q/K + unquantized FP16/BF16 V)
                if effective_backend == "sage1":
                    import sageattention.core as sage_core
                    seq_len = query.shape[1]
                    pad_len = ((seq_len + 63) // 64) * 64
                    if pad_len != seq_len:
                        pad_q = F.pad(query, (0, 0, 0, 0, 0, pad_len - seq_len))
                        pad_k = F.pad(key, (0, 0, 0, 0, 0, pad_len - seq_len))
                        pad_v = F.pad(value, (0, 0, 0, 0, 0, pad_len - seq_len))
                    else:
                        pad_q, pad_k, pad_v = query, key, value
                    out = sage_core.sageattn_qk_int8_pv_fp16_cuda(
                        pad_q, pad_k, pad_v, tensor_layout="NHD", pv_accum_dtype="fp32"
                    )
                    if pad_len != seq_len:
                        out = out[:, :seq_len]
                    return out

                # Sage 2.0 Patched (INT8 Q/K + FP8 V + FP32 accumulator + 64-token stride padding)
                if effective_backend in ("sage", "sage2_patched", "sage2"):
                    import sageattention.core as sage_core
                    seq_len = query.shape[1]
                    pad_len = ((seq_len + 63) // 64) * 64
                    if pad_len != seq_len:
                        pad_q = F.pad(query, (0, 0, 0, 0, 0, pad_len - seq_len))
                        pad_k = F.pad(key, (0, 0, 0, 0, 0, pad_len - seq_len))
                        pad_v = F.pad(value, (0, 0, 0, 0, 0, pad_len - seq_len))
                    else:
                        pad_q, pad_k, pad_v = query, key, value
                    out = sage_core.sageattn_qk_int8_pv_fp8_cuda(
                        pad_q, pad_k, pad_v, tensor_layout="NHD", pv_accum_dtype="fp32+fp32"
                    )
                    if pad_len != seq_len:
                        out = out[:, :seq_len]
                    return out

            return original_dispatch(query, key, value, *args, backend="sage", **kwargs)

        return original_dispatch(query, key, value, *args, backend=backend, **kwargs)

    guarded_dispatch._aiv_h3_sage_fallback = True
    guarded_dispatch._aiv_original_dispatch = original_dispatch
    return guarded_dispatch


def _install_h3_sage_fallback(mode: str = "sage") -> None:
    """Route only unsupported MiniMax-H3 Sage calls to native SDPA."""
    from diffusers.models.transformers import transformer_minimax_h3 as h3_transformer_module

    original_dispatch = getattr(
        h3_transformer_module.dispatch_attention_fn,
        "_aiv_original_dispatch",
        h3_transformer_module.dispatch_attention_fn,
    )
    h3_transformer_module.dispatch_attention_fn = _guard_h3_sage_dispatch(original_dispatch, mode=mode)


def _uninstall_h3_sage_fallback() -> None:
    """Put the module's own dispatcher back. Idempotent, and safe to call when nothing is installed.

    Leaving the wrapper in place after a failed warm-up is worse than never installing it. The
    guard resolves `effective_backend = backend_name or mode`, so once the registry has been reset
    and Diffusers starts calling with `backend=None`, every attention call falls through to the
    *requested* Sage mode -- the one that just proved unavailable. The job then dies at the first
    transformer block, roughly 20 GB of weights into the load, having logged that it was falling
    back to native SDPA.
    """
    from diffusers.models.transformers import transformer_minimax_h3 as h3_transformer_module

    installed = h3_transformer_module.dispatch_attention_fn
    original = getattr(installed, "_aiv_original_dispatch", None)
    if original is not None:
        h3_transformer_module.dispatch_attention_fn = original


def configure_attention_backend(transformer, requested: str) -> str:
    """Configure the requested attention backend (Native, Sage 2 Patched, Sage 1, or Sage 2 Stock)."""
    transformer.reset_attention_backend()
    norm_req = (requested or "native").lower().strip()
    if norm_req in ("native", "sdpa"):
        log("attention native PyTorch SDPA (Lossless C++ / cuDNN)")
        return "native"

    try:
        import torch
        from diffusers.models.attention_dispatch import (
            _AttentionBackendRegistry,
            dispatch_attention_fn,
        )

        previous_backend = _AttentionBackendRegistry.get_active_backend()[0]
        _install_h3_sage_fallback(mode=norm_req)

        # Warm up test kernel with 64-padded test query
        query = torch.randn((1, 128, 8, 128), device="cuda", dtype=torch.bfloat16)
        dispatch_attention_fn(query, query, query, backend="sage")
        torch.cuda.synchronize()
        del query
        torch.cuda.empty_cache()

        transformer.set_attention_backend("sage")
        _AttentionBackendRegistry.set_active_backend(previous_backend)

        if norm_req == "sage1":
            log("attention Sage 1.0 enabled (High Quality · INT8/FP16 · Safe)")
            return "sage1"
        elif norm_req in ("sage", "sage2_patched", "sage2"):
            log("attention Sage 2.0 Patched enabled (Fast FP8 · FP32 Accumulator · Safe)")
            return "sage"
        elif norm_req == "sage2_stock":
            log("attention Sage 2.0 Stock enabled (Unpatched / Experimental)")
            return "sage2_stock"
    except Exception as error:
        transformer.reset_attention_backend()
        # Uninstall before touching the registry. Resetting the active backend is what starts the
        # `backend=None` calls that the wrapper would misroute, so the order matters.
        _uninstall_h3_sage_fallback()
        if "previous_backend" in locals():
            _AttentionBackendRegistry.set_active_backend(previous_backend)
        log(
            "attention Sage unavailable "
            f"({type(error).__name__}: {error}); falling back to native PyTorch SDPA"
        )
        return "native"

    log("attention Sage 2.0 Patched enabled")
    return "sage"


def log_memory(label: str) -> None:
    """Report process and CUDA memory at a real stage boundary."""
    import psutil

    rss = psutil.Process().memory_info().rss / 1024**3
    parts = [f"host RSS={rss:.2f} GiB"]
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        parts.extend(
            (
                f"VRAM allocated={torch.cuda.memory_allocated() / 1024**3:.2f} GiB",
                f"reserved={torch.cuda.memory_reserved() / 1024**3:.2f} GiB",
                f"peak={torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB",
                f"free={free / 1024**3:.2f}/{total / 1024**3:.2f} GiB",
            )
        )
    log(f"memory · {label} · " + " · ".join(parts))


def denoise_profile_context(args, transformer, workflow: str, profile: Mapping):
    """Build the opt-in observation context shared by FL2VA and Ref2VA workers."""
    if not profile_requested(args):
        return contextlib.nullcontext()
    width, height, num_frames = resolve_geometry(args)
    gpu_name = None
    if torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
        except (AssertionError, RuntimeError):
            gpu_name = None
    metadata = {
        "workflow": workflow,
        "sampling_mode": args.sampling_mode,
        "steps": int(args.steps),
        "scheduler_grid_points": scheduler_grid_points(args),
        "width": width,
        "height": height,
        "num_frames": num_frames,
        "attention_backend": args.attention_backend,
        "offload": args.offload,
        "offload_profile": dict(profile),
        "transformer_path": str(getattr(args, "transformer_path", None) or ""),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": gpu_name,
        "process_id": os.getpid(),
    }
    return MiniMaxH3DenoiseProfiler(
        transformer,
        resolve_profile_path(args),
        metadata=metadata,
        logger=log,
    )


def _result_value(result, name: str):
    value = result.get(name) if isinstance(result, Mapping) else getattr(result, name, None)
    if value is None:
        raise RuntimeError(f"MiniMax-H3 stage produced no {name!r}")
    return value


def save_denoised_latents(result, path: str, num_frames: int) -> None:
    """Persist only the decode contract so process exit can reclaim the denoiser completely."""
    latents = _result_value(result, "latents").detach().to("cpu")
    audio_latents = _result_value(result, "audio_latents").detach().to("cpu")
    torch.save(
        {"latents": latents, "audio_latents": audio_latents, "num_frames": int(num_frames)}, path
    )
    log(
        f"stage denoise complete · video latents {tuple(latents.shape)} {latents.dtype} · "
        f"audio latents {tuple(audio_latents.shape)} {audio_latents.dtype} -> {path}"
    )


def _minimal_decode_pipeline(repo_root: str, workflow: str, block_name: str, component_name: str):
    """Load one decoder block and its one model component—never both VAEs together."""
    from diffusers import ModularPipeline

    blocks = ModularPipeline.from_pretrained(repo_root).blocks.get_workflow(workflow)
    for name in tuple(blocks.sub_blocks):
        if name != block_name:
            blocks.sub_blocks.pop(name)
    pipeline = blocks.init_pipeline(repo_root)
    pipeline.load_components(
        names=[component_name],
        pretrained_model_name_or_path=repo_root,
        # The audio VAE's numerically sensitive filters are FP32. The video VAE asks for FP16 and
        # does not get it: `AutoencoderKLMiniMaxH3._keep_in_fp32_modules` pins `encoder`,
        # `decoder`, `quant_conv` and `post_quant_conv` to float32, so the request is a no-op and
        # the resident model is FP32 either way. `decode_video_frames_to_cpu` runs it under a
        # float16 autocast, which is where the precision saving actually comes from.
        dtype=torch.float32 if component_name == "audio_vae" else torch.float16,
    )
    return pipeline


def _decoded_chunk_to_frames(chunk: torch.Tensor) -> list:
    """Denormalize one completed GPU chunk and materialize independent CPU uint8 frames."""
    if chunk.shape[0] != 1:
        raise ValueError(f"the H3 exporter supports one video at a time, got batch {chunk.shape[0]}")
    mean = torch.tensor(H3_PIXEL_MEAN, dtype=torch.float32).view(1, -1, 1, 1, 1)
    std = torch.tensor(H3_PIXEL_STD, dtype=torch.float32).view(1, -1, 1, 1, 1)
    pixels = (chunk.to("cpu").float() * std + mean).clamp_(0, 1)
    packed = (pixels[0].permute(1, 2, 3, 0) * 255).round_().to(torch.uint8).numpy()
    # A copy per frame releases the chunk-sized backing allocation immediately.
    return [frame.copy() for frame in packed]


def temporal_decode_plan(num_latent_frames: int, tokens_chunk_size: int, token_drop: int) -> tuple[int, int]:
    """How many latent frames to pad, and how many temporal chunks to decode.

    Mirrors `AutoencoderKLMiniMaxH3._decode`, plus the guard ComfyUI carries and Diffusers does not
    (`comfy/ldm/minimax/vae.py::_decode_temporal_chunks`). Without it the chunk count is zero below
    seven latent frames -- `token_drop` subtracts a whole chunk that the padding never restores --
    and the decode loop then hands `torch.cat` an empty list. Padding one extra chunk is what
    ComfyUI does, and it is the reason ComfyUI can decode a short clip where Diffusers raises.

    A 124-frame generation encodes to 37 latent frames and decodes back to 124, so production never
    reached this; only short requests did.
    """
    pseudo_total_tokens = num_latent_frames + token_drop
    pad_tokens = (-pseudo_total_tokens) % tokens_chunk_size
    num_chunks = (pseudo_total_tokens + pad_tokens) // tokens_chunk_size - int(token_drop > 0)
    if num_chunks < 1:
        pad_tokens += tokens_chunk_size
        num_chunks += 1
    return pad_tokens, num_chunks


def decode_video_frames_to_cpu(vae, normalized_latents: torch.Tensor, tile_size: int) -> list:
    """Decode temporal clips and retire every completed clip directly to CPU frames.

    This mirrors AutoencoderKLMiniMaxH3._decode, including token padding and overlap blending,
    but avoids retaining all decoded GPU chunks for a final torch.cat.
    """
    device = next(vae.parameters()).device
    if device.type not in ("cuda", "cpu"):
        raise RuntimeError(f"unsupported video VAE device {device}")

    vae.enable_tiling(
        tile_sample_min_height=tile_size,
        tile_sample_min_width=tile_size,
        tile_sample_min_overlap_height=VAE_TILE_OVERLAP,
        tile_sample_min_overlap_width=VAE_TILE_OVERLAP,
    )
    ratio = vae.spatial_compression_ratio
    pixel_height = normalized_latents.shape[-2] * ratio
    pixel_width = normalized_latents.shape[-1] * ratio
    y_indices, _, _ = vae._split_tiles(pixel_height, tile_size, VAE_TILE_OVERLAP)
    x_indices, _, _ = vae._split_tiles(pixel_width, tile_size, VAE_TILE_OVERLAP)
    tile_count = len(y_indices) * len(x_indices)
    if tile_count == 1:
        log(
            f"video VAE tiling · {pixel_width}x{pixel_height} fits one {tile_size}px tile; "
            "decode is spatially untiled"
        )
    else:
        log(
            f"video VAE tiling · {pixel_width}x{pixel_height} · {len(x_indices)}x{len(y_indices)} "
            f"grid ({tile_count} tiles) · tile={tile_size} overlap={VAE_TILE_OVERLAP}"
        )

    latents_mean = torch.tensor(vae.config.latents_mean, device=device).view(1, -1, 1, 1, 1)
    latents_std = torch.tensor(vae.config.latents_std, device=device).view(1, -1, 1, 1, 1)
    z = normalized_latents.to(device) * latents_std + latents_mean

    tokens_chunk_size = vae.tokens_chunk_size
    token_drop = vae.config.token_drop
    temporal_ratio = vae.temporal_compression_ratio
    chunk_num_frames = tokens_chunk_size * temporal_ratio
    original_num_tokens = z.shape[2]
    pad_tokens, num_chunks = temporal_decode_plan(original_num_tokens, tokens_chunk_size, token_drop)
    if pad_tokens:
        z = torch.cat([z, z[:, :, -1:].repeat(1, 1, pad_tokens, 1, 1)], dim=2)

    log(
        f"video VAE temporal decode · {original_num_tokens} latent frames · "
        f"{num_chunks} clips · streaming completed frames to CPU"
    )
    frames = []
    overlap = None
    started = time.perf_counter()
    # Follow the weights. An FP16 autocast over BF16 weights would cast them down per operation
    # and throw away exactly the exponent range BF16 was chosen for.
    weight_dtype = next(vae.parameters()).dtype
    autocast_dtype = weight_dtype if weight_dtype in (torch.float16, torch.bfloat16) else torch.float16
    with torch.inference_mode(), torch.autocast(
        device_type=device.type, dtype=autocast_dtype, enabled=device.type == "cuda"
    ):
        for index in range(num_chunks):
            start = index * tokens_chunk_size
            clip = vae._decode_clip(z[:, :, start : start + tokens_chunk_size + vae.token_overlap])
            for part in range(int(token_drop > 0) + 1):
                frame_start = part * chunk_num_frames
                chunk = clip[:, :, frame_start : frame_start + chunk_num_frames]
                chunk = chunk[:, :, vae.frame_pre_padding :]
                if part == 0:
                    if overlap is not None:
                        chunk = vae._blend(overlap, chunk, vae.frame_overlap, dim=-3)
                    frames.extend(_decoded_chunk_to_frames(chunk))
                else:
                    # Copy only the overlap slice; otherwise the view pins the whole decoded clip.
                    overlap = chunk.contiguous()
            del clip, chunk
            if device.type == "cuda":
                torch.cuda.synchronize()
            log(
                f"video VAE progress · clip {index + 1}/{num_chunks} · "
                f"{len(frames)} frames on CPU · {time.perf_counter() - started:.1f}s"
            )
            log_memory(f"video clip {index + 1}/{num_chunks}")

        if overlap is not None:
            frames.extend(_decoded_chunk_to_frames(overlap))
            del overlap

    if pad_tokens:
        intra_tail = vae.config.clip_length % temporal_ratio
        pad_frames = sum(
            intra_tail if intra_tail and (original_num_tokens + k) % tokens_chunk_size == 0 else temporal_ratio
            for k in range(pad_tokens)
        )
        if pad_frames:
            del frames[-pad_frames:]
    log(f"video VAE complete · {len(frames)} CPU frames · {time.perf_counter() - started:.1f}s")
    return frames


def run_decode(args: argparse.Namespace, workflow: str | None = None) -> None:
    """Stage 3. In a fresh process, decode audio then video with one resident VAE at a time."""
    workflow = workflow or MODE_WORKFLOWS[args.mode]
    saved = torch.load(args.latents_path, map_location="cpu", weights_only=True)
    audio_latents = saved["audio_latents"]
    video_latents = saved["latents"]
    expected_frames = int(saved["num_frames"])
    del saved

    log(f"stage decode · workflow {workflow} · isolated audio VAE starting")
    audio_pipeline = _minimal_decode_pipeline(args.repo_root, workflow, "decode.audio", "audio_vae")
    audio_pipeline.audio_vae.requires_grad_(False)
    audio_pipeline.audio_vae.to("cuda")
    torch.cuda.reset_peak_memory_stats()
    log_memory("audio VAE resident")
    with torch.inference_mode():
        audio_result = audio_pipeline(
            audio_latents=audio_latents.to("cuda"), output=["audio", "sampling_rate"]
        )
    audio = _result_value(audio_result, "audio").detach().to("cpu")
    sampling_rate = int(_result_value(audio_result, "sampling_rate"))
    write_audio_output(args, audio, sampling_rate)
    del audio_result, audio_pipeline, audio_latents, audio
    gc.collect()
    torch.cuda.empty_cache()
    log_memory("audio VAE released")

    log("stage decode · isolated video VAE starting")
    video_pipeline = _minimal_decode_pipeline(args.repo_root, workflow, "decode.video", "vae")
    vae = video_pipeline.vae
    vae.requires_grad_(False)
    # Before `.to("cuda")`: this is the placement that holds the whole model on the card, so it is
    # where FP16/BF16 buys the most -- 10.4 GB against 5.2 GB, on a 16 GB budget shared with the
    # temporal chunk buffers that dominate peak decode memory.
    cast_video_vae(vae)
    vae.to("cuda")
    torch.cuda.reset_peak_memory_stats()
    log_memory("video VAE resident")
    frames = decode_video_frames_to_cpu(vae, video_latents, args.vae_tile)
    if len(frames) != expected_frames:
        raise RuntimeError(
            f"video VAE produced {len(frames)} frames, expected the denoised {expected_frames}"
        )
    del video_latents, vae, video_pipeline
    gc.collect()
    torch.cuda.empty_cache()
    log_memory("video VAE released")
    write_video_output(args, frames, len(frames))


def run_denoise(args: argparse.Namespace) -> None:
    """Stage 2. Denoise to CPU latents and exit before either output decoder is loaded."""
    import torch
    from diffusers import ModularPipeline
    from diffusers.hooks import apply_group_offloading

    width, height, num_frames = resolve_geometry(args)
    # Opened once, here, because two decisions depend on whether this request has any: whether the
    # video VAE is loaded at all, and what the pipeline is called with at the end.
    keyframes = _keyframes(args, width, height)
    workflow = MODE_WORKFLOWS[args.mode]
    work_label = (
        f"{TURBO_NFE[args.sampling_mode]} NFE Turbo"
        if args.sampling_mode in TURBO_NFE else f"{args.steps} NFE Base"
    )
    log(f"stage denoise · {workflow} · {width}x{height} · {num_frames} frames · {work_label}")

    blocks = ModularPipeline.from_pretrained(args.repo_root).blocks.get_workflow(workflow)
    # The conditioner already ran in the other process. Dropping the block is what stops
    # `load_components` from pulling 25 GiB of Qwen3-VL back in beside the transformer.
    blocks.sub_blocks.pop("text_encoder", None)
    blocks.sub_blocks.pop("decode.video", None)
    blocks.sub_blocks.pop("decode.audio", None)
    pipeline = blocks.init_pipeline(args.repo_root)
    transformer = load_transformer(args)
    if args.sampling_mode in TURBO_NFE:
        if getattr(transformer, "_is_turbo_baked", False):
            log("turbo LoRA: already fused in checkpoint · skipping runtime PEFT adapter")
        else:
            apply_turbo_lora(transformer, args.turbo_lora_path, TURBO_NFE[args.sampling_mode])
    pipeline.update_components(transformer=transformer)
    pipeline.load_components(
        names=["scheduler", "audio_scheduler"],
        pretrained_model_name_or_path=args.repo_root,
        dtype=torch.bfloat16,
    )
    instrument_scheduler_grid(pipeline)
    # `run_encode` keeps only the `text_encoder` block, so a request carrying `image` or
    # `last_image` reaches `vae.encode` for the FIRST time here, in the denoise process. A text
    # request never touches the VAE -- `MiniMaxH3KeyframeVaeEncoderStep` iterates over
    # `block_state.keyframes` and reads `components.vae` per item -- so loading 10 GiB of it for a
    # t2va run buys nothing. The audio VAE is decode-only and must not share this process.
    if keyframes:
        pipeline.load_components(
            names=["vae"], pretrained_model_name_or_path=args.repo_root
        )

    if args.sampling_mode in TURBO_NFE:
        video_shift, audio_shift = turbo_shifts(args.turbo_lora_path)
        assert_schedules_agree(video_shift, audio_shift, scheduler_grid_points(args))
        pipeline.scheduler.set_shift(video_shift)
        pipeline.audio_scheduler.set_shift(audio_shift)
        log(
            f"Turbo scheduler shifts · video={video_shift:g} · "
            f"audio={audio_shift:g} · grid={scheduler_grid_points(args)} points"
        )
    configure_attention_backend(pipeline.transformer, args.attention_backend)

    profile = effective_offload_profile(
        args.offload,
        getattr(pipeline.transformer, "_aiv_fp8_mode", None),
    )
    if getattr(pipeline.transformer, "_aiv_fp8_mode", None) == "weight-only" and offload_profile(
        args.offload
    )["use_stream"]:
        log("offload: disabled streamed prefetch for the SM89 bfloat16 FP8 fallback to prevent VRAM spill")
    log(
        f"offload {args.offload} \u00b7 {profile['blocks_per_group']} blocks/group \u00b7 "
        f"stream={profile['use_stream']} \u00b7 pin={'on-demand' if profile['low_cpu_mem_usage'] else 'all'} \u00b7 "
        f"vae={'resident' if profile['vae_resident'] else 'offloaded'}"
    )
    pipeline.transformer.requires_grad_(False)
    # Bound once: both the transformer below and the keyframe VAE further down offload to the same
    # pair of devices, and this used to be defined inside the transformer's `else` alone.
    devices = {"onload_device": torch.device("cuda"), "offload_device": torch.device("cpu")}
    if profile.get("is_resident") or args.offload in ("resident", "none"):
        log("denoiser: resident VRAM mode enabled · zero PCIe swapping")
        pipeline.transformer.to("cuda")
    else:
        pipeline.transformer.enable_group_offload(
            offload_type="block_level",
            num_blocks_per_group=profile["blocks_per_group"],
            use_stream=profile["use_stream"],
            record_stream=profile["record_stream"],
            low_cpu_mem_usage=profile["low_cpu_mem_usage"],
            **devices,
        )

    # The keyframe VAE has to be placed on a device, and between 2026-08-25 and 2026-08-26 it was
    # not. `781f61f` deleted this block during the ConvRot rebase and left a comment asserting a
    # "CPU-resident VAE" was fine for keyframe encoding. It is not:
    # `MiniMaxH3KeyframeVaeEncoderStep` moves its pixels to `components._execution_device`, which
    # is CUDA because the transformer is there, and the first Conv3D then raises
    #
    #     RuntimeError: Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor)
    #     should be the same
    #
    # which is the exact failure the deleted comment had predicted. Every keyframe mode -- image,
    # first-last, last-only, clip-extend, clip-to-clip -- died in Stage 2 for a full day; only
    # text-to-video survived, because it never reaches the block.
    #
    # Leaf level, not block level: this VAE carries `conv_in`, `quant_conv`, `post_quant_conv` and
    # the decoder's entry/output projections outside its repeated block lists
    # (`_no_split_modules` names only the resnet and transformer blocks), and a block-level hook
    # follows the repeated blocks alone -- leaving exactly those projections behind on CPU.
    denoise_vae = getattr(pipeline, "vae", None)
    if denoise_vae is not None:
        denoise_vae.requires_grad_(False)
        cast_video_vae(denoise_vae)
        if hasattr(denoise_vae, "enable_tiling"):
            denoise_vae.enable_tiling(
                tile_sample_min_height=VAE_TILE_SIZE,
                tile_sample_min_width=VAE_TILE_SIZE,
                tile_sample_min_overlap_height=VAE_TILE_OVERLAP,
                tile_sample_min_overlap_width=VAE_TILE_OVERLAP,
            )
        # Not float16, whatever it was asked for: `AutoencoderKLMiniMaxH3._keep_in_fp32_modules`
        # pins `encoder`, `decoder`, `quant_conv` and `post_quant_conv` to float32, and `encode`
        # casts the incoming pixels to the encoder's own dtype. Reported rather than assumed,
        # because this file believed it was holding an FP16 VAE for a long time.
        vae_dtype = next(denoise_vae.parameters()).dtype  # after any AIV_H3_VAE_DTYPE cast
        if profile["vae_resident"]:
            denoise_vae.to("cuda")
            log(f"keyframe VAE · resident on CUDA · {vae_dtype}")
        else:
            apply_group_offloading(
                denoise_vae,
                offload_type=VIDEO_VAE_OFFLOAD_TYPE,
                use_stream=profile["use_stream"],
                record_stream=profile["record_stream"],
                low_cpu_mem_usage=profile["low_cpu_mem_usage"],
                **devices,
            )
            log(f"keyframe VAE · {VIDEO_VAE_OFFLOAD_TYPE} group offload · {vae_dtype}")

    first_block_cache = install_first_block_cache(
        pipeline.transformer,
        enabled=getattr(args, "first_block_cache", False),
        threshold=getattr(
            args, "first_block_cache_threshold", DEFAULT_FIRST_BLOCK_CACHE_THRESHOLD
        ),
        total_evaluations=args.steps,
    )
    if first_block_cache is not None:
        log(
            f"first block cache enabled · threshold={first_block_cache.threshold:.3f} · "
            f"warmup={first_block_cache.warmup_evaluations}/{first_block_cache.total_evaluations} evaluations"
        )

    saved = torch.load(args.embeds_path, map_location="cpu", weights_only=True)
    # This generator draws the denoising noise. It is NOT what encodes the keyframes, and the
    # difference matters enough to state: `MiniMaxH3KeyframeVaeEncoderStep` calls
    # `encode_vae_condition(..., components.keyframe_encode_seed)`, and that helper does
    # `posterior.sample(generator=torch.Generator().manual_seed(encode_seed))` with
    # `keyframe_encode_seed` fixed at 42 -- a fresh generator, seeded independently of the
    # request -- then rounds the sample through float16 and returns float32 on CPU. So the
    # posterior IS sampled rather than taken at its mode, but the same keyframe always encodes to
    # the same anchor whatever seed the user asked for, and passing a generator here or in
    # `run_encode` would not reach it. Read against the installed diffusers on 2026-09-08.
    generator = torch.Generator().manual_seed(args.seed)
    with denoise_profile_context(args, pipeline.transformer, workflow, profile), torch.inference_mode():
        result = pipeline(
            prompt=args.prompt,
            prompt_embeds=saved["prompt_embeds"].to("cuda", dtype=torch.bfloat16),
            text_token_tags=saved["text_token_tags"],
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=scheduler_grid_points(args),
            generator=generator,
            **keyframes,
        )
    log_memory("denoise complete")
    if first_block_cache is not None:
        log(first_block_cache.summary())
    save_denoised_latents(result, args.latents_path, num_frames)


def write_video_output(args: argparse.Namespace, frames, num_frames: int) -> None:
    """Write the CPU frame stream to the editor's silent MP4 asset.

    Deliberately NOT `diffusers.utils.export_to_video`. That helper assumes numpy frames are float
    in `[0, 1]` and rescales unconditionally::

        video_frames = [(frame * 255).astype(np.uint8) for frame in video_frames]

    `_decoded_chunk_to_frames` already returns uint8 in `[0, 255]`, because the streaming decode
    exists to keep the frame list small and float32 frames are four times the size. Under NumPy 2's
    NEP 50 rules `uint8 * 255` stays uint8 and wraps modulo 256, so that line maps every pixel to
    `256 - x`: a full inversion, and one that folds true black onto itself instead of onto white.
    It cost a whole generation on 2026-08-25 -- correct content, inverted, with the wrap
    discontinuity showing as burned speckle wherever the picture was darkest.

    Nobody else feeds that helper uint8. MiniMax's own `encode_video` guards the rescale on a range
    check, diffusers' pipeline keeps float `[0, 1]` and lets the writer scale, and ComfyUI converts
    to uint8 only at save time. Below is what `export_to_video` does anyway -- imageio with its own
    defaults -- minus the rescale, so the encode settings are unchanged and only the pixels differ.
    """
    import numpy as np

    # Checked before anything else is touched -- before the encoder is imported and before a
    # directory is created. Load-bearing, not defensive: should the decoder ever hand back float
    # frames again, the writer would silently darken them to near-black rather than inverting them,
    # which is far harder to spot than the inversion was. Putting it first also means the error
    # names the real problem on a machine that has no encoder installed, instead of a
    # ModuleNotFoundError that says nothing about the caller's mistake.
    if frames:
        dtype = getattr(frames[0], "dtype", None)
        if dtype != np.uint8:
            raise TypeError(
                f"the video writer takes uint8 RGB frames in [0, 255], got {dtype}. Rescale at the "
                "decoder (`_decoded_chunk_to_frames`), not here; this writer must never scale."
            )

    import imageio

    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    # `quality=5.0` and `macro_block_size=16` are `export_to_video`'s defaults, restated so a
    # change upstream cannot silently re-encode this output differently.
    with imageio.get_writer(str(output), fps=FPS, quality=5.0, macro_block_size=16) as writer:
        for frame in frames:
            writer.append_data(frame)
    log(f"wrote {output.name} ({num_frames} frames)")


#: 32-bit float WAV. `soundfile`'s default subtype for WAV is PCM_16, which maps float 1.0 to full
#: scale and hard-clips anything above it -- and the H3 audio decoder does NOT bound its output.
#: The released decoder clamps the video branch (`decoders.py`: `(video * std + mean).clamp(0, 1)`)
#: and deliberately leaves the audio branch unbounded, so a loud generation arrives with samples
#: past +/-1.0 and every one of them used to be squared off on the way to disk. That is exactly the
#: harsh, crackling "blown microphone" audio reported upstream (ComfyUI #15283, frontend #15040) --
#: except that in our case the distortion was ours, not the model's. Diffusers' own `encode_video`
#: never writes a 16-bit WAV either; it muxes AAC through PyAV.
AUDIO_WAV_SUBTYPE = "FLOAT"


def write_audio_output(args: argparse.Namespace, audio: torch.Tensor, rate: int) -> None:
    """Write the CPU stereo waveform to the editor's linked WAV asset, losslessly.

    Nothing here scales, clamps or normalizes. The peak is reported instead, because a peak above
    unity is information about the generation -- and silently normalizing it away would hide both
    a genuinely hot take and a future decoder fault behind the same quiet output.
    """
    import numpy as np
    import soundfile

    audio_output = audio_path_for(args)
    audio_output.parent.mkdir(parents=True, exist_ok=True)
    samples = audio.detach().float().numpy()
    samples = np.squeeze(samples)
    # soundfile wants (samples, channels); the pipeline emits (1, 2, num_samples).
    if samples.ndim == 2 and samples.shape[0] < samples.shape[1]:
        samples = samples.T
    peak = float(np.abs(samples).max()) if samples.size else 0.0
    soundfile.write(str(audio_output), samples, rate, subtype=AUDIO_WAV_SUBTYPE)
    headroom = (
        f"peak {peak:.3f}"
        if peak <= 1.0
        else f"peak {peak:.3f} -- above unity, and preserved: a PCM_16 write would have clipped "
             f"{float((np.abs(samples) > 1.0).mean()) * 100:.2f}% of samples"
    )
    log(f"wrote {audio_output.name} ({rate} Hz, {AUDIO_WAV_SUBTYPE}, {headroom})")


def write_outputs(args: argparse.Namespace, result, num_frames: int) -> None:
    """Compatibility helper for callers that already hold both decoded outputs."""
    frames = result["videos"] if isinstance(result, dict) else result.videos
    if frames and isinstance(frames[0], (list, tuple)):
        frames = frames[0]
    audio = result["audio"] if isinstance(result, dict) else result.audio
    rate = int(result["sampling_rate"] if isinstance(result, dict) else result.sampling_rate)
    write_audio_output(args, audio.detach().to("cpu"), rate)
    write_video_output(args, frames, num_frames)


def run_all(args: argparse.Namespace) -> None:
    """Drive conditioner, denoiser and decoder in mutually exclusive processes."""
    embeds_handle, embeds_path = tempfile.mkstemp(suffix=".h3-embeds.pt")
    latents_handle, latents_path = tempfile.mkstemp(suffix=".h3-latents.pt")
    os.close(embeds_handle)
    os.close(latents_handle)
    try:
        base = [sys.executable, str(Path(__file__).resolve())]
        passthrough = sys.argv[1:]
        for stage in ("encode", "denoise", "decode"):
            command = base + passthrough + [
                "--stage", stage,
                "--embeds-path", embeds_path,
                "--latents-path", latents_path,
            ]
            # stderr is inherited so progress streams through live; stdout is CAPTURED because the
            # child emits the same result JSON this process is about to, and the Rust side reads
            # stdout as one line. Nothing is lost -- every log line already goes to stderr.
            completed = subprocess.run(
                command, cwd=os.getcwd(), check=False, stdout=subprocess.PIPE, text=True
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"the MiniMax-H3 {stage} stage exited with code {completed.returncode}"
                )
            if completed.stdout and completed.stdout.strip():
                log(f"stage {stage} returned {completed.stdout.strip()}")
    finally:
        Path(embeds_path).unlink(missing_ok=True)
        Path(latents_path).unlink(missing_ok=True)


def main() -> int:
    args = build_parser().parse_args()
    validate(args)
    width, height, num_frames = resolve_geometry(args)
    if args.dry_run:
        print(json.dumps({
            "ok": True,
            "mode": args.mode,
            "workflow": MODE_WORKFLOWS[args.mode],
            "width": width,
            "height": height,
            "numFrames": num_frames,
            "seconds": round(num_frames / FPS, 3),
            "steps": args.steps,
            "samplingMode": args.sampling_mode,
            "schedulerGridPoints": scheduler_grid_points(args),
            "attentionBackend": args.attention_backend,
            "offload": args.offload,
            "profileDenoise": profile_requested(args),
            "profilePath": str(resolve_profile_path(args)) if profile_requested(args) else None,
            "outputPath": args.output_path,
            "audioPath": str(audio_path_for(args)),
        }))
        return 0

    if args.stage == "all":
        run_all(args)
    elif args.stage == "encode":
        run_encode(args)
    elif args.stage == "denoise":
        run_denoise(args)
    else:
        run_decode(args)

    if args.stage in ("all", "decode"):
        print(json.dumps({
            "ok": True,
            "outputPath": args.output_path,
            "audioPath": str(audio_path_for(args)),
            "numFrames": num_frames,
            "frameRate": FPS,
        }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
