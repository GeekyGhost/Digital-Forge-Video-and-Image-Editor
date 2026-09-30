"""Kiwi-Edit 5B clip editing adapter tuned for a single 16 GB NVIDIA GPU.

Kiwi-Edit performs instruction-guided and reference-guided editing of an existing video clip.
The editor calls this worker with a source clip, a source-relative time range and an instruction;
the worker regenerates only that range and writes an MP4 that the editor drops back onto the
timeline in place of the original media.

This adapter is written against the pipeline that actually ships in the pack
(`models/kiwi-edit-5b/pipeline_kiwi_edit.py`, class `KiwiEditPipeline`):

    pipeline(prompt=..., source_video=[PIL], ref_image=[PIL] | None, height=, width=,
             num_frames=, num_inference_steps=, seed=, output_type="pil")
        -> List[PIL.Image]

Its geometry rules come from `_check_resize(..., h_div=32, w_div=32, t_div=4, t_rem=1)`:
height and width must be multiples of 32 and the frame count must be 4n+1. The pack has no
mask parameter, so only instruction and reference modes exist.

Memory strategy, borrowed from how ComfyUI runs oversized models on small cards:

* **Staging.** Only one model group is resident at a time. The encoder (7.1 GiB) owns the GPU for
  step 3, is evicted, and the transformer + VAE (11.9 GiB) take its place for the rest. Peak usage
  is the largest stage, not the 19.0 GiB sum, and both stages run at full GPU speed. This is the
  same idea as ComfyUI unloading CLIP after conditioning and before sampling.
* **Block swapping.** For tighter cards, the first N of the transformer's 30 blocks (~0.31 GiB
  each) stream from the CPU, as ComfyUI-WanVideoWrapper does. Viable here because `_model_forward`
  runs `for block in t.blocks:`, an ordinary module call, so forward hooks fire.

Accelerate's own offload helpers cannot be used at all -- see `stage_components` for why.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

# The pipeline's own defaults, and the ceiling this profile allows: a 5B DiT plus a 3B Qwen2.5-VL
# encoder leaves little headroom for activations on a 16 GB card.
MAX_PIXELS = 832 * 480
MAX_FRAMES = 81
WORKING_FPS = 15.0
LONG_OVERLAP_FRAMES = 25
SPATIAL_MULTIPLE = 32
TEMPORAL_DIVISOR = 4
TEMPORAL_REMAINDER = 1


def force_utf8_io() -> None:
    """diffusers and transformers put emoji in their warnings. On Windows the default console
    encoding is cp1252, so merely *logging* one raises UnicodeEncodeError and aborts whatever
    was in progress. Re-encode our streams and never let an unprintable character be fatal."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Kiwi-Edit 5B clip editing for 16 GB GPUs.")
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--video-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--reference-image", default=None)
    parser.add_argument("--start-time", type=float, required=True)
    parser.add_argument("--end-time", type=float, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=("instruction", "reference"), default="instruction")
    parser.add_argument("--strategy", choices=("stable", "long"), default="stable")
    parser.add_argument("--working-fps", type=float, default=WORKING_FPS)
    parser.add_argument("--overlap-frames", type=int, default=LONG_OVERLAP_FRAMES)
    parser.add_argument("--steps", type=int, default=24)
    # Classifier-free guidance. The pack ships with its CFG branch COMMENTED OUT and `context_nega`
    # pinned to None, so these two were accepted and silently discarded: the instruction conditioned
    # the model and nothing steered toward it, which is why edits came back looking like the input.
    # `check_pack_guidance` below verifies the restored branch is present before trusting this.
    # 1.0 = off, and off is byte-identical to the old behaviour.
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--diagnostics", dest="diagnostics", action="store_true", default=True)
    parser.add_argument("--no-diagnostics", dest="diagnostics", action="store_false")
    # Deliberately not `choices=`: the editor binary and this worker are versioned separately, so
    # a freshly updated adapter can be invoked by an app build that predates it. Accept the older
    # spellings instead of aborting on a value we know how to translate.
    parser.add_argument("--offload", default="balanced")
    parser.add_argument("--dry-run", action="store_true")
    return parser


# `disk` and `cpu` were the sequential/model offload options this adapter used before it placed
# components explicitly. They map onto the closest equivalent so an older editor build still runs.
PLACEMENT_ALIASES = {
    "balanced": "balanced",
    "performance": "performance",
    "minimal": "minimal",
    "disk": "balanced",
    "cpu": "performance",
}


def normalise_placement(args: argparse.Namespace) -> None:
    requested = str(args.offload).strip().lower()
    resolved = PLACEMENT_ALIASES.get(requested)
    if resolved is None:
        raise ValueError(
            f"unknown placement {args.offload!r}; expected one of "
            + ", ".join(sorted(set(PLACEMENT_ALIASES.values())))
        )
    if resolved != requested:
        print(f"[kiwi-edit] placement {requested!r} is legacy; using {resolved!r}", file=sys.stderr)
    args.offload = resolved


def validate(args: argparse.Namespace) -> None:
    for value in (args.model_root, args.video_path):
        if not Path(value).exists():
            raise FileNotFoundError(value)
    pack = Path(args.model_root)
    if not (pack / "model_index.json").is_file():
        raise FileNotFoundError("kiwi-edit pack is missing model_index.json; reinstall the pack")
    if not (pack / "pipeline_kiwi_edit.py").is_file():
        raise FileNotFoundError("kiwi-edit pack is missing pipeline_kiwi_edit.py; reinstall the pack")
    if not args.instruction.strip():
        raise ValueError("instruction cannot be empty")
    if args.start_time < 0 or args.end_time <= args.start_time:
        raise ValueError("the clip editor requires 0 <= start-time < end-time")
    if Path(args.output_path).suffix.lower() != ".mp4":
        raise ValueError("clip editor output must be an MP4 file")
    if args.mode == "reference" and not args.reference_image:
        raise ValueError("reference mode requires --reference-image")
    if args.reference_image and not Path(args.reference_image).is_file():
        raise FileNotFoundError(args.reference_image)
    if not 4 <= args.steps <= 60:
        raise ValueError("steps must be between 4 and 60")
    if not 5.0 <= args.working_fps <= 30.0:
        raise ValueError("working FPS must be between 5 and 30")
    if args.overlap_frames < 5 or args.overlap_frames >= MAX_FRAMES:
        raise ValueError(f"overlap must be between 5 and {MAX_FRAMES - 1} frames")
    if args.overlap_frames % TEMPORAL_DIVISOR != TEMPORAL_REMAINDER:
        raise ValueError("overlap must be 4n+1 frames so temporal latent boundaries stay aligned")


REINSTALL_HINT = (
    ".venv\\Scripts\\python.exe -m pip install -e \".\\ai-runtime[inference]\""
)


def require_dependencies() -> None:
    """Fail with an actionable message rather than a bare ImportError or a KeyError raised
    twelve frames inside vendored modeling code."""
    import importlib.util

    missing = [
        name for name in ("torch", "diffusers", "transformers", "einops", "torchvision")
        if importlib.util.find_spec(name) is None
    ]
    if missing:
        raise RuntimeError(
            "the clip editor needs these Python packages: " + ", ".join(missing)
            + ". Install them with: " + REINSTALL_HINT
        )

    # The pack vendors Qwen2.5-VL modeling code from the transformers 4.x era. transformers 5.x
    # dropped the "default" entry from ROPE_INIT_FUNCTIONS, and that code indexes it directly,
    # so the failure surfaces as `KeyError: 'default'` while building the text model.
    import transformers

    major = int(str(transformers.__version__).split(".")[0])
    if major >= 5:
        raise RuntimeError(
            f"transformers {transformers.__version__} is too new for the Kiwi-Edit pack, which "
            "vendors Qwen2.5-VL modeling code from the 4.x era (transformers 5 removed the "
            "'default' rope initialiser it depends on). Install a supported version with: "
            + REINSTALL_HINT
        )

    # Same class of problem: the pack needs WanTransformer3DModel, added in diffusers 0.33.
    import diffusers

    diffusers_parts = str(diffusers.__version__).split(".")
    if (int(diffusers_parts[0]), int(diffusers_parts[1])) < (0, 33):
        raise RuntimeError(
            f"diffusers {diffusers.__version__} is too old for the Kiwi-Edit pack, which needs "
            "WanTransformer3DModel from 0.33 or newer. Install a supported version with: "
            + REINSTALL_HINT
        )


def warn_on_low_memory() -> None:
    """Offloading targets a 16 GB card, but every weight stays resident in system RAM and is
    streamed to the GPU on demand, so the host needs far more memory than the GPU does."""
    try:
        import psutil
    except ImportError:
        return
    available = psutil.virtual_memory().available / (1024 ** 3)
    if available < 20:
        print(
            f"[kiwi-edit] only {available:.1f} GiB of system RAM is free. The pack weighs about "
            "19 GiB and is staged through RAM before placement; close other applications if this "
            "run is killed or starts thrashing.",
            file=sys.stderr,
        )


def probe(path: str) -> dict:
    """Reads the stream geometry with ffprobe so limits are enforced before any weights load."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,avg_frame_rate,r_frame_rate",
            "-show_entries", "format=duration", "-of", "json", path,
        ],
        capture_output=True, text=True, check=True,
    )
    payload = json.loads(result.stdout)
    stream = (payload.get("streams") or [{}])[0]
    numerator, _, denominator = (stream.get("avg_frame_rate") or "24/1").partition("/")
    try:
        fps = float(numerator) / float(denominator or 1)
    except (ValueError, ZeroDivisionError):
        fps = 24.0
    audio = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=index", "-of", "csv=p=0", path],
        capture_output=True, text=True, check=False,
    )
    return {
        "width": int(stream.get("width") or 0),
        "height": int(stream.get("height") or 0),
        "fps": fps if fps > 0 else 24.0,
        "duration": float((payload.get("format") or {}).get("duration") or 0.0),
        "hasAudio": audio.returncode == 0 and bool(audio.stdout.strip()),
    }


def plan(source: dict, args: argparse.Namespace) -> dict:
    """Derive a deterministic CFR working layout before any model weights are loaded."""
    width, height = source["width"], source["height"]
    if width <= 0 or height <= 0:
        raise ValueError("the source clip has no decodable video stream")

    scale = min(1.0, math.sqrt(MAX_PIXELS / float(width * height)))
    target_width = max(SPATIAL_MULTIPLE, round(width * scale / SPATIAL_MULTIPLE) * SPATIAL_MULTIPLE)
    target_height = max(SPATIAL_MULTIPLE, round(height * scale / SPATIAL_MULTIPLE) * SPATIAL_MULTIPLE)
    # Rounding to the grid can push both axes up and land above the VRAM budget, so step the
    # longer axis back down until the frame fits again.
    while target_width * target_height > MAX_PIXELS:
        if target_width >= target_height and target_width > SPATIAL_MULTIPLE:
            target_width -= SPATIAL_MULTIPLE
        elif target_height > SPATIAL_MULTIPLE:
            target_height -= SPATIAL_MULTIPLE
        else:
            break

    end = min(args.end_time, source["duration"]) if source["duration"] > 0 else args.end_time
    span = max(1.0 / args.working_fps, end - args.start_time)
    requested = max(1, int(round(span * args.working_fps)))
    windows = build_windows(requested, args.strategy, args.overlap_frames)
    return {
        "width": target_width,
        "height": target_height,
        "frames": requested,
        "requestedFrames": requested,
        "windows": windows,
        "fps": args.working_fps,
        "sourceFps": source["fps"],
        "strategy": args.strategy,
        "overlapFrames": args.overlap_frames if args.strategy == "long" else 0,
        "startTime": args.start_time,
        "endTime": end,
        "requestedDuration": end - args.start_time,
        "hasAudio": source.get("hasAudio", False),
    }


def check_pack_guidance(args: argparse.Namespace) -> None:
    """Refuse to pretend guidance is on when the pack's CFG branch is still commented out.

    `models.py install kiwi-edit` rewrites the pack from the snapshot, which would silently revert
    the restored branch and put us straight back to "the instruction is ignored" with no error and
    no clue. Cheap to check, and the failure it prevents costs a multi-minute run to discover.
    """
    if args.guidance_scale == 1.0:
        return
    source = Path(args.model_root) / "pipeline_kiwi_edit.py"
    try:
        text = source.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return
    live = "noise_pred_nega + guidance_scale" in text and "#     noise_pred_nega" not in text
    if not live:
        raise RuntimeError(
            f"--guidance-scale {args.guidance_scale} was requested but {source.name} still has its "
            "classifier-free-guidance branch commented out, so guidance would be accepted and then "
            "ignored. The pack was probably reinstalled and overwrote the restored branch. Re-apply "
            "it, or run with --guidance-scale 1.0 to acknowledge that the instruction only "
            "conditions and never steers."
        )


def _window_change(source_frames: list, edited_frames: list) -> tuple[float, float]:
    """Mean and peak per-frame change between what went into the model and what came out.

    The single measurement that separates "the model made no change" from "the model changed it and
    we threw the change away somewhere downstream". Reported as a percentage of full scale, so ~0.5%
    is encode noise and anything under about 1% means nothing meaningful happened.
    """
    from PIL import ImageChops, ImageStat

    pairs = min(len(source_frames), len(edited_frames))
    if not pairs:
        return 0.0, 0.0
    deltas = []
    for index in range(pairs):
        before = source_frames[index].convert("RGB")
        after = edited_frames[index].convert("RGB")
        if after.size != before.size:
            after = after.resize(before.size)
        channels = ImageStat.Stat(ImageChops.difference(before, after)).mean
        deltas.append(sum(channels) / (len(channels) * 255.0) * 100.0)
    return sum(deltas) / len(deltas), max(deltas)


def _snap_down(frames: int) -> int:
    """Largest valid 4n+1 count not exceeding `frames`."""
    return (frames - TEMPORAL_REMAINDER) // TEMPORAL_DIVISOR * TEMPORAL_DIVISOR + TEMPORAL_REMAINDER


def _snap_up(frames: int) -> int:
    """Smallest valid 4n+1 count not below ``frames``."""
    return int(math.ceil((frames - TEMPORAL_REMAINDER) / TEMPORAL_DIVISOR)) * TEMPORAL_DIVISOR + TEMPORAL_REMAINDER


def build_windows(requested: int, strategy: str, overlap_frames: int = LONG_OVERLAP_FRAMES) -> list:
    """Plan model windows without losing tail frames.

    Stable mode deliberately refuses ranges beyond the trained 81-frame horizon. Long mode uses
    overlapping windows and pads only the final model input; the merged output is trimmed back to
    the exact decoded CFR frame count.
    """
    minimum = TEMPORAL_DIVISOR + TEMPORAL_REMAINDER
    if requested < minimum:
        raise ValueError(
            f"the selected range is only {requested} frames; Kiwi-Edit needs at least {minimum}"
        )
    if strategy == "stable":
        if requested > MAX_FRAMES:
            maximum = MAX_FRAMES / WORKING_FPS
            raise ValueError(
                f"Stable mode supports at most {MAX_FRAMES} frames ({maximum:.2f}s at {WORKING_FPS:g} FPS); "
                "shorten the range or select Long edit"
            )
        return [{
            "startFrame": 0,
            "sourceFrames": requested,
            "modelFrames": max(minimum, _snap_up(requested)),
            "overlapFrames": 0,
        }]

    stride = MAX_FRAMES - overlap_frames
    windows = []
    start = 0
    while start < requested:
        source_frames = min(MAX_FRAMES, requested - start)
        model_frames = min(MAX_FRAMES, max(minimum, _snap_up(source_frames)))
        previous_end = 0 if not windows else windows[-1]["startFrame"] + windows[-1]["sourceFrames"]
        windows.append({
            "startFrame": start,
            "sourceFrames": source_frames,
            "modelFrames": model_frames,
            "overlapFrames": max(0, previous_end - start),
        })
        if start + source_frames >= requested:
            break
        start += stride
    return windows


def extract_range(args: argparse.Namespace, layout: dict, destination: Path) -> list:
    """Accurately decode the selected range once and conform VFR/CFR inputs to one CFR sequence."""
    destination.mkdir(parents=True, exist_ok=True)
    for stale in destination.glob("frame-*.png"):
        stale.unlink()
    duration = max(1.0 / layout["fps"], layout["endTime"] - layout["startTime"])
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-i", args.video_path,
            "-ss", f"{layout['startTime']:.6f}", "-t", f"{duration:.6f}",
            "-vf", f"fps=fps={layout['fps']:.6f}:round=near,scale={layout['width']}:{layout['height']}:flags=bicubic,setsar=1",
            "-an", str(destination / "frame-%05d.png"),
        ],
        check=True,
    )
    frames = sorted(destination.glob("frame-*.png"))
    if not frames:
        raise RuntimeError("ffmpeg produced no frames for the selected range")
    if len(frames) < TEMPORAL_DIVISOR + TEMPORAL_REMAINDER:
        raise RuntimeError(f"the selected range decoded to only {len(frames)} working frames; Kiwi-Edit needs at least 5")
    return frames


def encode_video(frames_dir: Path, args: argparse.Namespace, layout: dict) -> None:
    """Muxes the edited frames back to MP4 and carries the original audio for the range across."""
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    duration = layout["renderedFrames"] / layout["fps"]
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-framerate", f"{layout['fps']:.6f}", "-i", str(frames_dir / "edited-%05d.png"),
        "-ss", f"{layout['startTime']:.6f}", "-t", f"{duration:.6f}", "-i", args.video_path,
        "-map", "0:v:0", "-map", "1:a:0?",
        "-c:v", "libx264", "-preset", "medium", "-crf", "16", "-pix_fmt", "yuv420p",
    ]
    if layout.get("hasAudio"):
        command.extend(["-af", f"atrim=duration={duration:.6f},apad=pad_dur={duration:.6f}", "-c:a", "aac", "-b:a", "192k"])
    command.extend(["-t", f"{duration:.6f}", "-movflags", "+faststart", args.output_path])
    subprocess.run(command, check=True)


def load_pipeline(args: argparse.Namespace, torch):
    """The pack declares `"_class_name": ["pipeline_kiwi_edit", "KiwiEditPipeline"]`, and its VAE
    and encoder are custom classes too, so the snapshot loads as trusted remote code. Diffusers has
    moved this API around between releases, so try the documented spellings in order and report
    each failure rather than dying on the first one."""
    from diffusers import DiffusionPipeline

    # trust_remote_code is mandatory for this pack, not optional: the pipeline, the Wan VAE and
    # the Qwen encoder are all custom classes stored inside the snapshot. Only the dtype keyword
    # has actually moved between releases, so that is the only axis worth retrying.
    attempts = [
        {"torch_dtype": torch.bfloat16, "trust_remote_code": True},
        {"dtype": torch.bfloat16, "trust_remote_code": True},
        {"trust_remote_code": True},
    ]
    failures = []
    for kwargs in attempts:
        try:
            return DiffusionPipeline.from_pretrained(args.model_root, **kwargs)
        except Exception as error:  # noqa: BLE001 - a later spelling may still work
            failures.append(error)
            print(f"[kiwi-edit] load via {sorted(kwargs)} failed: {error}", file=sys.stderr)
    # Report the first failure: that is the documented path, so its error is the real diagnosis.
    # Later attempts only vary the dtype keyword and tend to fail with less useful messages.
    raise RuntimeError(f"could not load the Kiwi-Edit pipeline: {failures[0]}")


# Everything the denoising half of the pipeline needs. The encoder is staged separately because
# it runs exactly once, before any of these are touched.
DENOISE_COMPONENTS = ("transformer", "vae", "source_embedder", "ref_embedder")

# Submodules diffusers deliberately keeps in fp32 that the pack's custom forward calls directly,
# with a bfloat16 tensor, bypassing the cast diffusers would normally do for them.
FP32_MODULES_CALLED_DIRECTLY = ("condition_embedder.time_embedder",)


def _resolve_attribute(root, path: str):
    node = root
    for attribute in path.split("."):
        node = getattr(node, attribute, None)
        if node is None:
            return None
    return node


def _bridge_module_dtype(module, torch) -> None:
    """Cast into a module's own weight dtype on the way in, and back on the way out."""
    original_forward = module.forward

    def forward(sample, *args, **kwargs):
        try:
            weight_dtype = next(module.parameters()).dtype
        except StopIteration:
            return original_forward(sample, *args, **kwargs)
        if torch.is_tensor(sample) and sample.dtype != weight_dtype:
            output = original_forward(sample.to(weight_dtype), *args, **kwargs)
            return output.type_as(sample) if torch.is_tensor(output) else output
        return original_forward(sample, *args, **kwargs)

    module.forward = forward


def align_fp32_module_dtypes(pipeline, torch) -> None:
    """Reconcile diffusers' fp32 keep-list with the pack's hand-written transformer forward.

    `WanTransformer3DModel._keep_in_fp32_modules` is
    `["time_embedder", "scale_shift_table", "norm1", "norm2", "norm3"]`, so loading with
    `torch_dtype=bfloat16` still leaves those in fp32 for numerical stability. diffusers casts
    around them inside its own forwards -- `WanTimeTextImageEmbedding.forward` does
    `timestep.to(time_embedder_dtype)` then `.type_as(encoder_hidden_states)`.

    The pack reimplements that forward and calls `t.condition_embedder.time_embedder(...)` with a
    bfloat16 tensor, so the cast never happens and the matmul fails with "mat1 and mat2 must have
    the same dtype, but got BFloat16 and Float". Reproducing diffusers' cast at the module boundary
    fixes it without flattening the model to one dtype and losing the stability the keep-list buys.

    The transformer blocks need no help: their forward is diffusers' own and already does
    `self.norm1(hidden_states.float()) ... .type_as(hidden_states)`.
    """
    transformer = getattr(pipeline, "transformer", None)
    if transformer is None:
        return
    for path in FP32_MODULES_CALLED_DIRECTLY:
        module = _resolve_attribute(transformer, path)
        if module is None:
            continue
        try:
            weight_dtype = next(module.parameters()).dtype
        except StopIteration:
            continue
        _bridge_module_dtype(module, torch)
        print(f"[kiwi-edit] dtype bridge on transformer.{path} ({weight_dtype})", file=sys.stderr)

# The transformer is 30 blocks of ~0.31 GiB. Streaming two thirds of them frees ~6 GiB at the cost
# of one host-to-device copy per block per step.
DEFAULT_SWAP_BLOCKS = 20


def _move(pipeline, names, device) -> None:
    for name in names:
        component = getattr(pipeline, name, None)
        if component is not None:
            component.to(device)


def enable_block_swap(pipeline, swap_count: int, torch) -> int:
    """Stream the first N transformer blocks from the CPU, the technique ComfyUI's
    WanVideoWrapper calls block swapping.

    This works where accelerate's offload does not because `_model_forward` runs
    `for block in t.blocks:` -- an ordinary module call, so forward hooks actually fire. The
    top-level `transformer.forward` is never invoked, which is why hooking *it* achieves nothing.
    """
    transformer = getattr(pipeline, "transformer", None)
    blocks = getattr(transformer, "blocks", None)
    if blocks is None or swap_count <= 0:
        return 0
    device = torch.device("cuda")
    swap_count = min(swap_count, len(blocks))

    def onload(module, _args):
        module.to(device, non_blocking=True)

    def offload(module, _args, output):
        module.to("cpu", non_blocking=True)
        return output

    for block in list(blocks)[:swap_count]:
        block.to("cpu")
        block.register_forward_pre_hook(onload)
        block.register_forward_hook(offload)
    print(
        f"[kiwi-edit] block swap: streaming {swap_count} of {len(blocks)} transformer blocks",
        file=sys.stderr,
    )
    return swap_count


def stage_components(pipeline, mode: str, torch) -> None:
    """Hold one model group on the GPU at a time, the way ComfyUI stages a text encoder and a
    sampler.

    Neither of accelerate's helpers works with this pack:

    * `enable_sequential_cpu_offload` swaps parameters for meta tensors and materialises them from
      a hook on `forward`. The encoder reads `self.ref_queries` (an `nn.Parameter`) directly inside
      its own forward, so it gets a meta tensor and dies with "Cannot copy out of meta tensor".
    * `enable_model_cpu_offload` keeps real tensors but also hooks `Module.forward`, and this
      pipeline calls submodules directly -- `transformer.patch_embedding(...)`, `vae.encode(...)` --
      so the hook never fires and the weights never reach the GPU.

    Staging sidesteps both, and beats pinning the encoder to the CPU because peak usage becomes the
    largest *stage* rather than the sum of the parts: encoder 7.1 GiB, then transformer + VAE
    11.9 GiB, against 19.0 GiB for everything at once. Both stages run at full GPU speed.
    """
    device = torch.device("cuda")
    encoder = getattr(pipeline, "mllm_encoder", None)

    if mode == "performance":
        _move(pipeline, DENOISE_COMPONENTS, device)
        if encoder is not None:
            encoder.to(device)
        print("[kiwi-edit] all components resident on GPU", file=sys.stderr)
    else:
        # Start with nothing on the accelerator; the encoder's wrapper drives the handover.
        _move(pipeline, DENOISE_COMPONENTS, "cpu")
        if encoder is not None:
            encoder.to("cpu")
            _stage_around_encoder(pipeline, encoder, device, torch)
        else:
            _move(pipeline, DENOISE_COMPONENTS, device)
        print("[kiwi-edit] staged placement: encoder first, then transformer and VAE", file=sys.stderr)

    # `_execution_device` walks `self.components`, whose names come from an unordered set, so while
    # components sit on different devices it can legitimately return the CPU one. Everything the
    # pipeline allocates keys off this, so pin it to the accelerator.
    type(pipeline)._execution_device = property(lambda _self: device)


def _stage_around_encoder(pipeline, encoder, device, torch) -> None:
    """Wrap the encoder's forward so the handover happens at exactly the right moment.

    The pipeline runs MLLM encoding at step 3, before it allocates noise or touches the VAE, so
    the encoder can own the GPU alone and hand it over on the way out.
    """
    original_forward = encoder.forward

    def staged_forward(*args, **kwargs):
        # Re-entrant: on a second call the denoising set is already resident, so evict it first.
        _move(pipeline, DENOISE_COMPONENTS, "cpu")
        torch.cuda.empty_cache()

        encoder.to(device)
        try:
            context = original_forward(*args, **kwargs)
        finally:
            # Free the encoder before the transformer needs the room. This is the whole point.
            encoder.to("cpu")
            torch.cuda.empty_cache()

        _move(pipeline, DENOISE_COMPONENTS, device)
        return context.to(device) if torch.is_tensor(context) else context

    encoder.forward = staged_forward


def build_pipeline(args: argparse.Namespace, torch):
    """Load, dtype-correct and place the pipeline once, for reuse across every window."""
    pipeline = load_pipeline(args, torch)
    align_fp32_module_dtypes(pipeline, torch)
    stage_components(pipeline, args.offload, torch)
    if args.offload == "minimal":
        enable_block_swap(pipeline, DEFAULT_SWAP_BLOCKS, torch)
    if hasattr(pipeline, "vae"):
        for toggle in ("enable_slicing", "enable_tiling"):
            method = getattr(pipeline.vae, toggle, None)
            if method is not None:
                method()
    return pipeline


def edit_window(pipeline, args, layout, source_video, torch):
    """Run one window through the pipeline, with the block-swap fallback on OOM."""
    from PIL import Image  # noqa: F401 - imported by the caller's frame loading

    ref_image = None
    if args.mode == "reference" and args.reference_image:
        from PIL import Image as _Image

        ref_image = [_Image.open(args.reference_image).convert("RGB")]

    call_kwargs = {
        "prompt": args.instruction,
        "negative_prompt": args.negative_prompt,
        "guidance_scale": args.guidance_scale,
        "source_video": source_video,
        "ref_image": ref_image,
        "height": layout["height"],
        "width": layout["width"],
        "num_frames": len(source_video),
        "num_inference_steps": args.steps,
        "seed": args.seed,
        "tiled": True,
        "output_type": "pil",
    }
    try:
        edited = pipeline(**call_kwargs)
    except torch.cuda.OutOfMemoryError:
        # Denoising needs more headroom than this card has left. Rather than failing, fall back to
        # streaming transformer blocks -- the same trade ComfyUI users make by hand with a block
        # swap node. Slower per step, but it completes.
        if args.offload == "minimal":
            raise
        swapped = enable_block_swap(pipeline, DEFAULT_SWAP_BLOCKS, torch)
        if not swapped:
            raise
        torch.cuda.empty_cache()
        print("[kiwi-edit] retrying with block swap after running out of VRAM", file=sys.stderr)
        edited = pipeline(**call_kwargs)

    # The pipeline returns a plain list of PIL frames; tolerate a diffusers-style output too.
    if hasattr(edited, "frames"):
        edited = edited.frames[0]
    if not edited:
        raise RuntimeError("the pipeline returned no frames")
    return edited


def _frame_difference(left, right) -> float:
    """Cheap perceptual seam score on low-resolution luminance."""
    from PIL import ImageChops, ImageStat

    size = (96, 54)
    a = left.convert("L").resize(size)
    b = right.convert("L").resize(size)
    return float(ImageStat.Stat(ImageChops.difference(a, b)).mean[0])


def merge_overlapping_frames(existing: list, incoming: list, overlap: int, blend_radius: int = 3) -> tuple[list, dict]:
    """Choose the least-disruptive overlap seam and softly blend only around that cut."""
    from PIL import Image

    overlap = min(overlap, len(existing), len(incoming))
    if overlap <= 0:
        return [*existing, *incoming], {"overlap": 0, "seam": 0, "score": 0.0}
    old = existing[-overlap:]
    new = incoming[:overlap]
    margin = min(4, max(1, overlap // 5))
    candidates = range(margin, max(margin + 1, overlap - margin))
    scores = [(index, _frame_difference(old[index], new[index])) for index in candidates]
    seam, score = min(scores, key=lambda item: item[1])
    blend_start = max(0, seam - blend_radius)
    blend_end = min(overlap, seam + blend_radius + 1)
    merged_overlap = []
    for index in range(overlap):
        if index < blend_start:
            merged_overlap.append(old[index])
        elif index >= blend_end:
            merged_overlap.append(new[index])
        else:
            alpha = (index - blend_start + 1) / (blend_end - blend_start + 1)
            merged_overlap.append(Image.blend(old[index], new[index], alpha))
    return [*existing[:-overlap], *merged_overlap, *incoming[overlap:]], {
        "overlap": overlap, "seam": seam, "score": round(score, 4),
    }


def _load_window_frames(frame_paths: list, window: dict, continuity_frames: list | None = None) -> list:
    from PIL import Image

    start = window["startFrame"]
    count = window["sourceFrames"]
    source = [Image.open(path).convert("RGB") for path in frame_paths[start:start + count]]
    if not source:
        raise RuntimeError(f"window at frame {start} has no source frames")
    overlap = min(window.get("overlapFrames", 0), len(source), len(continuity_frames or []))
    if overlap:
        source[:overlap] = [frame.copy() for frame in continuity_frames[-overlap:]]
    while len(source) < window["modelFrames"]:
        source.append(source[-1].copy())
    return source


def run_inference(args: argparse.Namespace, layout: dict, frames_dir: Path) -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("the Kiwi-Edit profile requires a CUDA GPU")

    decoded_dir = frames_dir / "decoded"
    frame_paths = extract_range(args, layout, decoded_dir)
    layout["decodedFrames"] = len(frame_paths)
    layout["frames"] = len(frame_paths)
    layout["windows"] = build_windows(len(frame_paths), args.strategy, args.overlap_frames)
    windows = layout["windows"]

    pipeline = build_pipeline(args, torch)
    merged = []
    seams = []

    for index, window in enumerate(windows, start=1):
        continuity = merged if args.strategy == "long" else None
        print(
            f"[kiwi-edit] window {index}/{len(windows)} · {window['modelFrames']} model frames "
            f"(from frame {window['startFrame']})",
            file=sys.stderr,
        )
        source_video = _load_window_frames(frame_paths, window, continuity)
        edited = edit_window(pipeline, args, layout, source_video, torch)
        edited = list(edited)[:window["modelFrames"]]
        if args.diagnostics:
            mean_change, peak_change = _window_change(source_video, edited)
            print(
                f"[kiwi-edit][change] window {index}: mean {mean_change:.2f}% peak {peak_change:.2f}% "
                f"of full scale versus the frames that went in",
                file=sys.stderr,
            )
            if mean_change < 1.0:
                print(
                    "[kiwi-edit][change] Under 1% is indistinguishable from a re-encode. The model "
                    "returned its own input, so nothing downstream can be at fault. With "
                    f"--guidance-scale {args.guidance_scale:g} "
                    + ("the instruction only conditions and never steers -- raise it above 1.0."
                       if args.guidance_scale == 1.0 else "guidance is on, so the instruction itself "
                       "is not moving this content; try a blunter one or more steps."),
                    file=sys.stderr,
                )
        if index == 1:
            merged = edited
        else:
            merged, seam = merge_overlapping_frames(merged, edited, window["overlapFrames"])
            seam["window"] = index
            seams.append(seam)
            print(
                f"[kiwi-edit] seam {index - 1}: overlap={seam['overlap']} "
                f"cut={seam['seam']} score={seam['score']}",
                file=sys.stderr,
            )
        del source_video, edited
        torch.cuda.empty_cache()

    merged = merged[:len(frame_paths)]
    if not merged:
        raise RuntimeError("no windows produced any frames")
    for number, frame in enumerate(merged, start=1):
        frame.save(frames_dir / f"edited-{number:05d}.png")
    layout["renderedFrames"] = len(merged)
    layout["seams"] = seams

    del pipeline
    torch.cuda.empty_cache()


def main() -> int:
    force_utf8_io()
    args = build_parser().parse_args()
    normalise_placement(args)
    validate(args)
    check_pack_guidance(args)
    source = probe(args.video_path)
    layout = plan(source, args)
    if args.dry_run:
        print(json.dumps({"ok": True, "mode": args.mode, "offload": args.offload, "layout": layout}))
        return 0
    require_dependencies()
    warn_on_low_memory()
    print(
        f"[kiwi-edit] {layout['width']}x{layout['height']} · {layout['frames']} frames in "
        f"{len(layout['windows'])} window(s) · {args.steps} steps · {args.mode} · {args.offload}",
        file=sys.stderr,
    )
    # The range is the one thing that differs between a whole-clip edit and a segment edit, so state
    # it plainly rather than leaving "did it even cut the right piece?" as an open question.
    print(
        f"[kiwi-edit] source range {layout['startTime']:.3f}s – {layout['endTime']:.3f}s "
        f"({layout['requestedDuration']:.3f}s of a {source['duration']:.3f}s file) · guidance "
        + (f"{args.guidance_scale:g}" if args.guidance_scale != 1.0
           else "OFF (1.0) — the instruction conditions but never steers"),
        file=sys.stderr,
    )
    with tempfile.TemporaryDirectory(prefix="aiv-kiwi-") as scratch:
        frames_dir = Path(scratch)
        import torch

        with torch.inference_mode():
            run_inference(args, layout, frames_dir)
        encode_video(frames_dir, args, layout)
    print(json.dumps({"ok": True, "outputPath": args.output_path, "mode": args.mode, "layout": layout}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
