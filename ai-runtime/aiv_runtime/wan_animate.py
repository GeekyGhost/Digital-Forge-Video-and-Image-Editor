"""Wan-Animate-2 character worker (VFX Phase 5): one character image + a driving clip -> a video of
that character performing the clip's motion, framed exactly like the (cropped) driving clip.

Runs in its own venv (`vendor/Wan-Animate/.venv`) with Diffusers 0.40.0, whose Animate-2 is a MODULAR
pipeline (`WanAnimate2DistilledBlocks`), not the classic `WanAnimate2Pipeline` the model card's
`model_index.json` still names. Weights: `models/Wan-Animate-2/distilled` (Wan-AI/Wan2.2-Animate-2-14B-
Distilled-Diffusers at the revision pinned in `scripts/runtime-matrix.psd1`; Apache-2.0).

Animate-2 is "Move" mode only: it generates the background from the prompt; there is no mask or
background input (that was Wan 2.2 Animate's replacement mode). So insertion into a plate is:
Animate-2 on a plain background -> SAM 2 + BiRefNet matte -> composite in the editor.

The 16 GB pattern (copied from the Spritesheet Maker's runtime, nothing shared):
- the 14B DiT is baked once to FP8 shards (`bake`), loaded with BF16 compute via layerwise casting
  and streamed to the GPU one block group at a time;
- CLIP and the VAE use synchronous leaf-level offload, so they never sit beside the DiT;
- the reference K/V cache lives in pinned CPU memory as INT8 (`wan_animate_cache`);
- FlexAttention itself is compiled once, outside the offload hooks (compiling whole blocks captures
  each block's hook closure, Dynamo hits its recompile limit, and the dense eager fallback
  materialises a ~15 GiB score matrix at 512 px).

Two stages, each its own process (process exit is the only reliable way to hand commit back on
Windows): `encode` (UMT5-XXL, 11 GB bf16 -> prompt embeddings) and `animate` (everything else).
`generate` runs both and writes `frames/00000.png ...`, `animate.mp4` and `stats.json`.

Geometry, which the editor relies on to lay the result over the plate:
- the output size W x H comes from the plate's aspect and an area (`plan_size`), multiples of 16;
- the plate is centre-cropped to exactly W:H (`plate_crop`, at most a few pixels) and scaled to W x H,
  so upstream's letterbox is the identity and output frame i covers that crop of plate frame i;
- the character image is padded (not stretched) to W x H with its own border colour;
- driving frames are used as-is (`driving_video_fps=None`): no resampling, frame i <-> frame i.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODELS = PROJECT_ROOT / "models" / "Wan-Animate-2"
REPO_ID = "Wan-AI/Wan2.2-Animate-2-14B-Distilled-Diffusers"
LICENCE = "Wan2.2-Animate-2-14B-Distilled: Apache-2.0 (Wan-AI / Alibaba)"
MOD = 16  # VAE stride 8 x patch 2
DEFAULT_AREA = 512 * 512
#: The reference prompt upstream uses for the driving-video context (its default, verbatim).
PROMPT_REF = "人物动作的参考视频"
#: Appended so the matte has an easy job: Animate-2 invents the background from the prompt.
PLAIN_BACKGROUND = "plain flat neutral grey studio background, even lighting, no props"
MANIFEST = "wan-animate-bake.json"
BAKE_SCHEMA = 2  # 2: Diffusers names from the official layout; fp32 kept where upstream keeps it


def emit(event: str, **fields) -> None:
    """One JSON object per line on stdout; the Rust side parses these for the status line."""
    print(json.dumps({"event": event, **fields}), flush=True)


def log(message: str) -> None:
    print(f"[wan-animate] {message}", file=sys.stderr, flush=True)


def paths(models: Path) -> dict:
    distilled = models / "distilled"
    return {
        "repo": distilled,
        "transformer": distilled / "transformer",
        "text_encoder": distilled / "text_encoder",
        "tokenizer": distilled / "tokenizer",
        "image_encoder": distilled / "image_encoder",
        "vae": distilled / "vae",
        "scheduler": distilled / "scheduler",
        "baked": models / "baked",
    }


# ------------------------------------------------------------------------------------------------
# Geometry (pure; the editor places the result with the numbers these return, via stats.json)
# ------------------------------------------------------------------------------------------------

def plan_size(plate_width: int, plate_height: int, area: int = DEFAULT_AREA) -> tuple[int, int]:
    """The largest W x H, both multiples of 16, with the plate's aspect as closely as 16 px allows and
    W * H <= area. Upstream's own sizing does the same from the reference image's aspect."""
    if plate_width <= 0 or plate_height <= 0:
        raise ValueError("the plate has no size")
    area = max(256 * 256, min(1024 * 1024, int(area)))
    aspect = plate_width / plate_height
    best = None
    for height in range(MOD * 8, 1024 + 1, MOD):
        exact = height * aspect / MOD
        for width in {math.floor(exact) * MOD, math.ceil(exact) * MOD}:
            # Within one 16 px step of the aspect (the plate crop takes up the difference).
            if width < MOD * 8 or width > 1280 or width * height > area or abs(width - height * aspect) >= MOD:
                continue
            key = (width * height, -abs(width / height - aspect))
            if best is None or key > best[0]:
                best = (key, (width, height))
    if best is None:
        raise ValueError(f"no size fits {plate_width}x{plate_height} in {area} pixels")
    return best[1]


def plate_crop(plate_width: int, plate_height: int, width: int, height: int) -> tuple[int, int, int, int]:
    """(x, y, w, h): the centred region of the plate with exactly the aspect width:height. The output
    frame covers this region; the few pixels outside it are the only part of the plate not driven."""
    target = width / height
    if plate_width / plate_height > target:
        w = int(round(plate_height * target))
        h = plate_height
    else:
        w = plate_width
        h = int(round(plate_width / target))
    w, h = min(w, plate_width), min(h, plate_height)
    return (plate_width - w) // 2, (plate_height - h) // 2, w, h


def upstream_resolved_size(image_width: int, image_height: int, height_in: int, width_in: int) -> tuple[int, int]:
    """What Diffusers 0.40's `WanAnimate2ProcessImagesInputStep` resolves (height, width) to, verbatim
    arithmetic: the image is first rounded to multiples of 16 (`get_default_height_width`), then the
    requested area is re-shaped to its aspect and floored to multiples of 16."""
    mod = MOD
    max_area = image_width * image_height
    aspect = image_height / image_width
    ih = round(math.sqrt(max_area * aspect)) // mod * mod
    iw = round(math.sqrt(max_area / aspect)) // mod * mod
    aspect_ratio = ih / iw
    area = height_in * width_in
    return int(math.sqrt(area * aspect_ratio)) // mod * mod, int(math.sqrt(area / aspect_ratio)) // mod * mod


def area_inputs(width: int, height: int) -> tuple[int, int]:
    """(height, width) to pass so upstream resolves a W x H reference to exactly W x H. Passing W x H
    itself can floor to 16 px less when sqrt(W*H*H/W) lands a hair under H in floating point."""
    for dh, dw in ((0, 0), (0, 1), (1, 0), (1, 1), (2, 2), (4, 4)):
        if upstream_resolved_size(width, height, height + dh, width + dw) == (height, width):
            return height + dh, width + dw
    raise ValueError(f"cannot make the pipeline resolve to {width}x{height}")


def segment_length(width: int, height: int, frames: int, requested: int = 0) -> int:
    """Frames per inference segment (4n+1). The Spritesheet Maker measured 33 at <= 512^2 and 17 above
    as the 16 GB fit; the reference cache and the sparse-attention workspace grow with it."""
    if requested:
        if requested < 5 or (requested - 1) % 4:
            raise ValueError("the segment length must be 4n+1, at least 5")
        return requested
    cap = 33 if width * height <= 512 * 512 else 17
    return max(5, min(cap, 4 * max(1, math.ceil((frames - 1) / 4)) + 1))


def border_colour(image) -> tuple[int, int, int]:
    """The median colour of the image's outermost pixels: the padding blends into its background."""
    import numpy as np

    pixels = np.asarray(image.convert("RGB"))
    border = np.concatenate([pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]])
    return tuple(int(value) for value in np.median(border, axis=0))


def pad_reference(image, width: int, height: int):
    """The character image fitted inside W x H without stretching, centred, the rest filled by
    extending its own backdrop (upstream letterboxes in black; a flat fill leaves a visible square)."""
    from PIL import Image

    image = image.convert("RGB")
    scale = min(width / image.width, height / image.height)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return extend_backdrop(image.resize(size, Image.Resampling.LANCZOS), (width - size[0]) // 2, (height - size[1]) // 2, width, height)


def _backdrop_difference(image):
    """Per-pixel difference from the reference's backdrop, modelled per row as a straight blend between
    the row's left and right edge colours (median of the outer 3% of columns). A studio backdrop has a
    lighting gradient; one border colour for the whole picture counted the darker side as character
    (second 4090 run, 2026-09-29: bounds x 0..709 for a soldier standing in the middle)."""
    import numpy as np

    pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
    height, width, _ = pixels.shape
    k = max(2, int(width * 0.03))
    left = np.median(pixels[:, :k], axis=1)
    right = np.median(pixels[:, -k:], axis=1)
    t = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :, None]
    backdrop = left[:, None, :] * (1 - t) + right[:, None, :] * t
    return np.abs(pixels - backdrop).max(axis=2)


def character_bounds(image, threshold: int = 40) -> tuple[int, int, int, int] | None:
    """(x0, y0, x1, y1), exclusive ends: the character in a reference shot on a plain or graded studio
    backdrop -- pixels differing from the row's backdrop model by more than `threshold`, keeping rows
    and columns where at least 1% of the line differs. None when nothing stands out."""
    import numpy as np

    differs = _backdrop_difference(image) > threshold
    height, width = differs.shape
    rows = np.flatnonzero(differs.sum(axis=1) > max(1, width // 100))
    cols = np.flatnonzero(differs.sum(axis=0) > max(1, height // 100))
    if not len(rows) or not len(cols):
        return None
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def character_centre_x(image, bounds, threshold: int = 40) -> float:
    """The centroid x of the character's pixels inside its bounds (an arm held out moves a box centre)."""
    import numpy as np

    x0, y0, x1, y1 = bounds
    differs = _backdrop_difference(image)[y0:y1, x0:x1] > threshold
    xs = np.nonzero(differs)[1]
    return float(x0 + xs.mean() + 0.5) if len(xs) else (x0 + x1) / 2


def extend_backdrop(image, left: int, top: int, width: int, height: int):
    """`image` placed at (left, top) on a width x height canvas, the rest filled by extending the
    image's own edge pixels outward. No hard rectangle: pasting a lit studio shot onto one flat colour
    left a visible square that Animate-2 reproduced as a grey panel in the scene, which SAM 2 then
    tracked with the character (second 4090 run)."""
    import numpy as np
    from PIL import Image

    pixels = np.asarray(image.convert("RGB"))
    ih, iw = pixels.shape[:2]
    # Crop what falls outside the canvas, then edge-pad what the image does not cover.
    x0, y0 = max(0, -left), max(0, -top)
    x1, y1 = min(iw, width - left), min(ih, height - top)
    if x1 <= x0 or y1 <= y0:
        return Image.new("RGB", (width, height), border_colour(image))
    inner = pixels[y0:y1, x0:x1]
    pad_left, pad_top = max(0, left), max(0, top)
    pad_right, pad_bottom = width - pad_left - inner.shape[1], height - pad_top - inner.shape[0]
    extended = np.pad(inner, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode="edge")
    return Image.fromarray(extended, mode="RGB")


def region_crop(plate_width: int, plate_height: int, width: int, height: int, region, margin: float = 0.12) -> tuple[int, int, int, int]:
    """The crop of the plate to drive with: the smallest W:H box holding `region` (x0, y0, x1, y1
    normalised to the plate: the actor's union box over the range) grown by `margin` of its height on
    every side, moved inside the plate, and never larger than the whole-plate crop. Driving with the
    actor's neighbourhood instead of the whole plate puts most of the output's pixels on the character
    (second 4090 run: a soldier shrunk to 213 of 384 rows lost his face detail)."""
    full = plate_crop(plate_width, plate_height, width, height)
    x0, y0, x1, y1 = region
    rx0, ry0, rx1, ry1 = x0 * plate_width, y0 * plate_height, x1 * plate_width, y1 * plate_height
    pad = margin * max(1.0, ry1 - ry0)
    rx0, ry0, rx1, ry1 = rx0 - pad, ry0 - pad, rx1 + pad, ry1 + pad
    aspect = width / height
    h = max(ry1 - ry0, (rx1 - rx0) / aspect, height)  # never upscale the plate more than 1:1 would need
    w = h * aspect
    if w >= full[2] or h >= full[3]:
        return full
    w, h = int(round(w)), int(round(w / aspect))
    cx, cy = (rx0 + rx1) / 2, (ry0 + ry1) / 2
    x = int(round(min(max(cx - w / 2, 0), plate_width - w)))
    y = int(round(min(max(cy - h / 2, 0), plate_height - h)))
    return x, y, w, h


def output_target(actor, crop, width: int, height: int, plate_width: int, plate_height: int) -> tuple[float, float, float]:
    """The actor as (cx, feet_y, standing_height) in OUTPUT pixels, from (cx, feet_y, standing_height)
    normalised to the plate: the matte's centroid x on the first driving frame (on the torso; a box
    centre drifts toward an outstretched leg), the bottom of its box there (the feet line), and the
    tallest box height over the range (standing height; a crouch or stride is shorter)."""
    x, y, w, h = crop
    cx, feet, standing = actor
    return ((cx * plate_width - x) * width / w, (feet * plate_height - y) * height / h, standing * plate_height * height / h)


def frame_reference(image, width: int, height: int, target):
    """The character framed like the actor it replaces. Animate-2 keeps the REFERENCE's framing -- a
    character that fills the reference comes out taller than the actor (first 4090 run, 2026-09-29:
    the soldier's feet landed well below the dancer's). So the character is scaled until its height is
    the actor's standing height and placed with its feet on the actor's feet line, its centroid on the
    actor's; the rest of the canvas extends the reference's own backdrop. `target` is (cx, feet_y,
    standing_height) in output pixels. Falls back to `pad_reference` when the character cannot be found."""
    image = image.convert("RGB")
    bounds = character_bounds(image)
    if bounds is None or target is None:
        return pad_reference(image, width, height), None
    x0, y0, x1, y1 = bounds
    centre = character_centre_x(image, bounds)
    cx, feet, standing = target
    scale = standing / max(1, y1 - y0)
    from PIL import Image

    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    left = round(cx - centre * scale)
    top = round(feet - y1 * scale)
    canvas = extend_backdrop(image.resize(size, Image.Resampling.LANCZOS), left, top, width, height)
    return canvas, {"scale": round(scale, 4), "left": left, "top": top, "characterBounds": list(bounds), "characterCentreX": round(centre, 1)}


def driving_filter(crop: tuple[int, int, int, int], width: int, height: int, first: int, count: int) -> str:
    """ffmpeg filter for exactly source frames [first, first + count): frame-exact by decode index
    (`select` on n), then the aspect crop and the scale to W x H."""
    x, y, w, h = crop
    return (f"select='between(n\\,{first}\\,{first + count - 1})',"
            f"crop={w}:{h}:{x}:{y},scale={width}:{height}:flags=lanczos,format=rgb24")


def extract_driving(video: Path, out_dir: Path, crop, width: int, height: int, first: int, count: int) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-i", str(video),
               "-vf", driving_filter(crop, width, height, first, count), "-fps_mode", "passthrough",
               "-frames:v", str(count), str(out_dir / "%05d.png")]
    subprocess.run(command, check=True)
    frames = sorted(out_dir.glob("*.png"))
    if len(frames) != count:
        raise ValueError(f"the driving clip gave {len(frames)} frames, {count} were asked for (frames {first}..{first + count - 1})")
    return frames


# ------------------------------------------------------------------------------------------------
# Memory accounting (shared shape with the Wan Alpha worker's stats)
# ------------------------------------------------------------------------------------------------

def memory_report() -> dict:
    report: dict = {}
    try:
        from .wan_alpha import MemoryWatch  # noqa: F401  (same venv family; only for the helpers below)
        from .wan_alpha import process_peak_commit, cuda_report, _gb

        report["processPeakCommitGb"] = _gb(process_peak_commit())
        report.update(cuda_report())
    except Exception:  # torch or ctypes unavailable: the stage error already says why
        pass
    return report


# ------------------------------------------------------------------------------------------------
# Bake: BF16 shards -> FP8 shards, one tensor at a time
# ------------------------------------------------------------------------------------------------

class Layout:
    """What the bake must produce, asked of Diffusers itself on a meta-device model so bake and runtime
    cannot disagree: every state-dict name, the ones `enable_layerwise_casting` stores in FP8 (the
    class's own skip patterns), and the ones upstream keeps in float32 under a bf16 load
    (`_keep_in_fp32_modules`, matched per dotted name part exactly as Diffusers' loader does)."""

    def __init__(self, names: set[str], fp8: set[str], keep_fp32: tuple[str, ...]):
        self.names, self.fp8, self.keep_fp32 = names, fp8, keep_fp32

    def dtype_for(self, name: str):
        import torch

        if name in self.fp8:
            return torch.float8_e4m3fn
        return torch.float32 if any(part in name.split(".") for part in self.keep_fp32) else torch.bfloat16


def model_layout(transformer_dir: Path) -> Layout:
    import torch
    from accelerate import init_empty_weights
    from diffusers import WanAnimate2Transformer3DModel

    config = WanAnimate2Transformer3DModel.load_config(str(transformer_dir))
    with init_empty_weights():
        model = WanAnimate2Transformer3DModel.from_config(config)
    names = set(model.state_dict().keys())
    model.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)
    fp8 = {name for name, param in model.named_parameters() if param.dtype == torch.float8_e4m3fn}
    if not fp8:
        raise RuntimeError("layerwise casting selected no parameters; the Diffusers API has changed")
    return Layout(names, fp8, tuple(getattr(WanAnimate2Transformer3DModel, "_keep_in_fp32_modules", None) or ()))


def fp8_parameter_names(transformer_dir: Path) -> set[str]:
    return model_layout(transformer_dir).fp8


def diffusers_names(keys) -> dict[str, str]:
    """{checkpoint name: Diffusers name}. The published revision is still in the official layout
    (`blocks.N.block.self_attn.q.weight`); Diffusers 0.40's own single-file converter for this class,
    `convert_wan_animate_2_transformer_to_diffusers`, is the mapping (asked name by name, no tensors)."""
    from diffusers.loaders.single_file_utils import convert_wan_animate_2_transformer_to_diffusers

    keys = list(keys)
    renamed = convert_wan_animate_2_transformer_to_diffusers({key: key for key in keys})
    mapping = {old: new for new, old in renamed.items()}
    if len(mapping) != len(keys):
        raise RuntimeError(f"the key conversion merged {len(keys) - len(mapping)} names")
    return mapping


def shard_files(transformer_dir: Path) -> list[Path]:
    index = transformer_dir / "diffusion_pytorch_model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        return [transformer_dir / name for name in sorted(set(weight_map.values()))]
    single = transformer_dir / "diffusion_pytorch_model.safetensors"
    if single.is_file():
        return [single]
    raise FileNotFoundError(f"no diffusion_pytorch_model*.safetensors in {transformer_dir}")


def bake(transformer_dir: Path, out_dir: Path, layout: Layout, log=log) -> dict:
    """Stream every checkpoint tensor into FP8/bf16/fp32 shards under its Diffusers name. The renamed
    set must be exactly the model's state dict, or nothing is written."""
    import torch
    from safetensors.torch import save_file

    from .safetensors_reader import describe_backend, open_tensors

    shards = shard_files(transformer_dir)
    keys = []
    for shard in shards:
        with open_tensors(shard) as handle:
            keys.extend(handle.keys())
    mapping = diffusers_names(keys)
    renamed = set(mapping.values())
    missing, extra = sorted(layout.names - renamed), sorted(renamed - layout.names)
    if missing or extra:
        raise RuntimeError(f"the checkpoint does not match WanAnimate2Transformer3DModel: {len(missing)} missing "
                           f"(e.g. {missing[:3]}), {len(extra)} unexpected (e.g. {extra[:3]})")
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": BAKE_SCHEMA, "files": [], "fp8Tensors": 0, "fp32Tensors": 0, "bf16Tensors": 0, "bytes": 0,
                "renamed": sum(1 for old, new in mapping.items() if old != new),
                "sources": [path.name for path in shards], "fp8Dtype": "float8_e4m3fn"}
    for index, shard in enumerate(shards):
        out = {}
        with open_tensors(shard) as handle:
            for name in handle.keys():
                tensor = handle.get_tensor(name)
                target_name = mapping[name]
                dtype = layout.dtype_for(target_name) if tensor.is_floating_point() else tensor.dtype
                out[target_name] = tensor.to(dtype).contiguous()
                key = {torch.float8_e4m3fn: "fp8Tensors", torch.float32: "fp32Tensors"}.get(dtype, "bf16Tensors")
                manifest[key] += 1
                manifest["bytes"] += out[target_name].numel() * out[target_name].element_size()
        target = out_dir / f"wan-animate-2-dit-fp8-{index + 1:05d}.safetensors"
        save_file(out, str(target), metadata={"format": "pt", "aiv": "wan-animate-bake"})
        manifest["files"].append(target.name)
        log(f"bake: wrote {target.name} ({len(out)} tensors)")
        del out
    manifest["readBackend"] = describe_backend()
    (out_dir / MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def baked_files(out_dir: Path) -> list[Path] | None:
    manifest_path = out_dir / MANIFEST
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != BAKE_SCHEMA:
        return None
    files = [out_dir / name for name in manifest.get("files", [])]
    return files if files and all(path.is_file() for path in files) else None


def stage_bake(args) -> dict:
    p = paths(Path(args.models))
    if baked_files(p["baked"]) and not args.force:
        log(f"bake already present in {p['baked']}")
        return {"skipped": True}
    started = time.time()
    layout = model_layout(p["transformer"])
    log(f"{len(layout.fp8)} of {len(layout.names)} tensors will be stored as float8_e4m3fn; fp32 kept for {layout.keep_fp32}")
    manifest = bake(p["transformer"], p["baked"], layout)
    manifest["seconds"] = round(time.time() - started, 1)
    return manifest


# ------------------------------------------------------------------------------------------------
# Stages
# ------------------------------------------------------------------------------------------------

def stage_download(args) -> dict:
    from huggingface_hub import snapshot_download

    models = Path(args.models)
    p = paths(models)
    started = time.time()
    ignore = ["*.md", ".gitattributes"]
    # The BF16 transformer shards (31 GB) are only read by the bake; once it exists they may have
    # been deleted on purpose, so a rerun must not fetch them again unless asked.
    need_shards = args.with_transformer or not baked_files(p["baked"])
    if not need_shards:
        ignore.append("transformer/diffusion_pytorch_model-*.safetensors")
        log("bake present: skipping the BF16 transformer shards (pass --with-transformer to fetch them)")
    snapshot_download(REPO_ID, revision=args.revision, local_dir=p["repo"], ignore_patterns=ignore)
    required = [p["transformer"] / "config.json", p["text_encoder"] / "config.json", p["image_encoder"] / "config.json",
                p["vae"] / "config.json", p["scheduler"] / "scheduler_config.json", p["tokenizer"]]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"downloads finished but these are missing: {missing}")
    return {"seconds": round(time.time() - started, 1), "transformerShards": need_shards, "revision": args.revision}


def stage_encode(args) -> dict:
    import torch
    from safetensors.torch import save_file
    from transformers import AutoTokenizer, UMT5EncoderModel
    from diffusers.modular_pipelines.wan_animate_2.encoders import get_t5_prompt_embeds

    p = paths(Path(args.models))
    device = torch.device("cuda" if torch.cuda.is_available() and args.t5_device != "cpu" else "cpu")
    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(str(p["tokenizer"]))
    encoder = UMT5EncoderModel.from_pretrained(str(p["text_encoder"]), dtype=torch.bfloat16, low_cpu_mem_usage=True).eval()
    encoder.requires_grad_(False).to(device)
    loaded = time.time()
    with torch.no_grad():
        prompt = get_t5_prompt_embeds(encoder, tokenizer, args.prompt, args.max_sequence_length, device)
        reference = get_t5_prompt_embeds(encoder, tokenizer, PROMPT_REF, args.max_sequence_length, device)
    save_file({"prompt_embeds": prompt.to(torch.bfloat16).cpu().contiguous(),
               "prompt_ref_embeds": reference.to(torch.bfloat16).cpu().contiguous()}, args.embeds)
    return {"device": str(device), "loadSeconds": round(loaded - started, 1), "encodeSeconds": round(time.time() - loaded, 2)}


_FLEX_COMPILED = False


def compile_flex_attention() -> bool:
    """Compile FlexAttention itself, once, outside the offload hooks (see the module docstring). Needs
    Triton (triton-windows on Windows); without it the sparse pattern cannot fit on a 16 GB GPU."""
    global _FLEX_COMPILED
    if _FLEX_COMPILED:
        return True
    import importlib

    import torch

    try:
        from torch.utils._triton import has_triton

        ready = has_triton()
    except (ImportError, RuntimeError):
        ready = False
    if not ready:
        return False
    module = importlib.import_module("torch.nn.attention.flex_attention")
    if not getattr(module.flex_attention, "_aiv_compiled", False):
        # Dynamic, because the sequence length changes with the frame count and size: a static graph
        # per shape would use up Dynamo's recompile budget and fall back to the dense eager path.
        compiled = torch.compile(module.flex_attention, fullgraph=True, dynamic=True)
        compiled._aiv_compiled = True
        module.flex_attention = compiled
    _FLEX_COMPILED = True
    return True


def load_transformer(p: dict, device):
    """The baked FP8 DiT with BF16 compute hooks. Every FP8 tensor must be exactly what layerwise
    casting would store in FP8, or the load refuses (bake and runtime disagreeing is silent otherwise)."""
    import torch
    from accelerate import init_empty_weights
    from diffusers import WanAnimate2Transformer3DModel

    from .safetensors_reader import load_state_dict

    files = baked_files(p["baked"])
    if not files:
        raise FileNotFoundError(f"no baked DiT in {p['baked']}; run: wan_animate bake")
    config = WanAnimate2Transformer3DModel.load_config(str(p["transformer"]))
    with init_empty_weights():
        model = WanAnimate2Transformer3DModel.from_config(config)
    state: dict = {}
    for path in files:
        state.update(load_state_dict(path))
    fp8 = {name for name, tensor in state.items() if tensor.dtype == torch.float8_e4m3fn}
    model.load_state_dict(state, strict=True, assign=True)
    del state
    meta = [name for name, param in model.named_parameters() if param.is_meta]
    if meta:
        raise RuntimeError(f"{len(meta)} DiT parameters were not loaded, e.g. {meta[:3]}")
    model.eval().requires_grad_(False)
    model.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16, non_blocking=True)
    wrong = [name for name, param in model.named_parameters() if (name in fp8) != (param.dtype == torch.float8_e4m3fn)]
    if wrong:
        raise RuntimeError(f"baked FP8 layout disagrees with layerwise casting for {len(wrong)} tensors, e.g. {wrong[:3]}")
    return model


def load_scheduler(p: dict):
    """The scheduler class `model_index.json` names (DPMSolverMultistepScheduler at the pinned revision),
    configured from `scheduler/scheduler_config.json` -- exactly what upstream's
    `ModularPipeline.from_pretrained` + `load_components` builds. The folder's own `_class_name` says
    UniPC, whose `step()` has no `generator` argument, and Diffusers 0.40's denoise loop passes one."""
    import diffusers

    index = p["repo"] / "model_index.json"
    name = "DPMSolverMultistepScheduler"
    if index.is_file():
        entry = json.loads(index.read_text(encoding="utf-8")).get("scheduler")
        if isinstance(entry, list) and len(entry) == 2:
            name = entry[1]
    return getattr(diffusers, name).from_pretrained(str(p["scheduler"]))


def build_pipeline(components: dict, cache_codec: str):
    """Diffusers 0.40's distilled modular blocks without the text encoder (its outputs come from the
    encode stage), with the quantised CPU reference cache installed for every segment."""
    from functools import partial

    from diffusers import WanAnimate2DistilledBlocks
    from diffusers.modular_pipelines.wan_animate_2 import denoise as denoise_module

    from .wan_animate_cache import StreamingKVCache

    denoise_module.WanAnimate2KVCache = partial(StreamingKVCache, codec=cache_codec)
    blocks = WanAnimate2DistilledBlocks()
    blocks.sub_blocks.pop("text_encoder")
    pipeline = blocks.init_pipeline()
    pipeline.update_components(**components)
    return pipeline


class StepCounter:
    """Counts denoising forwards (kv_cache_mode "cached") for progress; the modular loop has no callback."""

    def __init__(self, total: int):
        self.total = total
        self.done = 0
        self.started = time.time()

    def __call__(self, module, args, kwargs):
        if kwargs.get("kv_cache_mode") == "cached":
            self.done += 1
            emit("step", step=self.done, total=self.total, seconds=round(time.time() - self.started, 1))


def stage_animate(args) -> dict:
    import numpy as np
    import torch
    from PIL import Image
    from diffusers import AutoencoderKLWan
    from diffusers.hooks import apply_group_offloading
    from transformers import CLIPVisionModel

    from .safetensors_reader import load_state_dict
    from .wan_animate_cache import StreamingKVCache

    cuda_ok = torch.cuda.is_available()
    if not cuda_ok and not args.allow_cpu:
        raise RuntimeError("the Animate-2 stage needs CUDA")
    p = paths(Path(args.models))
    device = torch.device("cuda" if cuda_ok else "cpu")
    out = Path(args.out_dir)
    work = out / "work"
    work.mkdir(parents=True, exist_ok=True)

    width, height = args.width, args.height
    crop = tuple(int(value) for value in args.crop.split(","))
    started = time.time()
    frames = extract_driving(Path(args.driving_video), work / "driving", crop, width, height, args.first_frame, args.frame_count)
    driving = [Image.open(frame).convert("RGB") for frame in frames]
    target = None
    if args.actor_frame:
        actor = [float(value) for value in args.actor_frame.split(",")]
        if len(actor) != 3:
            raise ValueError("--actor-frame is cx,feet,standing normalised to the plate")
        target = output_target(actor, crop, width, height, args.plate_width, args.plate_height)
    reference, framing = frame_reference(Image.open(args.character), width, height, target)
    reference.save(work / "reference.png")
    segment = segment_length(width, height, len(driving), args.segment_frames)
    prepared = time.time()

    flex = compile_flex_attention() if cuda_ok else False
    if cuda_ok and not flex:
        raise RuntimeError("Triton is not available, so FlexAttention cannot be compiled; rerun Setup-Wan-Animate.bat")
    transformer = load_transformer(p, device)
    image_encoder = CLIPVisionModel.from_pretrained(str(p["image_encoder"]), dtype=torch.bfloat16, low_cpu_mem_usage=True).eval()
    # The Wan VAE decodes in float32 upstream (the model card's example loads it so).
    vae = AutoencoderKLWan.from_pretrained(str(p["vae"]), dtype=torch.float32, low_cpu_mem_usage=True).eval()
    if hasattr(vae, "enable_tiling"):
        vae.enable_tiling()
    scheduler = load_scheduler(p)
    if cuda_ok:
        cpu = torch.device("cpu")
        apply_group_offloading(transformer, onload_device=device, offload_device=cpu, offload_type="block_level",
                               num_blocks_per_group=max(1, min(4, args.blocks_per_group)), use_stream=True,
                               record_stream=False, low_cpu_mem_usage=True)
        # Synchronous on Windows: asynchronous leaf streaming pins nearly the whole state dict and can
        # exhaust CUDA's host allocation limit (the Spritesheet Maker found this).
        for component in (image_encoder, vae):
            apply_group_offloading(component, onload_device=device, offload_device=cpu, offload_type="leaf_level", use_stream=False)
    pipeline = build_pipeline({"transformer": transformer, "image_encoder": image_encoder, "vae": vae, "scheduler": scheduler},
                              args.cache)
    loaded = time.time()

    embeds = load_state_dict(args.embeds)
    effective = segment - 1
    segments = max(1, math.ceil((len(driving) - 1) / effective))
    counter = StepCounter(args.steps * segments)
    handle = transformer.register_forward_pre_hook(counter, with_kwargs=True)
    height_in, width_in = area_inputs(width, height)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    emit("animate-start", frames=len(driving), segments=segments, segmentFrames=segment, steps=args.steps, width=width, height=height)
    try:
        with torch.no_grad():
            state = pipeline(
                image=reference, driving_video=driving, driving_video_fps=None,
                prompt_embeds=embeds["prompt_embeds"].to(device), prompt_ref_embeds=embeds["prompt_ref_embeds"].to(device),
                height=height_in, width=width_in, segment_frame_length=segment, prev_segment_conditioning_frames=1,
                num_inference_steps=args.steps, generator=generator, output_type="np",
            )
    finally:
        handle.remove()
    resolved = (state.get("height"), state.get("width"))
    if resolved != (height, width):
        raise RuntimeError(f"the pipeline resolved {resolved[1]}x{resolved[0]}, not the planned {width}x{height}")
    video = state.get("videos")[0]
    video = np.asarray(video)
    if video.shape[0] != len(driving):
        raise RuntimeError(f"the pipeline returned {video.shape[0]} frames for {len(driving)} driving frames")
    denoised = time.time()

    frame_dir = out / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    pixels = (np.clip(video, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    for index, frame in enumerate(pixels):
        Image.fromarray(frame, mode="RGB").save(frame_dir / f"{index:05d}.png", compress_level=1)
    write_mp4(pixels, out / "animate.mp4", args.fps)
    return {
        "frames": int(pixels.shape[0]), "width": width, "height": height, "crop": list(crop),
        "segments": segments, "segmentFrames": segment, "steps": args.steps, "cache": args.cache,
        "cachePeakGb": round(StreamingKVCache.peak_bytes / 1024 ** 3, 2),
        "prepareSeconds": round(prepared - started, 1), "loadSeconds": round(loaded - prepared, 1),
        "denoiseSeconds": round(denoised - loaded, 1), "flexCompiled": flex, "framing": framing,
    }


def write_mp4(pixels, path: Path, fps: float) -> None:
    """The output as H.264 for SAM 2 (its tracker reads a video), BT.709 limited and tagged."""
    height, width = pixels.shape[1:3]
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{width}x{height}", "-r", f"{fps}", "-i", "-", "-c:v", "libx264", "-crf", "12", "-preset", "fast",
               "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
               "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv", str(path)]
    subprocess.run(command, input=pixels.tobytes(), check=True, capture_output=True)


STAGES = {"download": stage_download, "bake": stage_bake, "encode": stage_encode, "animate": stage_animate}


def run_stage(args) -> int:
    started = time.time()
    try:
        from .wan_alpha import MemoryWatch
        watch = MemoryWatch()
    except Exception:
        watch = None
    if watch:
        watch.__enter__()
    try:
        result = STAGES[args.command](args)
        ok, error = True, None
    except Exception as exc:
        import traceback

        traceback.print_exc()
        result, ok, error = {}, False, f"{type(exc).__name__}: {exc}"
    finally:
        if watch:
            watch.__exit__(None, None, None)
    stats = {"stage": args.command, "ok": ok, "error": error, "seconds": round(time.time() - started, 1), **result}
    stats.update(memory_report())
    if watch:
        stats.update({key: value for key, value in watch.report().items() if key != "processPeakCommitGb"})
    if args.stats:
        Path(args.stats).write_text(json.dumps(stats, indent=2), encoding="utf-8")
    emit("stage", **stats)
    return 0 if ok else 1


def generate(args) -> int:
    """bake (if needed) -> encode -> animate, one process each; writes stats.json."""
    out = Path(args.out_dir)
    work = out / "work"
    work.mkdir(parents=True, exist_ok=True)
    width, height = plan_size(args.plate_width, args.plate_height, args.area)
    crop = plate_crop(args.plate_width, args.plate_height, width, height)
    if args.actor_region:
        region = [float(value) for value in args.actor_region.split(",")]
        if len(region) != 4:
            raise ValueError("--actor-region is x0,y0,x1,y1 normalised to the plate")
        crop = region_crop(args.plate_width, args.plate_height, width, height, region)
    prompt = args.prompt.strip()
    prompt = f"{prompt}, {PLAIN_BACKGROUND}" if prompt else PLAIN_BACKGROUND
    common = ["--models", args.models]
    embeds = str(work / "embeds.safetensors")
    plan = []
    if not baked_files(paths(Path(args.models))["baked"]):
        plan.append(["bake", *common])
    plan += [
        ["encode", *common, "--prompt", prompt, "--embeds", embeds, "--t5-device", args.t5_device],
        ["animate", *common, "--embeds", embeds, "--character", args.character, "--driving-video", args.driving_video,
         "--first-frame", str(args.first_frame), "--frame-count", str(args.frame_count),
         "--width", str(width), "--height", str(height), "--crop", ",".join(str(v) for v in crop), "--fps", str(args.fps),
         "--steps", str(args.steps), "--seed", str(args.seed), "--cache", args.cache,
         "--blocks-per-group", str(args.blocks_per_group), "--segment-frames", str(args.segment_frames), "--out-dir", str(out),
         "--plate-width", str(args.plate_width), "--plate-height", str(args.plate_height), "--actor-frame", args.actor_frame],
    ]
    summary = {"prompt": prompt, "plate": [args.plate_width, args.plate_height], "width": width, "height": height,
               "crop": list(crop), "firstFrame": args.first_frame, "frameCount": args.frame_count, "fps": args.fps,
               "steps": args.steps, "seed": args.seed, "cache": args.cache, "licence": LICENCE, "actorFrame": args.actor_frame, "actorRegion": args.actor_region, "stages": []}
    emit("plan", width=width, height=height, crop=list(crop), frames=args.frame_count)
    started = time.time()
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    for stage in plan:
        stats_path = work / f"stage-{stage[0]}.json"
        emit("stage-start", stage=stage[0])
        command = [sys.executable, "-m", "aiv_runtime.wan_animate", *stage, "--stats", str(stats_path)]
        code = subprocess.run(command, env=env, cwd=str(PROJECT_ROOT / "ai-runtime")).returncode
        stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.is_file() else {"stage": stage[0], "ok": False}
        stats["exitCode"] = code
        summary["stages"].append(stats)
        if code != 0:
            summary["error"] = stats.get("error") or f"{stage[0]} exited with {code}"
            break
    summary["seconds"] = round(time.time() - started, 1)
    summary["ok"] = "error" not in summary
    measured = [s for s in summary["stages"] if s.get("stage") != "bake"]
    summary["peakVramReservedGb"] = max((s.get("vramPeakReservedGb") or 0 for s in measured), default=None)
    summary["peakProcessCommitGb"] = max((s.get("processPeakCommitGb") or 0 for s in measured), default=None)
    summary["minFreeCommitGb"] = min((s["systemMinFreeCommitGb"] for s in measured if s.get("systemMinFreeCommitGb") is not None), default=None)
    (out / "stats.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    emit("done", ok=summary["ok"], seconds=summary["seconds"], error=summary.get("error"), outDir=str(out))
    return 0 if summary["ok"] else 1


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = root.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--models", default=str(DEFAULT_MODELS))
        p.add_argument("--stats", default=None)

    download = sub.add_parser("download", help="fetch the pinned weights into models/Wan-Animate-2")
    common(download)
    download.add_argument("--revision", required=True)
    download.add_argument("--with-transformer", action="store_true", help="fetch the BF16 shards even when a bake exists")

    bake_parser = sub.add_parser("bake", help="store the DiT as FP8 shards (once)")
    common(bake_parser)
    bake_parser.add_argument("--force", action="store_true")

    encode = sub.add_parser("encode")
    common(encode)
    encode.add_argument("--prompt", required=True)
    encode.add_argument("--embeds", required=True)
    encode.add_argument("--max-sequence-length", type=int, default=512)
    encode.add_argument("--t5-device", choices=["auto", "cpu"], default="auto")

    def run_options(p):
        p.add_argument("--character", required=True)
        p.add_argument("--driving-video", required=True)
        p.add_argument("--first-frame", type=int, default=0)
        p.add_argument("--frame-count", type=int, required=True)
        p.add_argument("--fps", type=float, default=24.0)
        p.add_argument("--steps", type=int, default=10)
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--cache", choices=["int8", "int4", "bf16"], default="int8")
        p.add_argument("--blocks-per-group", type=int, default=1)
        p.add_argument("--segment-frames", type=int, default=0, help="0 picks 33 at <= 512^2, else 17")
        p.add_argument("--actor-frame", default="", help="cx,feet,standing of the actor, normalised to the plate (see output_target)")
        p.add_argument("--out-dir", required=True)

    animate = sub.add_parser("animate")
    common(animate)
    run_options(animate)
    animate.add_argument("--embeds", required=True)
    animate.add_argument("--width", type=int, required=True)
    animate.add_argument("--height", type=int, required=True)
    animate.add_argument("--crop", required=True, help="x,y,w,h of the plate the output covers")
    animate.add_argument("--allow-cpu", action="store_true", help="tests only")
    animate.add_argument("--plate-width", type=int, default=0)
    animate.add_argument("--plate-height", type=int, default=0)

    gen = sub.add_parser("generate", help="bake if needed, then encode -> animate")
    common(gen)
    run_options(gen)
    gen.add_argument("--prompt", default="")
    gen.add_argument("--plate-width", type=int, required=True)
    gen.add_argument("--plate-height", type=int, required=True)
    gen.add_argument("--area", type=int, default=DEFAULT_AREA)
    gen.add_argument("--actor-region", default="", help="x0,y0,x1,y1 of the actor over the range, normalised to the plate: drive with that neighbourhood")
    gen.add_argument("--t5-device", choices=["auto", "cpu"], default="auto")
    return root


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if args.command == "generate":
        return generate(args)
    return run_stage(args)


if __name__ == "__main__":
    sys.exit(main())
