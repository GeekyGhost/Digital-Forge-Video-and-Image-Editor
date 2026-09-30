"""One-job image worker for ZiT, Qwen Image Edit, and BiRefNet.

The model implementations are intentionally isolated from LTX and MiniMax-H3. Process exit is the
final VRAM reclamation boundary; Qwen also tears down its custom block-streaming hooks explicitly.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
from pathlib import Path
from typing import Any

from PIL import Image, ImageFilter, ImageOps


def emit(event: str, **payload: Any) -> None:
    print(json.dumps({"event": event, **payload}, ensure_ascii=False), flush=True)


def seed_value(seed: int) -> int:
    return random.SystemRandom().randint(0, 2**31 - 1) if seed < 0 else seed


def image(path: str) -> Image.Image:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"image not found: {source}")
    opened = Image.open(source)
    opened.load()
    return opened.convert("RGB")


# The Qwen pipeline resizes every input independently to this area before the
# VAE, so it is the floor for any pre-scaling we do here: shrinking below it only
# throws away detail the pipeline then has to interpolate back.
QWEN_VAE_AREA = 1024 * 1024
# Guard against a source file large enough to hurt host RAM on decode. Well above
# anything the model will actually consume.
QWEN_MAX_AXIS = 4096
# An explicit edit output size (the Image Studio "Output size" override): multiples of 16 (VAE
# factor 8 x the 2x2 latent packing), 256..2048 a side, at most 2048 x 2048 in area. The same
# rule as `snapEditSize` in src/domain/qwenEditSize.ts.
EDIT_SIZE_MIN = 256
EDIT_SIZE_MAX = 2048


def edit_output_size(width: int, height: int) -> tuple[int, int] | None:
    """The override as the edit will make it, or None (follow Image 1) when either side is 0."""
    if width <= 0 or height <= 0:
        return None
    k = min(1.0, math.sqrt(EDIT_SIZE_MAX * EDIT_SIZE_MAX / (width * height)), EDIT_SIZE_MAX / max(width, height))
    snap = lambda value: min(EDIT_SIZE_MAX, max(EDIT_SIZE_MIN, int(round(value * k / 16)) * 16))
    return snap(width), snap(height)


def qwen_inputs(paths: list[str], pixel_budget: int) -> list[Image.Image]:
    """Load the edit inputs, pre-scaling only where it cannot cost quality.

    The budget is applied per image, not to the summed pixels of the set -- a
    shared scale meant a third input shrank the first two for no benefit, since
    the pipeline normalises each one to ~1 MP regardless.
    """
    images = [image(path) for path in paths]
    ceiling = max(QWEN_VAE_AREA, max(1, pixel_budget) // max(1, len(images)))
    resized: list[Image.Image] = []
    for item in images:
        area = max(1, item.width * item.height)
        scale = min(1.0, math.sqrt(ceiling / area), QWEN_MAX_AXIS / max(item.size))
        if scale >= 0.999:
            resized.append(item)
            continue
        width = max(64, int(item.width * scale) // 16 * 16)
        height = max(64, int(item.height * scale) // 16 * 16)
        resized.append(item.resize((width, height), Image.Resampling.LANCZOS))
    if any(before.size != after.size for before, after in zip(images, resized)):
        emit("inputs-resized", originals=[item.size for item in images], modelSizes=[item.size for item in resized])
    return resized


def run_zit(args: argparse.Namespace) -> dict[str, Any]:
    from aiv_runtime.image_studio.zit import ZitImageGenerationPipeline

    turbo = args.model_variant == "turbo"
    model_name = "Z-Image-Turbo" if turbo else "Z-Image Base"
    emit("loading", model=model_name, lora=bool(args.lora_path))
    pipeline = ZitImageGenerationPipeline.create(
        args.model_path,
        "cuda",
        turbo=turbo,
        lora_path=args.lora_path or None,
        lora_scale=args.lora_scale,
    )
    if args.lora_path:
        emit(
            "lora-loaded",
            path=pipeline.lora_path,
            adapters=pipeline.active_adapters,
            strength=args.lora_scale,
            model=model_name,
        )
    requested_outputs = json.loads(args.output_paths_json) if args.output_paths_json else [args.output_path]
    if not isinstance(requested_outputs, list) or not 1 <= len(requested_outputs) <= 4:
        raise ValueError("Z-Image generation requires one to four output paths")
    base_seed = seed_value(args.seed)
    outputs: list[dict[str, Any]] = []
    try:
        for index, output_path in enumerate(requested_outputs):
            used_seed = seed_value(-1) if args.seed < 0 and index > 0 else base_seed + index
            emit("denoise", candidate=index + 1, candidates=len(requested_outputs), step=0, total=args.steps, seed=used_seed)
            result = pipeline.generate(
                prompt=args.prompt,
                height=args.height,
                width=args.width,
                guidance_scale=args.cfg,
                num_inference_steps=args.steps,
                seed=used_seed,
                negative_prompt=args.negative_prompt,
            )
            output = Path(str(output_path))
            output.parent.mkdir(parents=True, exist_ok=True)
            result.images[0].save(output, format="PNG")
            outputs.append({
                "outputPath": str(output), "seed": used_seed,
                "width": result.images[0].width, "height": result.images[0].height,
                "loraActive": bool(pipeline.active_adapters), "loraPath": pipeline.lora_path,
            })
        return {
            "outputs": outputs, **outputs[0], "model": model_name,
            "loraActive": bool(pipeline.active_adapters), "loraPath": pipeline.lora_path,
            "loraAdapters": pipeline.active_adapters,
        }
    finally:
        pipeline.release()


def run_qwen(args: argparse.Namespace) -> dict[str, Any]:
    from aiv_runtime.image_studio.qwen import QwenImageEditPipeline

    if not 1 <= len(args.image_path) <= 3:
        raise ValueError("Qwen Image Edit requires one to three input images")
    os.environ["AIV_QWEN_BASE_MODEL"] = args.base_model_path
    os.environ.setdefault("QWEN_OFFLOAD_MODE", "df_stream")
    os.environ.setdefault("QWEN_STREAM_RESERVE_GB", "4.0")
    os.environ.setdefault("QWEN_STREAM_PIN", "1")
    emit("loading", model="Qwen-Image-Edit-2511", offload=os.environ["QWEN_OFFLOAD_MODE"])
    pipeline = QwenImageEditPipeline.create(
        args.model_path,
        "cuda",
        lora_path=args.lora_path or None,
        lora_scale=args.lora_scale,
    )
    if args.steps <= 8 and not pipeline.lightning_lora_loaded:
        raise RuntimeError("the 4-step Qwen Lightning LoRA did not load; refusing a low-step garbage result")
    used_seed = seed_value(args.seed)
    refs = qwen_inputs(args.image_path, args.pixel_budget)

    def progress(step: int, total: int) -> None:
        emit("denoise", step=step, total=total, seed=used_seed)

    try:
        result = pipeline.edit(
            prompt=args.prompt,
            images=refs,
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.steps,
            true_cfg_scale=args.cfg,
            seed=used_seed,
            step_callback=progress,
            output_size=edit_output_size(args.out_width, args.out_height),
        )
        output = Path(args.output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        result.images[0].save(output, format="PNG")
        return {
            "outputPath": str(output), "seed": used_seed,
            "width": result.images[0].width, "height": result.images[0].height,
            "lightningLora": pipeline.lightning_lora_loaded,
            "fp8Native": pipeline.fp8_native_active,
        }
    finally:
        pipeline.release()


def processed_mask(mask: Image.Image, args: argparse.Namespace) -> Image.Image:
    value = mask.convert("L")
    if args.blur > 0:
        value = value.filter(ImageFilter.GaussianBlur(args.blur))
    if args.dilate > 0:
        value = value.filter(ImageFilter.MaxFilter(args.dilate * 2 + 1))
    if args.erode > 0:
        value = value.filter(ImageFilter.MinFilter(args.erode * 2 + 1))
    if args.threshold >= 0:
        value = value.point(lambda pixel: 255 if pixel >= args.threshold else 0)
    return ImageOps.invert(value) if args.invert else value


def run_birefnet(args: argparse.Namespace) -> dict[str, Any]:
    from aiv_runtime.image_studio.birefnet import BiRefNetPipeline

    emit("loading", model="BiRefNet")
    source = image(args.image_path[0])
    pipeline = BiRefNetPipeline.create(args.model_path, "cuda")
    emit("segmenting", width=source.width, height=source.height)
    result = pipeline.remove_background(source)
    mask = processed_mask(result.mask, args)
    rgba = source.convert("RGBA")
    rgba.putalpha(mask)
    output = Path(args.output_path)
    mask_output = Path(args.mask_output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    mask_output.parent.mkdir(parents=True, exist_ok=True)
    rgba.save(output, format="PNG")
    mask.save(mask_output, format="PNG")
    return {"outputPath": str(output), "maskPath": str(mask_output), "width": source.width, "height": source.height}


def run_composite(args: argparse.Namespace) -> dict[str, Any]:
    from aiv_runtime.image_studio.composite import composite_layers

    emit("compositing", crop=args.crop, canvasWidth=args.width, canvasHeight=args.height)
    return composite_layers(
        layers_json=args.layers_json,
        output_path=args.output_path,
        crop=args.crop,
        canvas_width=args.width,
        canvas_height=args.height,
        padding=args.padding,
        background=args.background,
    )


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Digital Forge Image Studio worker")
    value.add_argument("mode", choices=["generate", "edit", "remove-background", "composite"])
    value.add_argument("--model-path", default="")
    value.add_argument("--base-model-path", default="")
    value.add_argument("--lora-path", default="")
    value.add_argument("--lora-scale", type=float, default=1.0)
    value.add_argument("--model-variant", choices=["turbo", "base"], default="turbo")
    value.add_argument("--output-path", required=True)
    value.add_argument("--output-paths-json", default="")
    value.add_argument("--mask-output-path", default="")
    value.add_argument("--image-path", action="append", default=[])
    value.add_argument("--prompt", default="")
    value.add_argument("--negative-prompt", default="")
    value.add_argument("--width", type=int, default=1024)
    value.add_argument("--height", type=int, default=1024)
    value.add_argument("--out-width", type=int, default=0, help="edit output width; 0 follows Image 1")
    value.add_argument("--out-height", type=int, default=0, help="edit output height; 0 follows Image 1")
    value.add_argument("--steps", type=int, default=9)
    value.add_argument("--cfg", type=float, default=1.0)
    value.add_argument("--seed", type=int, default=-1)
    value.add_argument("--pixel-budget", type=int, default=4_000_000)
    value.add_argument("--blur", type=int, default=0)
    value.add_argument("--dilate", type=int, default=0)
    value.add_argument("--erode", type=int, default=0)
    value.add_argument("--threshold", type=int, default=-1)
    value.add_argument("--invert", action="store_true")
    value.add_argument("--layers-json", default="[]")
    value.add_argument("--crop", choices=["tight", "frame"], default="tight")
    value.add_argument("--background", choices=["transparent", "black", "white"], default="transparent")
    value.add_argument("--padding", type=int, default=0)
    return value


def main() -> int:
    args = parser().parse_args()
    torch_runtime = None
    try:
        if args.mode != "composite":
            import torch

            torch_runtime = torch
            if not torch.cuda.is_available():
                raise RuntimeError("Image Studio requires a CUDA-enabled PyTorch runtime")
        if args.mode in {"generate", "edit"} and not args.prompt.strip():
            raise ValueError("a prompt is required")
        if args.mode == "remove-background" and (not args.image_path or not args.mask_output_path):
            raise ValueError("background removal requires an input and mask output path")
        result = run_zit(args) if args.mode == "generate" else run_qwen(args) if args.mode == "edit" else run_birefnet(args) if args.mode == "remove-background" else run_composite(args)
        emit("complete", **result)
        return 0
    except Exception as exc:
        emit("failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        gc.collect()
        if torch_runtime is not None and torch_runtime.cuda.is_available():
            torch_runtime.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
