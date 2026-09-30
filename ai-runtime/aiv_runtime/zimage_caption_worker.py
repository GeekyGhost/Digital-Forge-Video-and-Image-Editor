"""One-process local captioning for Z-Image LoRA datasets."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

# Transformers v5 materialises checkpoint tensors on a four-thread pool. On this machine that path
# has faulted inside torch_cpu.dll (0xC0000005) while loading large language models, so load on the
# calling thread. Must be set before transformers is imported; setdefault keeps it overridable.
os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")

from PIL import Image, ImageOps

from aiv_runtime.zimage_training import (
    build_training_caption,
    caption_instruction,
    inspect_image,
    parse_model_json,
)

# The JSON answer carries eleven structured fields plus a prose caption. 192 tokens truncated it
# regularly, and a truncated answer used to fall through as raw JSON text into the training caption.
MAX_CAPTION_TOKENS = 512


def emit(event: str, **payload: Any) -> None:
    print(json.dumps({"event": event, **payload}, ensure_ascii=False), flush=True)


def confidence_value(value: Any) -> float:
    if isinstance(value, str) and value.strip().lower() in {"low", "medium", "high"}:
        return {"low": 0.45, "medium": 0.7, "high": 0.9}[value.strip().lower()]
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return 0.75


def string_list(value: Any) -> list[str]:
    """Coerce a model-provided list field to strings; Tauri deserialises it as Vec<String>."""
    if value is None:
        return []
    items = value if isinstance(value, list) else [value]
    return [str(item).strip() for item in items if str(item).strip()]


def load_model(backend: str, model_path: str, processor_path: str):
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    if backend == "qwen":
        from transformers import Qwen2_5_VLForConditionalGeneration

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            quantization_config=quantization,
            dtype=torch.bfloat16,
            device_map="auto",
            local_files_only=True,
        )
    else:
        from transformers import Gemma3ForConditionalGeneration

        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_path,
            quantization_config=quantization,
            dtype=torch.bfloat16,
            device_map="auto",
            local_files_only=True,
        )
    # Do not pass the image processor's ``backend`` option here. Qwen's AutoProcessor also builds
    # a Qwen2VLVideoProcessor and Transformers forwards the option into it, but that processor has
    # a read-only ``backend`` property. The model only receives still images in this worker, and
    # the processor's local configuration already selects the correct image implementation.
    processor = AutoProcessor.from_pretrained(processor_path, local_files_only=True)
    model.eval()
    return model, processor


def describe(model, processor, backend: str, image: Image.Image, instruction: str) -> str:
    import torch

    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": instruction}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
    inputs = {key: value.to(model.device) if hasattr(value, "to") else value for key, value in inputs.items()}
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=MAX_CAPTION_TOKENS, do_sample=False)
    prompt_length = inputs["input_ids"].shape[-1]
    return processor.batch_decode(generated[:, prompt_length:], skip_special_tokens=True)[0].strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["qwen", "gemma"], default="qwen")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--processor-path", required=True)
    parser.add_argument("--profile", choices=["subject", "character", "style", "product", "concept"], required=True)
    parser.add_argument("--trigger", required=True)
    parser.add_argument("--class-name", required=True)
    parser.add_argument("--image-path", action="append", default=[])
    args = parser.parse_args()
    if not args.image_path:
        raise ValueError("at least one image is required")

    instruction = caption_instruction(args.profile, args.trigger, args.class_name)
    emit("loading", backend=args.backend, images=len(args.image_path))
    model, processor = load_model(args.backend, args.model_path, args.processor_path)
    items: list[dict[str, Any]] = []
    exact: dict[str, str] = {}
    perceptual: dict[str, str] = {}
    for index, path in enumerate(args.image_path, 1):
        inspection = inspect_image(path)
        duplicate_of = exact.get(inspection.sha256) or perceptual.get(inspection.perceptual_hash)
        with Image.open(path) as opened:
            opened.draft("RGB", (2048, 2048))
            prepared = ImageOps.exif_transpose(opened).convert("RGB")
            prepared.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
            raw_text = describe(model, processor, args.backend, prepared, instruction)
        raw = parse_model_json(raw_text)
        caption, notes = build_training_caption(args.profile, args.trigger, args.class_name, raw)
        flags = list(inspection.quality_flags)
        if not raw:
            flags.append("caption-unparsed")
        flags.extend(notes)
        positive_flags = {
            "clear", "sharp", "well-lit", "well lit", "good quality", "decent focus", "high quality",
            "detailed", "vibrant", "well-composed", "well composed",
        }
        flags.extend(
            str(item) for item in string_list(raw.get("quality_flags"))
            if str(item).strip().casefold() not in positive_flags
        )
        if duplicate_of:
            flags.append("possible-duplicate")
        item = {
            "imagePath": path,
            "caption": caption,
            "tags": string_list(raw.get("tags")),
            "qualityFlags": sorted(set(flags)),
            "confidence": confidence_value(raw.get("confidence", 0.75)) if raw else 0.2,
            "width": inspection.width,
            "height": inspection.height,
            "duplicateOf": duplicate_of,
            "rawCaption": raw_text,
        }
        items.append(item)
        exact.setdefault(inspection.sha256, path)
        perceptual.setdefault(inspection.perceptual_hash, path)
        emit("captioned", index=index, total=len(args.image_path), item=item)
    emit("complete", items=items)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
