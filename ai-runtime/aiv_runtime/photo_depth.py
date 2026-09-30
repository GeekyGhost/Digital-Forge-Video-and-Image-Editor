"""Monocular depth for Photo Edit's relighting.

Relight needs a surface normal per pixel, and a normal is a gradient of depth. This writes the depth
map once per picture; the workspace caches it and every light change after that is arithmetic in the
viewport rather than another model run.

The map is deliberately produced at a bounded resolution. Depth Anything works internally at 518px
and the estimate carries no detail past that, so writing it at the full size of a 24 MP photograph
would store upsampled guesses and cost seconds doing it. The renderer resamples to whatever it needs.

Output is an 8-bit greyscale PNG, near = white, normalised across the frame. Normalised because the
model's raw output is relative disparity with no absolute meaning, so the only stable convention is
"the range this picture actually covers".
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

MAX_EDGE = 1024


def emit(event: str, **payload: Any) -> None:
    print(json.dumps({"event": event, **payload}, ensure_ascii=False), flush=True)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Estimate a depth map for one still image")
    value.add_argument("--model-path", required=True, help="a local Depth Anything V2 directory")
    value.add_argument("--image-path", required=True)
    value.add_argument("--output-path", required=True)
    value.add_argument("--max-edge", type=int, default=MAX_EDGE)
    return value


def main() -> int:
    args = parser().parse_args()
    source = Path(args.image_path)
    model_path = Path(args.model_path)
    if not source.is_file():
        raise SystemExit(f"image not found: {source}")
    if not model_path.is_dir():
        raise SystemExit(f"depth model not installed: {model_path}")

    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    image = Image.open(source)
    image.load()
    image = image.convert("RGB")
    limit = max(64, min(args.max_edge, MAX_EDGE))
    if max(image.size) > limit:
        scale = limit / max(image.size)
        image = image.resize(
            (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
            Image.Resampling.LANCZOS,
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(str(model_path))
    model = AutoModelForDepthEstimation.from_pretrained(str(model_path)).to(device).eval()
    print(f"[depth] {model_path.name} on {device} · {image.width}x{image.height}", file=sys.stderr)

    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        predicted = model(**inputs).predicted_depth

    # The model returns its own working resolution; put it back on the picture it came from.
    resized = torch.nn.functional.interpolate(
        predicted.unsqueeze(1), size=(image.height, image.width), mode="bicubic", align_corners=False,
    ).squeeze()
    depth = resized.detach().cpu().numpy().astype("float32")

    low, high = float(depth.min()), float(depth.max())
    # A perfectly flat scene would divide by zero; a flat mid-grey map is the honest answer for one.
    spread = high - low
    normalised = np.full(depth.shape, 128, dtype="uint8") if spread < 1e-6 \
        else ((depth - low) / spread * 255.0).clip(0, 255).astype("uint8")

    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(normalised, mode="L").save(output, format="PNG", optimize=True)
    emit("complete", outputPath=str(output), width=image.width, height=image.height)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
