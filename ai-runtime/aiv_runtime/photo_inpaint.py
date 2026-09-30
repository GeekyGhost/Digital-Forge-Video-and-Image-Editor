"""LaMa inpainting: fill a masked region with what should plausibly be behind it.

This exists because Qwen Image Edit takes no mask. Asked to remove one object it regenerates the
whole frame, so everything you did not want touched drifts. LaMa takes the mask directly and only
the masked pixels change, which is the difference between "remove that sign" working and not.

Expects the TorchScript export of big-lama, the same file the lama-cleaner / IOPaint ecosystem
ships. Its contract: image and mask as float32 NCHW in 0..1, both padded to a multiple of eight,
mask binarised at 0.5; the model returns NCHW in 0..1 (the lama-cleaner export) or 0..255 (others),
which `to_uint8` reads from the output.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PAD_TO = 8


def to_uint8(result):
    """The model's output as 0..255 bytes. The big-lama TorchScript files in circulation differ: the
    lama-cleaner export returns 0..1 (lama-cleaner multiplies by 255 itself), others 0..255. Taking
    0..1 as 0..255 filled every region with black (found by the VFX Phase 5 clean plate, 2026-09-29),
    so the range is read from the output rather than assumed."""
    import numpy as np

    peak = float(np.nanmax(result)) if result.size else 0.0
    scale = 255.0 if peak <= 1.5 else 1.0
    return np.clip(np.rint(result * scale), 0, 255).astype("uint8")


def emit(event: str, **payload: Any) -> None:
    print(json.dumps({"event": event, **payload}, ensure_ascii=False), flush=True)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Inpaint a masked region of one still image")
    value.add_argument("--model-path", required=True, help="big-lama TorchScript .pt")
    value.add_argument("--image-path", required=True)
    value.add_argument("--mask-path", required=True, help="white where the fill should happen")
    value.add_argument("--output-path", required=True)
    return value


def main() -> int:
    args = parser().parse_args()
    source, mask_path, model_path = Path(args.image_path), Path(args.mask_path), Path(args.model_path)
    for label, path in (("image", source), ("mask", mask_path)):
        if not path.is_file():
            raise SystemExit(f"{label} not found: {path}")
    if not model_path.is_file():
        raise SystemExit(f"LaMa model not installed: {model_path}")

    import numpy as np
    import torch
    from PIL import Image

    picture = Image.open(source)
    picture.load()
    # Alpha is set aside rather than fed to the model: LaMa is trained on three channels, and a
    # cutout's transparency has to survive the round trip untouched or the edit eats its own matte.
    alpha = picture.getchannel("A") if picture.mode in ("RGBA", "LA") else None
    picture = picture.convert("RGB")
    width, height = picture.size

    mask = Image.open(mask_path)
    mask.load()
    mask = mask.convert("L")
    if mask.size != picture.size:
        mask = mask.resize(picture.size, Image.Resampling.NEAREST)

    image_array = np.asarray(picture, dtype="float32") / 255.0
    mask_array = (np.asarray(mask, dtype="float32") / 255.0 > 0.5).astype("float32")
    if mask_array.sum() < 1:
        raise SystemExit("the mask is empty, so there is nothing to fill")

    pad_h = (PAD_TO - height % PAD_TO) % PAD_TO
    pad_w = (PAD_TO - width % PAD_TO) % PAD_TO
    if pad_h or pad_w:
        image_array = np.pad(image_array, ((0, pad_h), (0, pad_w), (0, 0)), mode="symmetric")
        mask_array = np.pad(mask_array, ((0, pad_h), (0, pad_w)), mode="constant")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[lama] {model_path.name} on {device} · {width}x{height}", file=sys.stderr)
    model = torch.jit.load(str(model_path), map_location=device).eval()

    image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0).to(device)
    mask_tensor = torch.from_numpy(mask_array).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        filled = model(image_tensor, mask_tensor)

    result = to_uint8(filled[0].permute(1, 2, 0).detach().cpu().numpy())[:height, :width]
    output_image = Image.fromarray(result, mode="RGB")
    if alpha is not None:
        output_image = output_image.convert("RGBA")
        output_image.putalpha(alpha)

    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output_image.save(output, format="PNG")
    emit("complete", outputPath=str(output), width=width, height=height)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
