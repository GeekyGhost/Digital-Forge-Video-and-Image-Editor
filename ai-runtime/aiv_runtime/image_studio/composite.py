"""Source-resolution compositor for the Image Studio canvas."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps


def _layer_bitmap(layer: dict[str, Any]) -> tuple[Image.Image, int, int]:
    source = Path(str(layer["imagePath"]))
    if not source.is_file():
        raise FileNotFoundError(f"canvas layer image not found: {source}")
    image = Image.open(source)
    image.load()
    image = image.convert("RGBA")
    width = max(1, int(layer["width"]))
    height = max(1, int(layer["height"]))
    if image.size != (width, height):
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    crop_left = max(0.0, min(99.0, float(layer.get("cropLeft", 0.0))))
    crop_top = max(0.0, min(99.0, float(layer.get("cropTop", 0.0))))
    crop_right = max(0.0, min(99.0, float(layer.get("cropRight", 0.0))))
    crop_bottom = max(0.0, min(99.0, float(layer.get("cropBottom", 0.0))))
    if crop_left + crop_right >= 100.0 or crop_top + crop_bottom >= 100.0:
        raise ValueError("canvas layer crop bounds must leave visible width and height")
    if crop_left or crop_top or crop_right or crop_bottom:
        left = min(width - 1, max(0, round(width * crop_left / 100.0)))
        top = min(height - 1, max(0, round(height * crop_top / 100.0)))
        right = max(left + 1, min(width, round(width * (100.0 - crop_right) / 100.0)))
        bottom = max(top + 1, min(height, round(height * (100.0 - crop_bottom) / 100.0)))
        alpha = image.getchannel("A")
        cropped_alpha = Image.new("L", image.size, 0)
        cropped_alpha.paste(alpha.crop((left, top, right, bottom)), (left, top))
        image.putalpha(cropped_alpha)
    rotation = float(layer.get("rotation", 0.0))
    if abs(rotation) > 1e-3:
        image = image.rotate(-rotation, resample=Image.Resampling.BICUBIC, expand=True)
    opacity = max(0.0, min(1.0, float(layer.get("opacity", 1.0))))
    if opacity < 0.999:
        alpha = image.getchannel("A").point(lambda value: round(value * opacity))
        image.putalpha(alpha)
    x = round(float(layer["centerX"]) - image.width / 2)
    y = round(float(layer["centerY"]) - image.height / 2)
    return image, x, y


def _alpha_composite_clipped(base: Image.Image, layer: Image.Image, x: int, y: int) -> None:
    left = max(0, x)
    top = max(0, y)
    right = min(base.width, x + layer.width)
    bottom = min(base.height, y + layer.height)
    if right <= left or bottom <= top:
        return
    source = layer.crop((left - x, top - y, right - x, bottom - y))
    base.alpha_composite(source, dest=(left, top))


def composite_layers(
    layers_json: str,
    output_path: str,
    crop: str,
    canvas_width: int,
    canvas_height: int,
    padding: int,
    background: str,
) -> dict[str, Any]:
    raw_layers = json.loads(layers_json)
    if not isinstance(raw_layers, list):
        raise ValueError("canvas layers must be a JSON list")
    visible = [layer for layer in raw_layers if isinstance(layer, dict) and layer.get("visible", True)]
    if not visible:
        raise ValueError("the canvas has no visible layers")
    if crop not in {"tight", "frame"}:
        raise ValueError("canvas crop must be tight or frame")
    if background not in {"transparent", "black", "white"}:
        raise ValueError("canvas background must be transparent, black, or white")
    if not 1 <= canvas_width <= 8192 or not 1 <= canvas_height <= 8192:
        raise ValueError("canvas dimensions must be between 1 and 8192 pixels")
    if canvas_width * canvas_height > 67_108_864:
        raise ValueError("canvas output is limited to 64 megapixels")

    rendered = [_layer_bitmap(layer) for layer in visible]
    if crop == "tight":
        min_x = min(x for _image, x, _y in rendered)
        min_y = min(y for _image, _x, y in rendered)
        max_x = max(x + image.width for image, x, _y in rendered)
        max_y = max(y + image.height for image, _x, y in rendered)
        base = Image.new("RGBA", (max(1, max_x - min_x), max(1, max_y - min_y)), (0, 0, 0, 0))
        for layer, x, y in rendered:
            base.alpha_composite(layer, dest=(x - min_x, y - min_y))
        alpha_bounds = base.getchannel("A").getbbox()
        if alpha_bounds is None:
            raise ValueError("the visible canvas layers are fully transparent")
        base = base.crop(alpha_bounds)
        if padding > 0:
            base = ImageOps.expand(base, border=min(4096, padding), fill=(0, 0, 0, 0))
    else:
        fill = (0, 0, 0, 0) if background == "transparent" else (0, 0, 0, 255) if background == "black" else (255, 255, 255, 255)
        base = Image.new("RGBA", (canvas_width, canvas_height), fill)
        for layer, x, y in rendered:
            _alpha_composite_clipped(base, layer, x, y)

    if base.width * base.height > 67_108_864:
        raise ValueError("composited image is larger than the 64 megapixel limit")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    base.save(output, format="PNG", optimize=True)
    return {"outputPath": str(output), "width": base.width, "height": base.height}
