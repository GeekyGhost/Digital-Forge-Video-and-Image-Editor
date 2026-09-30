from __future__ import annotations

import json

from PIL import Image, ImageDraw

from aiv_runtime.image_studio.composite import composite_layers


def _transparent_source(path) -> None:
    image = Image.new("RGBA", (20, 20), (0, 0, 0, 0))
    ImageDraw.Draw(image).rectangle((5, 6, 14, 15), fill=(255, 0, 0, 255))
    image.save(path)


def test_tight_crop_uses_visible_alpha_and_padding(tmp_path) -> None:
    source = tmp_path / "source.png"
    output = tmp_path / "tight.png"
    _transparent_source(source)
    layers = [{
        "imagePath": str(source), "centerX": 20, "centerY": 20,
        "width": 20, "height": 20, "rotation": 0, "opacity": 1, "visible": True,
    }]

    result = composite_layers(json.dumps(layers), str(output), "tight", 40, 40, 2, "transparent")

    rendered = Image.open(output).convert("RGBA")
    assert result["width"] == 14
    assert result["height"] == 14
    assert rendered.getpixel((0, 0))[3] == 0
    assert rendered.getpixel((2, 2)) == (255, 0, 0, 255)


def test_frame_crop_uses_requested_boundary_and_backplate(tmp_path) -> None:
    source = tmp_path / "source.png"
    output = tmp_path / "frame.png"
    _transparent_source(source)
    layers = [{
        "imagePath": str(source), "centerX": 32, "centerY": 16,
        "width": 20, "height": 20, "rotation": 0, "opacity": 1, "visible": True,
    }]

    result = composite_layers(json.dumps(layers), str(output), "frame", 64, 32, 0, "black")

    rendered = Image.open(output).convert("RGBA")
    assert result["width"] == 64
    assert result["height"] == 32
    assert rendered.getpixel((0, 0)) == (0, 0, 0, 255)
    assert rendered.getpixel((27, 12)) == (255, 0, 0, 255)


def test_layer_edge_crop_is_preserved_in_composited_output(tmp_path) -> None:
    source = tmp_path / "opaque.png"
    output = tmp_path / "layer-cropped.png"
    Image.new("RGBA", (20, 10), (20, 180, 240, 255)).save(source)
    layers = [{
        "imagePath": str(source), "centerX": 20, "centerY": 20,
        "width": 20, "height": 10, "rotation": 0, "opacity": 1, "visible": True,
        "cropLeft": 25, "cropTop": 20, "cropRight": 25, "cropBottom": 20,
    }]

    result = composite_layers(json.dumps(layers), str(output), "tight", 40, 40, 0, "transparent")

    rendered = Image.open(output).convert("RGBA")
    assert result["width"] == 10
    assert result["height"] == 6
    assert rendered.getpixel((0, 0)) == (20, 180, 240, 255)
