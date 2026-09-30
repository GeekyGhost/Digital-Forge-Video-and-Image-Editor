"""Write a finished image out to a user-chosen path, format and quality.

Everything Image Studio produces lands as PNG inside the project's own media folder, which is app
storage the user is not expected to browse. This is the step that gets a picture *out* -- and the
one decision it must not make silently is alpha.

Pillow's default behaviour when an RGBA image is saved as JPEG is to raise, and the usual "fix" is
`.convert("RGB")`, which composites the cutout onto **black**. A tight-alpha cutout exported to
JPEG would therefore come back with a black halo that looks like a bad matte rather than a format
limitation. So the matte colour is an explicit argument with no default guess: the caller either
picks one or picks a format that keeps the alpha.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Formats that can carry an alpha channel. Anything outside this set needs a matte decision.
ALPHA_FORMATS = {"PNG", "WEBP", "TIFF"}

SUFFIX_FORMATS = {
    ".png": "PNG", ".jpg": "JPEG", ".jpeg": "JPEG",
    ".webp": "WEBP", ".tif": "TIFF", ".tiff": "TIFF",
}


def emit(event: str, **payload: Any) -> None:
    print(json.dumps({"event": event, **payload}, ensure_ascii=False), flush=True)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Export an image to a chosen path and format")
    value.add_argument("--input", required=True)
    value.add_argument("--output", required=True)
    value.add_argument("--format", default="", help="PNG/JPEG/WEBP/TIFF; inferred from the suffix when blank")
    value.add_argument("--quality", type=int, default=92, help="JPEG and lossy WebP, 1-100")
    value.add_argument("--lossless", action="store_true", help="WebP only")
    value.add_argument(
        "--matte", default="",
        help="#rrggbb to composite alpha onto. Required to flatten into a format without alpha.",
    )
    value.add_argument("--width", type=int, default=0, help="optional resize, applied after loading")
    value.add_argument("--height", type=int, default=0)
    return value


def parse_matte(value: str) -> tuple[int, int, int]:
    text = value.strip().lstrip("#")
    if len(text) != 6:
        raise SystemExit(f"matte must be #rrggbb, got {value!r}")
    return tuple(int(text[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def main() -> int:
    from PIL import Image

    args = parser().parse_args()
    source = Path(args.input)
    destination = Path(args.output)
    if not source.is_file():
        raise SystemExit(f"image not found: {source}")

    fmt = (args.format or SUFFIX_FORMATS.get(destination.suffix.lower(), "")).upper()
    if fmt not in {"PNG", "JPEG", "WEBP", "TIFF"}:
        raise SystemExit(f"unsupported image format: {args.format or destination.suffix!r}")

    with Image.open(source) as opened:
        opened.load()
        image = opened.copy()

    if args.width > 0 and args.height > 0 and image.size != (args.width, args.height):
        image = image.resize((args.width, args.height), Image.Resampling.LANCZOS)

    has_alpha = image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info)
    keeps_alpha = fmt in ALPHA_FORMATS and not (fmt == "WEBP" and not args.lossless and args.matte)

    if has_alpha and not keeps_alpha:
        if not args.matte:
            raise SystemExit(
                f"{fmt} cannot store transparency and no matte colour was given. Pick a matte, or "
                f"export as PNG, WebP or TIFF."
            )
        flat = Image.new("RGB", image.size, parse_matte(args.matte))
        flat.paste(image.convert("RGBA"), mask=image.convert("RGBA").split()[-1])
        image = flat
    elif has_alpha:
        image = image.convert("RGBA")
    else:
        image = image.convert("RGB")

    destination.parent.mkdir(parents=True, exist_ok=True)
    options: dict[str, Any] = {}
    if fmt == "JPEG":
        # `optimize` costs a little time and is free quality; 4:4:4 keeps chroma detail that the
        # default 4:2:0 would throw away on graphics and text.
        options = {"quality": max(1, min(100, args.quality)), "optimize": True, "subsampling": 0}
    elif fmt == "WEBP":
        options = {"lossless": args.lossless, "quality": max(1, min(100, args.quality))}
    elif fmt == "TIFF":
        options = {"compression": "tiff_lzw"}

    image.save(destination, fmt, **options)
    emit(
        "complete",
        outputPath=str(destination), format=fmt,
        width=image.width, height=image.height,
        alpha=image.mode in ("RGBA", "LA"),
        bytes=destination.stat().st_size,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as error:  # noqa: BLE001 - the message is the entire point of this path
        print(f"image export failed: {error}", file=sys.stderr, flush=True)
        raise SystemExit(1)
