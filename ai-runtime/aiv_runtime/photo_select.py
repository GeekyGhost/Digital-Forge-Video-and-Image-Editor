"""SAM 2 click-to-select on one still image.

The video tracker next door already does the hard part across a clip. On a still there is no
propagation to do, so this is the image predictor: click the subject, get its mask. That mask is not
the final word -- Photo Edit stores it as the starting point and lets brush strokes correct the
edges it gets wrong, which is why it is written as a plain matte rather than baked into a cutout.

Points arrive normalised to the picture so the caller never has to know what resolution the
predictor ran at. Include points say "this is the subject"; exclude points carve back out of it,
which is how you tell it the shadow underneath is not part of the chair.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def emit(event: str, **payload: Any) -> None:
    print(json.dumps({"event": event, **payload}, ensure_ascii=False), flush=True)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Select a subject in one still image with SAM 2")
    value.add_argument("--model-root", required=True)
    value.add_argument("--image-path", required=True)
    value.add_argument("--output-path", required=True)
    value.add_argument(
        "--point", action="append", default=[],
        help="x,y,include with x and y in 0..1 and include as 1 or 0",
    )
    return value


def find_checkpoint(model_root: Path) -> tuple[Path, str]:
    """Locate the checkpoint and the Hydra config name, the same way the video tracker does."""
    checkpoint = next(iter(sorted(model_root.glob("*.pt"))), None)
    if checkpoint is None:
        raise FileNotFoundError(f"no SAM 2 checkpoint (*.pt) found under {model_root}")
    config = next(iter(sorted(model_root.glob("*.yaml"))), None)
    config_name = f"configs/sam2.1/{config.name}" if config else "configs/sam2.1/sam2.1_hiera_l.yaml"
    return checkpoint, config_name


def parse_points(raw: list[str], width: int, height: int) -> tuple[list[list[float]], list[int]]:
    coordinates: list[list[float]] = []
    labels: list[int] = []
    for entry in raw:
        parts = entry.split(",")
        if len(parts) != 3:
            raise SystemExit(f"a point must be x,y,include -- got {entry!r}")
        x, y, include = float(parts[0]), float(parts[1]), int(parts[2])
        coordinates.append([min(max(x, 0.0), 1.0) * width, min(max(y, 0.0), 1.0) * height])
        labels.append(1 if include else 0)
    if not coordinates:
        raise SystemExit("click the subject first: at least one point is required")
    if 1 not in labels:
        raise SystemExit("at least one point must be an include point")
    return coordinates, labels


def main() -> int:
    args = parser().parse_args()
    source, model_root = Path(args.image_path), Path(args.model_root)
    if not source.is_file():
        raise SystemExit(f"image not found: {source}")
    if not model_root.is_dir():
        raise SystemExit(f"SAM 2 pack not installed: {model_root}")

    import numpy as np
    import torch
    from PIL import Image
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    picture = Image.open(source)
    picture.load()
    picture = picture.convert("RGB")
    coordinates, labels = parse_points(args.point, picture.width, picture.height)

    checkpoint, config_name = find_checkpoint(model_root)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[sam2] {checkpoint.name} on {device} · {len(labels)} points", file=sys.stderr)

    model = build_sam2(config_name, str(checkpoint), device=device)
    predictor = SAM2ImagePredictor(model)
    predictor.set_image(np.asarray(picture))
    masks, scores, _ = predictor.predict(
        point_coords=np.asarray(coordinates, dtype="float32"),
        point_labels=np.asarray(labels, dtype="int32"),
        multimask_output=True,
    )
    # SAM 2 offers several readings of the same click -- typically the part, the whole, and
    # something between. Take the one it is most confident in rather than the first.
    best = int(np.argmax(scores))
    matte = (masks[best] > 0).astype("uint8") * 255

    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(matte, mode="L").save(output, format="PNG", optimize=True)
    emit("complete", outputPath=str(output), width=picture.width, height=picture.height,
         score=float(scores[best]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
