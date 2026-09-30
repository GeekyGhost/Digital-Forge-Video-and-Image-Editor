"""Point tracking with CoTracker3 (VFX Phase 4).

One job: follow a few query points through a clip and write their tracks as JSON. The app stores
the result inline in the Photo Edit document (``PointTrack``); the JSON under ``.aiv/cache/tracks``
is the provenance and the import source.

Coordinates. The app measures a track in the clip's SOURCE pixels as continuous coordinates (pixel
i spans [i, i + 1), so its centre is i + 0.5), which is what placement uses. CoTracker's queries and
tracks are pixel-centre indices on an ``align_corners=True`` grid (it scales by (W' - 1)/(W - 1)).
``to_model`` / ``from_model`` convert between the two and between the clip's size and the smaller
size the frames are decoded at; they are the only place that happens.

CoTracker3 is CC-BY-NC 4.0 (non-commercial). The app shows that next to the engine choice. The
built-in Lucas-Kanade tracker in the app (``src/domain/photoTrack.ts``) needs no model at all.

Run:  python -m aiv_runtime.motion_track --video V --out T.json --checkpoint C --query-frame Q --point x,y
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

_PACKAGE_ROOT = str(Path(__file__).resolve().parent.parent)
if _PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, _PACKAGE_ROOT)

LICENCE = "CoTracker3: CC-BY-NC 4.0 (Meta Platforms), non-commercial use only"
DEFAULT_REPO = Path(__file__).resolve().parents[2] / "vendor" / "co-tracker"
MAX_FRAMES = 600
DEFAULT_MAX_WIDTH = 768


def force_utf8_io() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CoTracker3 point tracking for a clip")
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--repo", default=str(DEFAULT_REPO))
    parser.add_argument("--query-frame", type=int, default=0, help="source frame the points were picked on")
    parser.add_argument("--point", action="append", default=[], help="x,y in source pixels (continuous)")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--frame-count", type=int, default=0, help="0 = to the end (at most MAX_FRAMES)")
    parser.add_argument("--max-width", type=int, default=DEFAULT_MAX_WIDTH)
    parser.add_argument("--cpu", action="store_true")
    return parser


def parse_points(raw: Sequence[str]) -> list[tuple[float, float]]:
    points = []
    for item in raw:
        parts = item.split(",")
        if len(parts) != 2:
            raise ValueError(f"a point is x,y, got {item!r}")
        points.append((float(parts[0]), float(parts[1])))
    if not points:
        raise ValueError("at least one --point is needed")
    return points


def to_model(x: float, y: float, width: int, height: int, model_width: int, model_height: int) -> tuple[float, float]:
    """Continuous source pixels -> pixel-centre indices of the decoded (smaller) frames."""
    sx = (model_width - 1) / max(1, width - 1)
    sy = (model_height - 1) / max(1, height - 1)
    return (x - 0.5) * sx, (y - 0.5) * sy


def from_model(x: float, y: float, width: int, height: int, model_width: int, model_height: int) -> tuple[float, float]:
    """The inverse of ``to_model``."""
    sx = max(1, width - 1) / max(1, model_width - 1)
    sy = max(1, height - 1) / max(1, model_height - 1)
    return x * sx + 0.5, y * sy + 0.5


def decoded_size(width: int, height: int, max_width: int) -> tuple[int, int]:
    """The frames are decoded no wider than ``max_width`` (even sizes, aspect kept)."""
    if width <= max_width:
        return width - width % 2, height - height % 2
    scale = max_width / width
    return max_width - max_width % 2, max(2, int(round(height * scale / 2)) * 2)


def probe(path: str) -> dict:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
         "-show_entries", "stream=width,height,avg_frame_rate,nb_read_packets", "-of", "json", path],
        capture_output=True, text=True, check=True,
    )
    stream = (json.loads(result.stdout).get("streams") or [{}])[0]
    numerator, _, denominator = (stream.get("avg_frame_rate") or "24/1").partition("/")
    try:
        fps = float(numerator) / float(denominator or 1)
    except (ValueError, ZeroDivisionError):
        fps = 24.0
    return {
        "width": int(stream.get("width") or 0), "height": int(stream.get("height") or 0),
        "fps": fps if fps > 0 else 24.0, "frames": int(stream.get("nb_read_packets") or 0),
    }


def read_frames(path: str, width: int, height: int, start: int, count: int) -> np.ndarray:
    """Frames [start, start + count) in presentation order, RGB uint8, (T, height, width, 3)."""
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-i", path,
        "-vf", f"select=gte(n\\,{start}),scale={width}:{height}:flags=area", "-fps_mode", "passthrough",
        "-frames:v", str(count), "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ]
    raw = subprocess.run(command, capture_output=True, check=True).stdout
    frame_bytes = width * height * 3
    frames = len(raw) // frame_bytes
    if not frames:
        raise RuntimeError("ffmpeg produced no frames for the selected range")
    return np.frombuffer(raw[: frames * frame_bytes], dtype=np.uint8).reshape(frames, height, width, 3)


# A predictor takes (video (T, H, W, 3) uint8, queries (N, 3) [t, x, y] in decoded pixel-centre
# indices) and returns (tracks (T, N, 2), visible (T, N) bool). The real one wraps CoTracker3.
Predictor = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


def cotracker_predictor(checkpoint: Path, repo: Path, device: str) -> Predictor:
    import torch  # noqa: PLC0415 -- only the real run needs torch

    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from cotracker.predictor import CoTrackerPredictor  # noqa: PLC0415

    model = CoTrackerPredictor(checkpoint=str(checkpoint), offline=True, window_len=60).to(device)

    def predict(video: np.ndarray, queries: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        tensor = torch.from_numpy(np.ascontiguousarray(video)).permute(0, 3, 1, 2)[None].float().to(device)
        query = torch.from_numpy(queries.astype(np.float32))[None].to(device)
        backward = bool((queries[:, 0] > 0).any())
        tracks, visible = model(tensor, queries=query, backward_tracking=backward)
        return tracks[0].cpu().numpy(), visible[0].cpu().numpy().astype(bool)

    return predict


def track(
    video: np.ndarray, points: Sequence[tuple[float, float]], query_frame: int, source_size: tuple[int, int],
    predictor: Predictor,
) -> list[list[list[float]]]:
    """Run the predictor and convert back to continuous source pixels: per point, [[x, y, seen], ...]."""
    frames, model_height, model_width = video.shape[:3]
    width, height = source_size
    queries = np.array([[query_frame, *to_model(x, y, width, height, model_width, model_height)] for x, y in points], dtype=np.float32)
    tracks, visible = predictor(video, queries)
    if tracks.shape[:2] != (frames, len(points)):
        raise RuntimeError(f"the tracker returned {tracks.shape}, expected ({frames}, {len(points)}, 2)")
    out = []
    for n in range(len(points)):
        samples = []
        for t in range(frames):
            x, y = from_model(float(tracks[t, n, 0]), float(tracks[t, n, 1]), width, height, model_width, model_height)
            samples.append([round(x, 3), round(y, 3), 1 if bool(visible[t, n]) else 0])
        out.append(samples)
    return out


def main(argv: Sequence[str] | None = None, predictor: Predictor | None = None) -> int:
    force_utf8_io()
    args = build_parser().parse_args(argv)
    started = time.perf_counter()
    points = parse_points(args.point)
    source = probe(args.video)
    width, height = source["width"], source["height"]
    if width <= 0 or height <= 0:
        raise RuntimeError("could not read the clip's size")
    available = source["frames"] - args.start_frame if source["frames"] else MAX_FRAMES
    count = min(MAX_FRAMES, args.frame_count or available, max(1, available))
    query = args.query_frame - args.start_frame
    if not 0 <= query < count:
        raise ValueError(f"the query frame {args.query_frame} is outside the tracked range")
    model_width, model_height = decoded_size(width, height, args.max_width)
    video = read_frames(args.video, model_width, model_height, args.start_frame, count)
    device = "cpu"
    if predictor is None:
        import torch  # noqa: PLC0415

        device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
        predictor = cotracker_predictor(Path(args.checkpoint), Path(args.repo), device)
    tracks = track(video, points, query, (width, height), predictor)
    result = {
        "ok": True, "source": "cotracker3", "licence": LICENCE, "video": args.video,
        "width": width, "height": height, "fps": source["fps"], "first": args.start_frame,
        "frames": int(video.shape[0]), "queryFrame": args.query_frame, "decoded": [model_width, model_height],
        "points": [{"query": list(point), "samples": samples} for point, samples in zip(points, tracks)],
        "device": device, "seconds": round(time.perf_counter() - started, 2),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "points"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
