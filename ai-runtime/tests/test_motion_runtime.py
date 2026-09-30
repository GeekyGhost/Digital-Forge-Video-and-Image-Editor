"""VFX Phase 4: the CoTracker3 and Video Depth Anything workers, with stub models.

The real models need the motion runtime (Setup-Motion.bat); these tests pin everything around them:
coordinate conversion, frame decoding, the JSON and the depth sequence layout. They need ffmpeg
(on PATH) to make and read a small clip; without it the end-to-end tests skip.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from aiv_runtime import motion_track, video_depth

HAVE_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def make_clip(path: Path, frames: int = 12, width: int = 160, height: int = 96) -> None:
    """A white square moving 4 px right per frame over a gradient; lossless, so frames decode exactly."""
    raw = bytearray()
    for t in range(frames):
        image = np.zeros((height, width, 3), dtype=np.uint8)
        image[..., 0] = np.linspace(0, 120, width, dtype=np.uint8)[None, :]
        image[..., 2] = t * 10
        x = 20 + 4 * t
        image[40:56, x:x + 16] = 255
        raw += image.tobytes()
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}", "-r", "24", "-i", "pipe:0", "-c:v", "ffv1", str(path),
    ], input=bytes(raw), check=True)


def test_coordinates_round_trip_and_match_align_corners():
    for size, model in (((1920, 1080), (768, 432)), ((160, 96), (160, 96))):
        for point in ((0.5, 0.5), (959.25, 540.75), (1919.5, 1079.5)):
            x, y = motion_track.to_model(*point, *size, *model)
            back = motion_track.from_model(x, y, *size, *model)
            assert back == pytest.approx(point, abs=1e-9)
    # The first and last pixel centres map onto the first and last model pixel centres.
    assert motion_track.to_model(0.5, 0.5, 1920, 1080, 768, 432) == (0.0, 0.0)
    assert motion_track.to_model(1919.5, 1079.5, 1920, 1080, 768, 432) == pytest.approx((767.0, 431.0))


def test_decoded_size_keeps_aspect_and_even():
    assert motion_track.decoded_size(1920, 1080, 768) == (768, 432)
    assert motion_track.decoded_size(640, 360, 768) == (640, 360)
    assert motion_track.decoded_size(641, 361, 768) == (640, 360)
    assert video_depth.working_size(3840, 2160, 1280) == (1280, 720)


def test_parse_points():
    assert motion_track.parse_points(["10,20.5", "1,2"]) == [(10.0, 20.5), (1.0, 2.0)]
    with pytest.raises(ValueError):
        motion_track.parse_points([])
    with pytest.raises(ValueError):
        motion_track.parse_points(["1,2,3"])


def test_normalise_is_per_clip_and_near_is_255():
    depths = np.stack([np.full((2, 2), 1.0), np.full((2, 2), 3.0), np.array([[1.0, 3.0], [2.0, np.nan]])]).astype(np.float32)
    encoded, low, high = video_depth.normalise(depths)
    assert encoded.dtype == np.uint8
    assert encoded[0].max() == 0 and encoded[1].min() == 255
    assert encoded[2, 1, 0] == 128  # 2.0 is halfway
    assert low == pytest.approx(1.0, abs=0.02) and high == pytest.approx(3.0, abs=0.02)


@pytest.mark.skipif(not HAVE_FFMPEG, reason="needs ffmpeg")
def test_read_frames_is_frame_exact(tmp_path):
    clip = tmp_path / "clip.mkv"
    make_clip(clip)
    source = motion_track.probe(str(clip))
    assert (source["width"], source["height"], source["frames"]) == (160, 96, 12)
    frames = motion_track.read_frames(str(clip), 160, 96, 3, 4)
    assert frames.shape == (4, 96, 160, 3)
    # Frame 3's blue channel is 30, and its square starts at x = 32.
    assert frames[0, 0, 0, 2] == 30
    assert frames[0, 48, 32].tolist() == [255, 255, 255]
    assert frames[0, 48, 31, 1] == 0


@pytest.mark.skipif(not HAVE_FFMPEG, reason="needs ffmpeg")
def test_track_end_to_end_with_a_stub_predictor(tmp_path):
    clip = tmp_path / "clip.mkv"
    make_clip(clip, width=320, height=192)
    seen = {}

    def stub(video: np.ndarray, queries: np.ndarray):
        seen["shape"] = video.shape
        seen["queries"] = queries.copy()
        frames = video.shape[0]
        tracks = np.zeros((frames, len(queries), 2), dtype=np.float32)
        for t in range(frames):
            tracks[t] = queries[:, 1:] + [(t - queries[0, 0]) * 2.0, 0.0]  # 4 source px/frame at half size
        visible = np.ones((frames, len(queries)), dtype=bool)
        visible[5, 0] = False
        return tracks, visible

    out = tmp_path / "track.json"
    code = motion_track.main([
        "--video", str(clip), "--out", str(out), "--checkpoint", "unused", "--query-frame", "4",
        "--point", "56.5,96.5", "--max-width", "160",
    ], predictor=stub)
    assert code == 0
    assert seen["shape"] == (12, 96, 160, 3)
    assert seen["queries"][0].tolist() == pytest.approx([4, (56.0) * 159 / 319, 96.0 * 95 / 191])
    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["source"] == "cotracker3" and "CC-BY-NC" in result["licence"]
    samples = result["points"][0]["samples"]
    assert len(samples) == 12
    assert samples[4][:2] == pytest.approx([56.5, 96.5], abs=1e-3)
    assert samples[5][2] == 0 and samples[6][2] == 1
    # One decoded pixel is (319 / 159) source pixels: 2 decoded px/frame ~ 4.01 source px.
    assert samples[8][0] - samples[4][0] == pytest.approx(4 * 2 * 319 / 159, abs=1e-2)


class StubDepth:
    def infer_video_depth(self, frames, target_fps, input_size=518, device="cuda", fp32=False):
        t, h, w = frames.shape[:3]
        ramp = np.linspace(0.0, 10.0, w, dtype=np.float32)[None, None, :].repeat(t, 0).repeat(h, 1)
        return ramp + np.arange(t, dtype=np.float32)[:, None, None] * 0.1, target_fps


@pytest.mark.skipif(not HAVE_FFMPEG, reason="needs ffmpeg")
def test_depth_end_to_end_writes_the_mask_cache_layout(tmp_path):
    clip = tmp_path / "clip.mkv"
    make_clip(clip, frames=7)
    cache = tmp_path / "depth"
    assert video_depth.main(["--video", str(clip), "--cache-dir", str(cache), "--checkpoint", "unused"], model=StubDepth()) == 0
    manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["kind"] == "depth" and manifest["frames"] == 7 and "Apache" in manifest["licence"]
    files = sorted((cache / "masks").glob("*.png"))
    assert [path.name for path in files] == [f"{index:05d}.png" for index in range(7)]
    first = np.asarray(Image.open(files[0]))
    assert first.shape == (96, 160) and first.dtype == np.uint8
    assert first[0, 0] < 10 and first[0, -1] > 235
    # A second run reuses the cache instead of loading the model.
    assert video_depth.main(["--video", str(clip), "--cache-dir", str(cache), "--checkpoint", "unused"], model=None) == 0
