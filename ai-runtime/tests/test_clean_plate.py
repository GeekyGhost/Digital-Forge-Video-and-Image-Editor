"""Clean plate for a locked-off shot (VFX Phase 5): the actor painted out from other frames."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

cv2 = pytest.importorskip("cv2")

from aiv_runtime import clean_plate as cp  # noqa: E402

W, H = 96, 64


def background(shift: int = 0) -> np.ndarray:
    """A textured static background (brick-like stripes and a gradient); `shift` pans the camera."""
    y, x = np.mgrid[0:H, 0:W + 64]
    image = np.stack([120 + 60 * ((x // 6 + y // 4) % 2), 40 + 2 * x, 60 + 2 * y], axis=-1).clip(0, 255).astype(np.uint8)
    return np.ascontiguousarray(image[:, 32 + shift:32 + shift + W])


def actor_box(frame: int) -> tuple[int, int, int, int]:
    """A dark actor 14 x 30 that stands at x 30..44 for the first frames, then steps right."""
    x = 30 + min(frame, 20)
    return x, 20, x + 14, 50


def with_actor(image: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    out = image.copy()
    x0, y0, x1, y1 = box
    out[y0:y1, x0:x1] = (20, 20, 25)
    return out


def matte(box: tuple[int, int, int, int]) -> np.ndarray:
    out = np.zeros((H, W), np.uint8)
    x0, y0, x1, y1 = box
    out[y0:y1, x0:x1] = 255
    return out


def test_sampling_spreads_and_walks_outward_from_the_range():
    assert cp.spread(10, 5, 8) == [10, 11, 12, 13, 14]
    picked = cp.spread(0, 100, 5)
    assert picked[0] == 0 and picked[-1] == 99 and len(picked) == 5
    before, after = cp.context_sides(total=1000, first=100, count=50, window=60, step=20)
    assert before == [80, 60, 40]  # nearest first
    assert after == [169, 189, 209]
    assert cp.context_sides(total=150, first=0, count=150, window=60, step=20) == ([], [])


def test_a_side_ends_at_the_camera_move():
    assert cp.agreeing_run([1, 2, 1, 9, 1, 2], limit=5) == 6  # one bad frame (a passer-by) is forgiven
    assert cp.agreeing_run([1, 2, 9, 9, 1, 1], limit=5) == 2  # two in a row: the setup changed
    assert cp.agreeing_run([9, 9, 1], limit=5) == 0


def test_region_is_the_union_box_grown_inside_the_frame():
    assert cp.region_of([matte((30, 20, 44, 50)), matte((50, 20, 64, 50))], 4, W, H) == (26, 16, 68, 54)
    assert cp.region_of([matte((0, 0, 5, 5))], 4, W, H) == (0, 0, 9, 9)
    assert cp.region_of([np.zeros((H, W), np.uint8)], 4, W, H) is None


def test_the_range_alone_recovers_what_the_actor_uncovers():
    clean = background()
    frames = [with_actor(clean, actor_box(index)) for index in range(40)]
    hidden = np.stack([cp.dilate(matte(actor_box(index)) > 127, 2) for index in range(40)])
    still, stats = cp.build_clean(np.stack(frames), hidden, [], ~hidden.any(axis=0))
    assert stats["residual"] < 0.5
    # The actor walks off his first spot, so every pixel is uncovered in enough frames: exact.
    assert stats["holes"] == 0
    assert np.abs(still.astype(int) - clean).max() == 0


def test_context_fills_the_rest_and_skips_a_moved_camera_and_the_standing_actor():
    clean = background()
    # Range: the actor only shuffles (3 px), so most of his spot is never uncovered.
    range_frames = np.stack([with_actor(clean, (30 + index % 3, 20, 44 + index % 3, 50)) for index in range(12)])
    hidden = np.stack([cp.dilate(matte((30 + index % 3, 20, 44 + index % 3, 50)) > 127, 1) for index in range(12)])
    # After the range: the same setup, the actor still standing there in 7 of 12 frames (a plain
    # median would return him), then the camera pans (those must be dropped).
    after = [with_actor(clean, (32, 20, 46, 50)) if index % 12 < 7 else with_actor(clean, (70, 20, 84, 50)) for index in range(12)]
    after += [background(shift=6) for _ in range(4)]
    still, stats = cp.build_clean(range_frames, hidden, [np.zeros((0, H, W, 3), np.uint8), np.stack(after)], ~hidden.any(axis=0))
    assert stats["contextKept"] == 12
    assert stats["holes"] == 0
    assert np.abs(still.astype(int) - clean)[20:50, 30:47].mean() < 2.0


def test_the_patch_is_opaque_over_the_actor_and_fades_out():
    alpha = cp.patch_alpha(matte((30, 20, 44, 50)), 6)
    assert alpha[35, 37] == 255 and alpha[20, 30] == 255
    assert 0 < alpha[35, 49] < 255  # feathered beyond the actor
    assert alpha[0, 0] == 0


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not on PATH")
def test_end_to_end_from_a_clip_and_a_mask_track(tmp_path: Path):
    frames_dir, masks = tmp_path / "frames", tmp_path / "track" / "masks"
    frames_dir.mkdir()
    masks.mkdir(parents=True)
    clean = background()
    total, first, count = 60, 10, 20
    for index in range(total):
        box = actor_box(max(0, index - first))
        Image.fromarray(with_actor(clean, box)).save(frames_dir / f"{index:05d}.png")
        if first <= index < first + count:
            Image.fromarray(matte(box)).save(masks / f"{index - first:05d}.png")
    video = tmp_path / "plate.mkv"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", "25", "-i", str(frames_dir / "%05d.png"),
                    "-c:v", "ffv1", "-pix_fmt", "bgr0", str(video)], check=True)
    out = tmp_path / "patch"
    code = cp.main(["--video", str(video), "--first", str(first), "--count", str(count), "--total-frames", str(total),
                    "--fps", "25", "--plate-width", str(W), "--plate-height", str(H), "--matte-dir", str(masks),
                    "--matte-first", str(first), "--out-dir", str(out)])
    assert code == 0
    files = sorted(out.glob("frame_*.png"))
    assert len(files) == count
    x, y, w, h = cp.region_of([matte(actor_box(index)) for index in range(count)], cp.dilation_radius(H, 0.06), W, H)
    patch = np.asarray(Image.open(files[0]))
    assert patch.shape == (h - y, w - x, 4)
    still = np.asarray(Image.open(out / "clean.png"))
    assert np.abs(still.astype(int) - clean[y:h, x:w]).mean() < 3.0
    # Patch-only from a (here unchanged) still: same frames.
    again = tmp_path / "again"
    assert cp.main(["--video", str(video), "--first", str(first), "--count", str(count), "--plate-width", str(W),
                    "--plate-height", str(H), "--matte-dir", str(masks), "--matte-first", str(first),
                    "--out-dir", str(again), "--still", str(out / "clean.png")]) == 0
    assert np.array_equal(np.asarray(Image.open(again / "frame_0000.png")), patch)


def test_a_moving_camera_is_refused(tmp_path: Path, capsys):
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is not on PATH")
    frames_dir, masks = tmp_path / "frames", tmp_path / "masks"
    frames_dir.mkdir()
    masks.mkdir()
    for index in range(12):
        Image.fromarray(with_actor(background(shift=3 * (index - 6)), (30, 20, 44, 50))).save(frames_dir / f"{index:05d}.png")
        Image.fromarray(matte((30, 20, 44, 50))).save(masks / f"{index:05d}.png")
    video = tmp_path / "pan.mkv"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", "25", "-i", str(frames_dir / "%05d.png"),
                    "-c:v", "ffv1", "-pix_fmt", "bgr0", str(video)], check=True)
    code = cp.main(["--video", str(video), "--first", "0", "--count", "12", "--plate-width", str(W), "--plate-height", str(H),
                    "--matte-dir", str(masks), "--out-dir", str(tmp_path / "out")])
    assert code == 1
    report = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert "not locked off" in report["error"]


def test_a_fill_that_changes_more_than_the_holes_is_refused(tmp_path: Path):
    clean = background()
    holes = np.zeros((H, W), np.uint8)
    holes[20:40, 30:50] = 255
    Image.fromarray(clean).save(tmp_path / "clean.png")
    Image.fromarray(holes).save(tmp_path / "holes.png")
    good = clean.copy()
    good[20:40, 30:50] = 90  # only the holes changed
    cp.check_filled(good, tmp_path)
    with pytest.raises(ValueError, match="outside the holes"):
        cp.check_filled(np.zeros_like(clean), tmp_path)  # a 0..1 model output read as 0..255: black
