"""The SAM 2 + BiRefNet matte combination (VFX Phase 5)."""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from aiv_runtime import character_matte


def _disc(size: int, radius: int, centre=None) -> np.ndarray:
    y, x = np.mgrid[:size, :size]
    cx, cy = centre or (size / 2, size / 2)
    return (((x - cx) ** 2 + (y - cy) ** 2) <= radius ** 2).astype(np.uint8) * 255


def test_birefnet_cannot_add_a_second_subject_or_punch_a_hole():
    sam2 = _disc(200, 50)
    soft = np.zeros((200, 200), np.uint8)
    soft[_disc(200, 55) > 0] = 180           # soft edge a little outside the hard one
    soft[95:105, 95:105] = 0                  # BiRefNet misses a patch in the middle
    soft[_disc(200, 12, (20, 20)) > 0] = 255  # and finds a stray object in the corner
    alpha = character_matte.combine(sam2, soft, 200)
    assert alpha[100, 100] == 255             # the interior comes from SAM 2
    assert alpha[20, 20] == 0                 # the stray object is outside the SAM 2 region
    edge = alpha[100, 100 + 53]                # just outside the hard edge: BiRefNet's soft alpha
    assert 0 < edge <= 180


def test_each_model_alone():
    sam2 = _disc(100, 30)
    feathered = character_matte.combine(sam2, None, 100)
    assert feathered[50, 50] == 255 and feathered[0, 0] == 0
    assert 0 < feathered[50, 50 + 30] < 255   # feathered edge
    soft = (np.arange(100, dtype=np.uint8)[None, :].repeat(100, 0))
    assert np.array_equal(character_matte.combine(None, soft, 100), soft)
    with pytest.raises(ValueError):
        character_matte.combine(None, None, 100)


def test_worker_writes_straight_alpha_rgba_frames_from_sam2_masks(tmp_path, capsys):
    frames, cache, out = tmp_path / "frames", tmp_path / "sam2", tmp_path / "element"
    frames.mkdir()
    (cache / "masks").mkdir(parents=True)
    for index in range(3):
        Image.new("RGB", (64, 48), (200, 40 + index, 40)).save(frames / f"{index:05d}.png")
        mask = np.zeros((48, 64), np.uint8)
        mask[10:40, 20 + index:44 + index] = 255
        Image.fromarray(mask, mode="L").save(cache / "masks" / f"{index:05d}.png")
    assert character_matte.main(["--frames-dir", str(frames), "--sam2-dir", str(cache), "--out-dir", str(out), "--smooth", "0"]) == 0
    stats = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert stats == {**stats, "frames": 3, "sam2": True, "birefnet": False}
    written = sorted(out.glob("frame_*.png"))
    assert [path.name for path in written] == ["frame_0000.png", "frame_0001.png", "frame_0002.png"]
    rgba = Image.open(written[1])
    assert rgba.mode == "RGBA" and rgba.getpixel((32, 25)) == (200, 41, 40, 255) and rgba.getpixel((2, 2))[3] == 0


def test_a_short_track_is_refused(tmp_path):
    frames, cache = tmp_path / "frames", tmp_path / "sam2"
    frames.mkdir()
    (cache / "masks").mkdir(parents=True)
    for index in range(3):
        Image.new("RGB", (8, 8)).save(frames / f"{index:05d}.png")
    Image.new("L", (8, 8)).save(cache / "masks" / "00000.png")
    with pytest.raises(ValueError, match="1 masks for 3 frames"):
        character_matte.main(["--frames-dir", str(frames), "--sam2-dir", str(cache), "--out-dir", str(tmp_path / "o")])
