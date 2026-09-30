"""Timeline background removal (clip_matte.py): the matte-cache layout, prompts, and the combine.

The models are stood in for (a BiRefNet that returns a soft disc, a SAM 2 that returns a box), so
this runs without either installed; ffmpeg must be on PATH.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from aiv_runtime import clip_matte

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")


def _clip(tmp_path: Path, frames: int = 12) -> Path:
    path = tmp_path / "clip.mp4"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    f"testsrc2=s=160x96:r=24:d={frames / 24}", "-pix_fmt", "yuv420p", str(path)], check=True)
    return path


class _FakeBiRefNet:
    @classmethod
    def create(cls, root, device):
        return cls()

    def remove_background(self, image):
        w, h = image.size
        y, x = np.mgrid[:h, :w]
        soft = np.clip(255 - np.hypot(x - w / 2, y - h / 2) * 4, 0, 255).astype(np.uint8)
        return types.SimpleNamespace(mask=Image.fromarray(soft, mode="L"))


@pytest.fixture
def fake_birefnet(monkeypatch, tmp_path):
    module = types.ModuleType("aiv_runtime.image_studio.birefnet")
    module.BiRefNetPipeline = _FakeBiRefNet
    monkeypatch.setitem(sys.modules, "aiv_runtime.image_studio.birefnet", module)
    root = tmp_path / "BiRefNet"
    root.mkdir()
    (root / "config.json").write_text("{}")
    return root


def test_prompts_carry_a_frame_and_are_checked():
    assert clip_matte.parse_prompts(["0,0.5,0.25,1", "36,0.1,0.9,0"]) == [(0, 0.5, 0.25, 1), (36, 0.1, 0.9, 0)]
    for bad in ["0,0.5,1", "-1,0.5,0.5,1", "0,1.5,0.5,1", "0,0.5,0.5,2"]:
        with pytest.raises(ValueError):
            clip_matte.parse_prompts([bad])


def test_birefnet_writes_a_matte_cache_for_every_frame(tmp_path, fake_birefnet, capsys):
    clip = _clip(tmp_path)
    cache = tmp_path / "cache"
    assert clip_matte.main(["--mode", "birefnet", "--video-path", str(clip), "--start-time", "0", "--end-time", "0.5",
                            "--cache-dir", str(cache), "--cache-id", "c1", "--model-root", str(fake_birefnet), "--device", "cpu"]) == 0
    manifest = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert manifest["frameCount"] == 12 and manifest["model"] == "birefnet" and manifest["method"] == "birefnet"
    assert (manifest["width"], manifest["height"]) == (160, 96)
    masks = sorted((cache / "masks").glob("*.png"))
    assert len(masks) == 12
    with Image.open(masks[0]) as first:
        assert first.mode == "L" and first.size == (160, 96)
        values = np.asarray(first)
        assert values[48, 80] == 255 and values[0, 0] == 0 and 0 < values[48, 30] < 255, "soft edges kept"
    # A second run reuses the cache.
    clip_matte.main(["--mode", "birefnet", "--video-path", str(clip), "--end-time", "0.5", "--cache-dir", str(cache),
                     "--cache-id", "c1", "--model-root", str(fake_birefnet)])
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["reused"] is True


def test_a_sam2_guide_limits_birefnet_to_the_subject(tmp_path, fake_birefnet, capsys):
    clip = _clip(tmp_path)
    guide = tmp_path / "guide"
    (guide / "masks").mkdir(parents=True)
    box = np.zeros((96, 160), dtype=np.uint8)
    box[30:66, 20:60] = 255  # the SAM 2 subject is on the left; BiRefNet's disc is in the middle
    for index in range(12):
        Image.fromarray(box, mode="L").save(guide / "masks" / f"{index:05d}.png")
    cache = tmp_path / "refined"
    clip_matte.main(["--mode", "birefnet", "--video-path", str(clip), "--end-time", "0.5", "--cache-dir", str(cache),
                     "--cache-id", "c2", "--model-root", str(fake_birefnet), "--guide-dir", str(guide), "--device", "cpu"])
    manifest = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert manifest["method"] == "sam2-birefnet" and manifest["model"] == "sam2"
    with Image.open(cache / "masks" / "00000.png") as refined:
        values = np.asarray(refined)
    assert values[48, 40] == 255, "the SAM 2 interior is kept"
    assert values[48, 120] == 0, "BiRefNet cannot add what SAM 2 did not pick"


def test_sam2_prompts_on_several_frames_and_a_still(tmp_path, monkeypatch, capsys):
    seen = {}

    def fake_track(model_root, frames_dir, source, count, prompts, device):
        seen["prompts"], seen["count"] = prompts, count
        mask = np.zeros((source["height"], source["width"]), dtype=bool)
        mask[10:40, 10:50] = True
        return {index: mask for index in range(1, count)}  # frame 0 left for the hold-back check

    monkeypatch.setattr(clip_matte, "track_prompts", fake_track)
    clip = _clip(tmp_path)
    cache = tmp_path / "sam"
    clip_matte.main(["--mode", "sam2", "--video-path", str(clip), "--end-time", "0.5", "--cache-dir", str(cache),
                     "--cache-id", "s", "--prompt", "6,0.2,0.3,1", "--prompt", "0,0.25,0.3,1", "--prompt", "6,0.9,0.9,0"])
    manifest = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert seen["count"] == 12 and len(seen["prompts"]) == 3 and manifest["promptFrames"] == 2
    assert manifest["frameCount"] == 12 and len(list((cache / "masks").glob("*.png"))) == 12
    with Image.open(cache / "masks" / "00005.png") as mask:
        assert np.asarray(mask)[20, 20] == 255
    with pytest.raises(ValueError):
        clip_matte.main(["--mode", "sam2", "--video-path", str(clip), "--end-time", "0.5", "--cache-dir", str(tmp_path / "x"),
                         "--cache-id", "x", "--prompt", "0,0.5,0.5,0"])
    # A still image is one frame.
    still = tmp_path / "still.png"
    Image.new("RGB", (64, 48), (10, 200, 30)).save(still)
    clip_matte.main(["--mode", "sam2", "--video-path", str(still), "--cache-dir", str(tmp_path / "st"), "--cache-id", "st",
                     "--prompt", "0,0.5,0.5,1"])
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["frameCount"] == 1 and seen["count"] == 1
