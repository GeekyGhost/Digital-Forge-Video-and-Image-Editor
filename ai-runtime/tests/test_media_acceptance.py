from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from aiv_runtime.kiwi_edit_16gb import extract_range


FFMPEG = shutil.which("ffmpeg")


def run(*args: str) -> None:
    subprocess.run([FFMPEG or "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


def make_source(path: Path, fps: int) -> None:
    run(
        "-f", "lavfi", "-i", f"testsrc2=size=160x96:rate={fps}:duration=2",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=2",
        "-c:v", "mpeg4", "-q:v", "3", "-c:a", "aac", "-shortest", str(path),
    )


@pytest.mark.skipif(FFMPEG is None, reason="FFmpeg is unavailable")
@pytest.mark.parametrize("fps", [15, 24, 30])
def test_trimmed_cfr_sources_decode_to_the_same_working_cadence(tmp_path: Path, fps: int) -> None:
    source = tmp_path / f"source-{fps}.mp4"
    frames = tmp_path / f"frames-{fps}"
    make_source(source, fps)
    layout = {"startTime": 0.25, "endTime": 1.25, "fps": 15.0, "width": 160, "height": 96}
    decoded = extract_range(SimpleNamespace(video_path=str(source)), layout, frames)
    assert 14 <= len(decoded) <= 16


@pytest.mark.skipif(FFMPEG is None, reason="FFmpeg is unavailable")
def test_vfr_source_is_conformed_once_without_chunk_seek_drift(tmp_path: Path) -> None:
    cfr = tmp_path / "cfr.mp4"
    vfr = tmp_path / "vfr.mkv"
    frames = tmp_path / "vfr-frames"
    make_source(cfr, 30)
    run("-i", str(cfr), "-vf", "select=not(eq(mod(n\\,5)\\,0))", "-fps_mode", "vfr", "-an", "-c:v", "mpeg4", str(vfr))
    layout = {"startTime": 0.2, "endTime": 1.7, "fps": 15.0, "width": 160, "height": 96}
    decoded = extract_range(SimpleNamespace(video_path=str(vfr)), layout, frames)
    assert 21 <= len(decoded) <= 24
    assert [path.name for path in decoded] == [f"frame-{index:05d}.png" for index in range(1, len(decoded) + 1)]
