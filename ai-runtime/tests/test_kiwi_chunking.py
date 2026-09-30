from types import SimpleNamespace

import pytest
from PIL import Image

from aiv_runtime.kiwi_edit_16gb import (
    LONG_OVERLAP_FRAMES,
    MAX_FRAMES,
    build_windows,
    merge_overlapping_frames,
    plan,
)


def test_stable_mode_enforces_the_supported_window() -> None:
    assert build_windows(MAX_FRAMES, "stable")[0]["modelFrames"] == MAX_FRAMES
    with pytest.raises(ValueError, match="Stable mode supports"):
        build_windows(MAX_FRAMES + 1, "stable")


def test_long_mode_covers_every_frame_with_valid_overlaps() -> None:
    windows = build_windows(200, "long", LONG_OVERLAP_FRAMES)
    assert windows[0]["startFrame"] == 0
    assert windows[-1]["startFrame"] + windows[-1]["sourceFrames"] == 200
    assert all(window["modelFrames"] <= MAX_FRAMES and window["modelFrames"] % 4 == 1 for window in windows)
    assert all(window["overlapFrames"] == LONG_OVERLAP_FRAMES for window in windows[1:])


@pytest.mark.parametrize("source_fps", [15.0, 24.0, 30.0])
def test_planning_uses_a_stable_model_cadence(source_fps: float) -> None:
    args = SimpleNamespace(start_time=0.5, end_time=10.5, working_fps=15.0, strategy="long", overlap_frames=25)
    layout = plan({"width": 1920, "height": 1080, "fps": source_fps, "duration": 20.0, "hasAudio": True}, args)
    assert layout["frames"] == 150
    assert layout["fps"] == 15.0
    assert layout["sourceFps"] == source_fps
    assert layout["requestedDuration"] == 10.0


def test_seam_analysis_preserves_length_and_finds_the_matching_region() -> None:
    old = [Image.new("RGB", (32, 24), (index * 4, 0, 0)) for index in range(40)]
    incoming = [Image.new("RGB", (32, 24), (60 + index * 4, 0, 0)) for index in range(25)]
    incoming.extend(Image.new("RGB", (32, 24), (255, index, 0)) for index in range(20))
    merged, seam = merge_overlapping_frames(old, incoming, 25, blend_radius=2)
    assert len(merged) == 60
    assert 4 <= seam["seam"] < 21
    assert seam["score"] >= 0
