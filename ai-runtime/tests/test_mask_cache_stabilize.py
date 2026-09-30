"""VFX Phase 7: the matte stabilisation baked once into the matte cache, for the compositor."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from aiv_runtime import mask_cache, mask_effects


def _cache(root: Path, frames: list[np.ndarray]) -> Path:
    cache = root / "cache"
    (cache / "masks").mkdir(parents=True)
    for index, frame in enumerate(frames):
        Image.fromarray(frame, mode="L").save(cache / "masks" / f"{index:05d}.png")
    (cache / "manifest.json").write_text(json.dumps({"cacheId": "x", "frameCount": len(frames), "width": 16, "height": 8, "fps": 24}))
    return cache


def test_stabilize_matches_the_old_renderer_and_is_reused(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    frames = [(rng.random((8, 16)) * 255).astype(np.uint8) for _ in range(5)]
    cache = _cache(tmp_path, frames)
    result = mask_cache.stabilize(cache, 2, 0)
    assert result["cachePath"].endswith("stable-s2-c0") and not result["reused"]
    expected = mask_effects.stabilize_masks([Image.fromarray(frame, mode="L") for frame in frames], 2, 0)
    for index, image in enumerate(expected):
        with Image.open(Path(result["cachePath"]) / "masks" / f"{index:05d}.png") as baked:
            assert np.array_equal(np.asarray(baked), np.asarray(image))
    assert mask_cache.stabilize(cache, 2, 0)["reused"]
