"""Pins the tiled-upscale overlap blend. Needs torch; no weights and no GPU."""

from __future__ import annotations

import torch

from aiv_runtime.upscale import EsrganBackend

SCALE = 2
TILE = 64
OVERLAP = 16


class BorderBlindModel:
    """A 2x upsampler that is wrong within six pixels of its own input border.

    That is not a contrived fault: a convolutional model sees only padding outside its input, so
    its outermost rows and columns are reconstructed from nothing. Hiding that is the entire job
    of the overlap, which is why a model without it cannot test the blend.
    """

    scale = SCALE

    def __call__(self, tensor):
        upsampled = torch.nn.functional.interpolate(tensor, scale_factor=SCALE, mode="nearest")
        edge = 6 * SCALE
        error = torch.zeros_like(upsampled)
        error[:, :, :edge, :] = 0.35
        error[:, :, -edge:, :] = 0.35
        error[:, :, :, :edge] = 0.35
        error[:, :, :, -edge:] = 0.35
        return (upsampled + error).clamp(0, 1)


def backend(flat: bool = False, tile: int = TILE) -> EsrganBackend:
    value = EsrganBackend.__new__(EsrganBackend)
    value.descriptor = BorderBlindModel()
    value.tile, value.overlap, value.scale, value.device = tile, OVERLAP, SCALE, "cpu"
    if flat:
        # What this used to do: accumulate every tile at weight 1 and divide by the count.
        value._feather = staticmethod(
            lambda height, width, device, dtype: torch.ones(
                1, 1, height, width, dtype=dtype, device=device
            )
        )
    return value


def ramp_image(height: int = 256, width: int = 256):
    """A smooth gradient: a seam shows as a discontinuity instead of hiding in texture."""
    rows, columns = torch.meshgrid(
        torch.linspace(0, 1, height), torch.linspace(0, 1, width), indexing="ij"
    )
    return torch.stack([rows, columns, (rows + columns) / 2])[None]


def test_feathering_beats_the_box_average_it_replaced():
    """The overlap has to be a blend, not a mean.

    Averaging two tiles across the overlap does not remove a seam: near its own border each tile
    is wrong in a different way, and the mean of two wrong answers is a visible band at every tile
    boundary. Weighting each tile by how far the pixel sits from that tile's border does remove it.
    """
    image = ramp_image()
    truth = torch.nn.functional.interpolate(image, scale_factor=SCALE, mode="nearest")
    interior = (slice(None), slice(None), slice(80, -80), slice(80, -80))

    flat = (backend(flat=True).run(image) - truth)[interior].abs().mean().item()
    feathered = (backend().run(image) - truth)[interior].abs().mean().item()
    assert feathered < flat / 2, f"feathering must at least halve the error: {feathered} vs {flat}"


def test_a_single_tile_input_passes_through_exactly():
    """The normalisation must divide by the accumulated weight, not by `clamp(min=1.0)`.

    Feathered weights are below 1 almost everywhere. Clamping the divisor up to 1 -- which the
    first version of this did -- darkens every seam and every image border, which is the artefact
    the feathering exists to remove.
    """
    image = ramp_image()[:, :, :TILE, :TILE]
    assert torch.allclose(backend().run(image), BorderBlindModel()(image), atol=1e-6)


def test_tiling_preserves_geometry_for_awkward_sizes():
    """Odd sizes, non-square frames, and the clamped last tile."""
    image = ramp_image()
    for height, width in [(256, 256), (201, 201), (129, 100), (200, 97), (97, 200), (65, 65), (33, 33)]:
        result = backend().run(image[:, :, :height, :width])
        assert tuple(result.shape[2:]) == (height * SCALE, width * SCALE), (height, width)
        assert torch.isfinite(result).all(), (height, width)


def test_the_feather_window_is_positive_everywhere():
    """A zero anywhere in the window would divide by the clamp floor and print as a black pixel."""
    for height, width in [(1, 1), (2, 7), (64, 64), (65, 129)]:
        window = EsrganBackend._feather(height, width, torch.device("cpu"), torch.float32)
        assert tuple(window.shape) == (1, 1, height, width), (height, width)
        assert window.min() > 0.0, (height, width, window.min().item())


# --------------------------------------------------------------- the batched frame loop

class FakeBackend:
    """A backend that records how it was called, so the loop can be tested without a model."""

    scale = 2
    device = "cpu"

    def __init__(self, batched: bool = True):
        self.batched = batched
        self.calls: list[int] = []

    @property
    def dtype(self):
        return torch.float32

    def run(self, tensor):
        return torch.nn.functional.interpolate(tensor, scale_factor=self.scale, mode="nearest")

    def run_many(self, tensors):
        self.calls.append(len(tensors))
        if not self.batched:
            return [self.run(tensor) for tensor in tensors]
        return [self.run(tensor) for tensor in tensors]


def frame_dir(tmp_path, count: int, size=(64, 48)):
    from PIL import Image

    source = tmp_path / "in"
    source.mkdir()
    for index in range(1, count + 1):
        # A distinct value per frame, so a loop that repeats or reorders is caught rather than
        # producing plausible-looking output.
        pixels = torch.full((size[1], size[0], 3), float(index) / 255.0)
        Image.fromarray((pixels.numpy() * 255).astype("uint8")).save(
            source / f"{index:08d}.png"
        )
    return source


def test_the_batched_loop_writes_every_frame_in_order(tmp_path):
    from PIL import Image

    from aiv_runtime.upscale import upscale_directory

    source = frame_dir(tmp_path, 10)
    destination = tmp_path / "out"
    destination.mkdir()
    frames = sorted(source.iterdir())
    backend = FakeBackend()
    written, size = upscale_directory(backend, frames, destination, 4, 2, "fake")

    assert written == 10
    assert size == (128, 96)
    assert sorted(p.name for p in destination.iterdir()) == [f"{i:08d}.png" for i in range(1, 11)]
    # Each output must carry its OWN source frame's value. A batch that reused one frame's result
    # across the group, or that wrote results against the wrong stems, lands here.
    for index in range(1, 11):
        with Image.open(destination / f"{index:08d}.png") as opened:
            assert opened.convert("RGB").getpixel((0, 0))[0] == index, index
    assert backend.calls == [4, 4, 2]


def test_the_loop_is_identical_without_threads_or_batching(tmp_path):
    from PIL import Image

    from aiv_runtime.upscale import upscale_directory

    source = frame_dir(tmp_path, 7)
    results = {}
    for label, batch, workers in [("batched", 4, 3), ("inline", 1, 0)]:
        destination = tmp_path / label
        destination.mkdir()
        written, _ = upscale_directory(
            FakeBackend(), sorted(source.iterdir()), destination, batch, workers, "fake"
        )
        assert written == 7, label
        results[label] = [
            Image.open(destination / f"{i:08d}.png").convert("RGB").getpixel((0, 0))
            for i in range(1, 8)
        ]
    assert results["batched"] == results["inline"]


def test_batch_size_budgets_output_pixels_not_frames():
    from aiv_runtime.upscale import BATCH_PIXEL_BUDGET, batch_size

    # A 4K output is 16x the pixels of 1080p, so a fixed frame count would ask for 16x the memory.
    assert batch_size(0, 1920, 1080, 2) < batch_size(0, 480, 270, 2)
    assert batch_size(0, 3840, 2160, 1) * 3840 * 2160 <= BATCH_PIXEL_BUDGET
    # An explicit request wins, and 1 is how batching is turned off.
    assert batch_size(3, 1920, 1080, 4) == 3
    assert batch_size(1, 64, 64, 2) == 1
    # Never zero, however large the frame.
    assert batch_size(0, 16000, 16000, 4) == 1


def test_a_backend_that_drops_frames_is_caught(tmp_path):
    from aiv_runtime.upscale import upscale_directory

    class Dropping(FakeBackend):
        def run_many(self, tensors):
            return [self.run(tensor) for tensor in tensors][:-1] or [self.run(tensors[0])]

    source = frame_dir(tmp_path, 6)
    destination = tmp_path / "out"
    destination.mkdir()
    try:
        upscale_directory(Dropping(), sorted(source.iterdir()), destination, 3, 0, "fake")
    except RuntimeError as error:
        assert "batch" in str(error)
    else:
        raise AssertionError("a short batch must not pass silently")
