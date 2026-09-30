"""AI upscaler for stills and for extracted video frames. Two backends, one frame loop.

**spandrel** runs the Real-ESRGAN weights. Deliberately not the upstream `realesrgan`/`basicsr`
packages: `basicsr` imports `torchvision.transforms.functional_tensor`, which torchvision removed
in 0.17, and upstream has left that broken. Spandrel is inference-only, reads the *same* published
`.pth` state dicts, and detects the architecture and scale from the weights themselves -- so
nothing here hardcodes `num_block` / `num_feat` per model and gets one of them subtly wrong.

**nvvfx** is NVIDIA's VideoSuperRes, from the `nvidia-vfx` wheel. Its weights ship inside the
package, so there is no model pack and no weight file. The important difference from spandrel is
not speed but geometry: spandrel models produce exactly their trained factor, while VideoSuperRes
takes an arbitrary output size up to 4x. A chain targeting 2x therefore runs ESRGAN at 4x and
resamples back down -- two interpolations of the model's output -- where nvvfx simply produces 2x.

One job per process, like every other worker here: process exit is the VRAM reclamation boundary.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

# How many *output* pixels one batch may hold. 2**24 is the figure the ComfyUI RTX VSR workflow
# uses to keep a batch inside VRAM, and it is a sensible budget rather than a hardware limit:
# batching by frame count would mean 8 frames of 4K and 8 frames of 480p ask for wildly different
# allocations. Two 4K frames or eight 1080p frames both land here.
BATCH_PIXEL_BUDGET = 16_777_216


def emit(event: str, **payload: Any) -> None:
    print(json.dumps({"event": event, **payload}, ensure_ascii=False), flush=True)


def log(message: str) -> None:
    """Progress goes to stderr; stdout carries the JSON events the Rust side parses."""
    print(f"[upscale] {message}", file=sys.stderr, flush=True)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Real-ESRGAN / NVIDIA VFX upscaler")
    value.add_argument("--backend", choices=("esrgan", "nvvfx"), default="esrgan")
    value.add_argument("--weights", help="a .pth from models/real-esrgan/ (esrgan backend only)")
    value.add_argument("--input", required=True, help="an image file, or a directory of frames")
    value.add_argument("--output", required=True, help="matching file, or directory")
    value.add_argument(
        "--tile", type=int, default=512,
        help="esrgan only: tile edge in pixels; 0 disables. VRAM scales with the square of this.",
    )
    value.add_argument(
        "--tile-overlap", type=int, default=32,
        help="blend margin between tiles. Too small and the seams show as faint gridlines.",
    )
    value.add_argument(
        "--quality", default="HIGH",
        help="nvvfx only: a VideoSuperRes.QualityLevel name, e.g. HIGH or HIGHBITRATE_ULTRA.",
    )
    value.add_argument(
        "--target-width", type=int, default=0,
        help="nvvfx only, and required there: the effect is constructed around a fixed output size.",
    )
    value.add_argument("--target-height", type=int, default=0)
    value.add_argument("--device", default="cuda")
    value.add_argument(
        "--batch", type=int, default=0,
        help="frames per model call; 0 sizes it from the output resolution, 1 disables batching.",
    )
    value.add_argument(
        "--io-workers", type=int, default=4,
        help="threads decoding and encoding PNGs alongside the model. 0 runs everything inline.",
    )
    return value


# ---------------------------------------------------------------- tensor plumbing

def to_tensor(image, device, dtype):
    """HWC uint8 image -> contiguous (1, 3, H, W) float32 in [0, 1] on the target device.

    `.contiguous()` is load-bearing, not tidiness. `permute` returns a *view* with reordered
    strides, and `.to()` preserves that layout, so the tensor stays non-contiguous all the way
    down. PyTorch operators handle strided tensors transparently, which is why the spandrel path
    never noticed -- but NVIDIA's VFX API is a C boundary that needs a real packed buffer, and it
    rejects anything else with "Input tensor must be contiguous."
    """
    import numpy as np
    import torch

    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return (
        torch.from_numpy(array)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .contiguous()
        .to(device=device, dtype=dtype)
    )


def to_image(tensor):
    import numpy as np
    from PIL import Image

    array = tensor.squeeze(0).permute(1, 2, 0).clamp(0, 1).float().cpu().numpy()
    return Image.fromarray((array * 255.0 + 0.5).astype(np.uint8))


# ---------------------------------------------------------------- spandrel backend

class EsrganBackend:
    """Real-ESRGAN through spandrel. Fixed factor, tiled by us."""

    def __init__(self, weights: Path, device: str, tile: int, overlap: int):
        import torch
        from spandrel import ImageModelDescriptor, ModelLoader

        if not weights.is_file():
            raise FileNotFoundError(f"weights not found: {weights}")
        descriptor = ModelLoader(device=torch.device(device)).load_from_file(str(weights))
        if not isinstance(descriptor, ImageModelDescriptor):
            raise TypeError(f"{weights.name} is not a single-image model")
        descriptor.eval()
        self.descriptor = descriptor
        self.device = device
        self.tile = tile
        self.overlap = overlap
        self.scale = descriptor.scale
        log(
            f"{weights.name} · {descriptor.architecture.name} · x{descriptor.scale} · "
            f"{descriptor.input_channels}ch · {device}"
        )

    @property
    def dtype(self):
        return next(self.descriptor.model.parameters()).dtype

    @staticmethod
    def _feather(height: int, width: int, device, dtype):
        """A separable tent: each tile is weighted by how far a pixel sits from that tile's border.

        This is what makes the overlap a *blend*. Accumulating each tile at a flat weight of 1 and
        dividing by the count -- which is what this used to do -- is a box average, and a box
        average of two tiles does not remove a seam: near its own border a convolutional model has
        only padding to work with, so both contributions are wrong there in different ways and the
        mean of them is a visible band at every tile boundary. Ramping the weight to zero at the
        border means a tile contributes nothing where it knows nothing.

        Feathering every side, including the ones that sit on the true image border, is safe
        because the result is normalised by the accumulated weight: where only one tile
        contributes, `patch * w / w` is that patch exactly, whatever `w` is.

        Measured against an untiled reference on a synthetic model whose output is wrong within
        six pixels of its own input border, 64 px tiles with 16 px of overlap at 2x:

        | weighting | mean abs error |
        |---|---|
        | flat 1.0 (what this used to do) | 0.0924 |
        | linear ramp over half the overlap | 0.0510 |
        | linear ramp over the whole overlap | 0.0371 |
        | **this tent** | **0.0355** |
        """
        import torch

        def axis(length: int):
            edge = length // 2
            if edge == 0:
                # A one-pixel axis has no border to fade from; anything else would return a window
                # of the wrong length and broadcast the patch into the accumulator misaligned.
                return torch.ones(length, device=device, dtype=torch.float32)
            values = torch.arange(1, edge + 1, device=device, dtype=torch.float32) / (edge + 1)
            if length % 2:
                return torch.cat([values, values[-1:], values.flip(0)])
            return torch.cat([values, values.flip(0)])

        window = axis(height)[:, None] * axis(width)[None, :]
        return window.to(dtype=dtype)[None, None, :, :]

    def run_many(self, tensors):
        """Real-ESRGAN is a single-image model, so a batch is a loop.

        It still runs under the batched frame loop, because the loop's other job -- decoding and
        encoding PNGs on worker threads while the GPU works -- helps this backend just as much.
        A 4K PNG encode is on the order of a tenth of a second of CPU, and there are thousands.
        """
        return [self.run(tensor) for tensor in tensors]

    def run(self, tensor):
        """Run the network, tiled when asked.

        Tiling is not optional at 4K: VRAM scales with input pixel count, and a 4K frame at 4x will
        not fit un-tiled on a 16 GB card alongside anything else. The overlap is feathered rather
        than butted or averaged so tile boundaries do not print as a grid -- barely visible on a
        still, and extremely visible once it crawls across a moving frame.
        """
        import torch

        descriptor, tile, overlap = self.descriptor, self.tile, self.overlap
        if tile <= 0:
            with torch.no_grad():
                return descriptor(tensor)

        scale = descriptor.scale
        _, _, height, width = tensor.shape
        if height <= tile and width <= tile:
            with torch.no_grad():
                return descriptor(tensor)

        output = torch.zeros(
            (1, tensor.shape[1], height * scale, width * scale),
            device=tensor.device, dtype=tensor.dtype,
        )
        weight = torch.zeros(
            (1, 1, height * scale, width * scale), device=tensor.device, dtype=tensor.dtype
        )
        step = max(1, tile - overlap)
        for top in range(0, height, step):
            for left in range(0, width, step):
                bottom = min(top + tile, height)
                right = min(left + tile, width)
                top_ = max(0, bottom - tile)
                left_ = max(0, right - tile)
                with torch.no_grad():
                    patch = descriptor(tensor[:, :, top_:bottom, left_:right])
                window = self._feather(
                    patch.shape[2], patch.shape[3], patch.device, patch.dtype
                )
                rows = slice(top_ * scale, top_ * scale + patch.shape[2])
                columns = slice(left_ * scale, left_ * scale + patch.shape[3])
                output[:, :, rows, columns] += patch * window
                weight[:, :, rows, columns] += window
        # NOT `clamp(min=1.0)`. Feathered weights are below 1 almost everywhere, and clamping the
        # divisor up to 1 would darken every seam and every image border -- the exact artefact this
        # is here to remove. Clamp only away from a division by zero.
        return output / weight.clamp(min=1e-6)


# ---------------------------------------------------------------- nvvfx backend

class NvvfxBackend:
    """NVIDIA VideoSuperRes. Arbitrary output size up to 4x, tiling handled internally."""

    def __init__(self, quality: str, width: int, height: int, device: str):
        import torch
        from nvvfx import VideoSuperRes

        if device != "cuda":
            raise RuntimeError("the NVIDIA VFX backend requires CUDA; it has no CPU path")
        if width <= 0 or height <= 0:
            raise ValueError("the NVIDIA VFX backend needs an explicit --target-width/--target-height")
        try:
            level = getattr(VideoSuperRes.QualityLevel, quality)
        except AttributeError as error:
            names = ", ".join(member.name for member in VideoSuperRes.QualityLevel)
            raise ValueError(f"unknown quality level {quality!r}; available: {names}") from error

        # The effect is built around one fixed output size, so a directory of frames must be
        # uniform. Input size is inferred per call, which is why only the output is set here.
        effect = VideoSuperRes(quality=level)
        effect.output_width = width
        effect.output_height = height
        effect.load()
        self.minimum_short_edge = 360
        self.effect = effect
        self.torch = torch
        self.device = "cuda"
        self.size = (width, height)
        #: None until a batched call has been tried. See `run_many`.
        self.batched = None
        log(f"NVIDIA VideoSuperRes · {quality} · target {width}x{height} · cuda")

    @property
    def dtype(self):
        # The documented contract: channels-first RGB float32 in [0, 1].
        return self.torch.float32

    def run_many(self, tensors):
        """One call for the whole batch where the binding allows it, a loop where it does not.

        **On the wheel installed as of 2026-08-19 it does not.** `scripts/probe-nvvfx.py` handed it
        stacks of 2, 4 and 8 frames and got `ValueError: too many values to unpack (expected 3)`
        every time -- the binding unpacks the shape into exactly three values, so `(3, H, W)` is
        the only thing it takes.

        The detection stays anyway. It costs one failed call per run, it is the difference between
        this file being right and being right by luck, and a later wheel that grows batch support
        gets picked up without anyone remembering to come back here. The speedup in the meantime
        comes from `upscale_directory` overlapping the PNG work with the model, which is the larger
        win regardless -- a batched `run()` would not have made the encoder any faster.

        A binding that *accepts* a batch but quietly returns fewer frames than it was given is
        handled the same way as one that refuses, because trusting it would repeat one frame across
        the whole group.
        """
        if len(tensors) == 1 or self.batched is False:
            return [self.run(tensor) for tensor in tensors]
        stack = self.torch.cat([tensor for tensor in tensors], dim=0)
        try:
            result = self.effect.run(stack.contiguous())
            output = self.torch.from_dlpack(result.image).clone()
        except Exception as error:  # noqa: BLE001 -- any refusal is the same decision
            self.batched = False
            log(f"batched run() refused ({type(error).__name__}: {error}); falling back to one frame per call")
            return [self.run(tensor) for tensor in tensors]
        if output.dim() != 4 or output.shape[0] != len(tensors):
            # Accepting the call is not the same as processing every frame. A binding that reads
            # only the first frame of the stack would return a 3-D result, or an N of 1, and
            # trusting it would silently repeat one frame across the whole batch.
            self.batched = False
            log(
                f"batched run() returned {tuple(output.shape)} for {len(tensors)} frames; "
                "falling back to one frame per call"
            )
            return [self.run(tensor) for tensor in tensors]
        if self.batched is None:
            self.batched = True
            log(f"batched run() accepted {len(tensors)} frames per call")
        return [output[index:index + 1] for index in range(output.shape[0])]

    def run(self, tensor):
        # The effect wants (3, H, W), not the (1, 3, H, W) the spandrel path uses.
        #
        # `.contiguous()` again, deliberately, even though to_tensor already guarantees it: this is
        # a hard requirement of the C API rather than a preference, and it is a no-op when the
        # tensor is already packed. Any future caller that hands this a view gets a copy instead of
        # an NvVFXError raised from inside the binding.
        result = self.effect.run(tensor.squeeze(0).contiguous())
        # The returned buffer is reused by the next run(), so clone before it can be overwritten.
        return self.torch.from_dlpack(result.image).clone().unsqueeze(0)


# ---------------------------------------------------------------- frame loop

def read_frame(backend, source: Path):
    """A file on disk as a tensor on the device. Safe to call from a worker thread."""
    from PIL import Image

    with Image.open(source) as opened:
        opened.load()
        return to_tensor(opened, backend.device, backend.dtype)


def warn_below_minimum(backend, width: int, height: int) -> None:
    """NVIDIA states 360p as the suggested minimum input for VideoSuperRes.

    Suggested, not enforced -- so this says so once rather than refusing. A 480x270 source will
    produce something; it just will not produce what the model is good at, and someone judging the
    upscaler on a small clip deserves to know that is what they are looking at.
    """
    minimum = getattr(backend, "minimum_short_edge", 0)
    if minimum and min(width, height) < minimum:
        log(
            f"WARNING: the source is {width}x{height} and this model's suggested minimum input is "
            f"{minimum}p on the short edge. It will run, but below that the result is not what the "
            "model was trained to produce."
        )


def write_frame(tensor, destination: Path) -> tuple[int, int]:
    """A tensor as a PNG. Also safe from a worker thread, and where most of the CPU time goes.

    PNG for frames and for anything without an explicit suffix: this output is either an
    intermediate on its way back into an encoder, or a finished still the user will convert
    deliberately. Neither wants a lossy stage inserted here without being asked.
    """
    result = to_image(tensor)
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.save(destination)
    return result.size


def upscale_one(backend, source: Path, destination: Path) -> tuple[int, int]:
    return write_frame(backend.run(read_frame(backend, source)), destination)


def batch_size(requested: int, width: int, height: int, scale) -> int:
    """How many frames one model call may carry.

    Sizing by frame count would mean a batch of 8 asks for 16x more memory at 4K than at 1080p, so
    the budget is in output pixels and the count falls out of the frame size. `requested` overrides
    it; 1 disables batching without needing a separate flag for that.
    """
    if requested > 0:
        return requested
    factor = scale if isinstance(scale, (int, float)) and scale > 0 else 1
    pixels = max(1, int(width * factor) * int(height * factor))
    return max(1, min(16, BATCH_PIXEL_BUDGET // pixels))


def upscale_directory(backend, frames: list[Path], destination: Path, batch: int, workers: int,
                      label: str):
    """Every frame, in order, in batches, with the PNG work overlapping the model.

    The model call is the part that needs the GPU; decoding the source PNG and encoding the result
    are pure CPU and, at 4K, take longer than people expect -- thousands of frames each way. Doing
    them on worker threads while the GPU is busy is most of the speedup here, and it applies to
    both backends whether or not the model itself accepts a batch.

    Order is preserved: results are written under the source frame's own stem, so nothing depends
    on completion order.
    """
    from PIL import Image

    with Image.open(frames[0]) as first:
        source_size = first.size
    warn_below_minimum(backend, source_size[0], source_size[1])
    size = batch_size(batch, source_size[0], source_size[1], getattr(backend, "scale", None))
    log(
        f"{len(frames)} frames · {source_size[0]}x{source_size[1]} source · "
        f"{size} frame{'' if size == 1 else 's'} per model call · {workers} io thread(s)"
    )
    emit("start", frames=len(frames), scale=getattr(backend, "scale", None),
         backend=label, batch=size)

    written = (0, 0)
    index = 0
    pool = ThreadPoolExecutor(max_workers=workers) if workers > 0 else None
    try:
        mapper = pool.map if pool else map
        pending = []
        for start in range(0, len(frames), size):
            group = frames[start:start + size]
            tensors = list(mapper(lambda frame: read_frame(backend, frame), group))
            results = backend.run_many(tensors)
            if len(results) != len(group):
                raise RuntimeError(
                    f"the backend returned {len(results)} frames for a batch of {len(group)}"
                )
            for frame, result in zip(group, results):
                target = destination / f"{frame.stem}.png"
                pending.append(
                    pool.submit(write_frame, result, target) if pool
                    else _Immediate(write_frame(result, target))
                )
            # Bounded, so a fast GPU cannot queue thousands of un-encoded 4K frames into host RAM
            # ahead of the disk. Two batches of slack is enough to keep the writers busy.
            while len(pending) > size * 2:
                written = pending.pop(0).result()
                index += 1
                emit("frame", index=index, total=len(frames), width=written[0], height=written[1])
        for future in pending:
            written = future.result()
            index += 1
            emit("frame", index=index, total=len(frames), width=written[0], height=written[1])
    finally:
        if pool:
            pool.shutdown(wait=True)
    return index, written


class _Immediate:
    """A already-finished stand-in for a Future, so the inline path shares one code path."""

    def __init__(self, value):
        self._value = value

    def result(self):
        return self._value


def build_backend(args, device: str):
    if args.backend == "nvvfx":
        return NvvfxBackend(args.quality, args.target_width, args.target_height, device)
    if not args.weights:
        raise SystemExit("the esrgan backend needs --weights")
    return EsrganBackend(Path(args.weights), device, args.tile, args.tile_overlap)


def main() -> int:
    args = parser().parse_args()
    import torch

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        if args.backend == "nvvfx":
            raise SystemExit("the NVIDIA VFX backend requires CUDA, which is not available")
        log("CUDA is not available; falling back to CPU. This will be extremely slow.")
        device = "cpu"

    source = Path(args.input)
    destination = Path(args.output)
    backend = build_backend(args, device)
    scale = getattr(backend, "scale", None)

    if source.is_dir():
        frames = sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        if not frames:
            raise SystemExit(f"no frames to upscale in {source}")
        destination.mkdir(parents=True, exist_ok=True)
        written, size = upscale_directory(
            backend, frames, destination, args.batch, max(0, args.io_workers), args.backend
        )
        if written != len(frames):
            raise SystemExit(f"wrote {written} frames for {len(frames)} inputs")
        emit("done", frames=written, width=size[0], height=size[1], backend=args.backend)
    else:
        emit("start", frames=1, scale=scale, backend=args.backend)
        width, height = upscale_one(backend, source, destination)
        emit("done", frames=1, width=width, height=height, outputPath=str(destination),
             backend=args.backend)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
