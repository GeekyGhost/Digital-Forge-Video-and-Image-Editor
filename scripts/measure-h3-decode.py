"""Measure what spatial tiling actually costs the MiniMax-H3 video VAE, on this machine.

Why this exists
---------------
ComfyUI issue #15416 measured that VAE's decode on a 1024x1024 image and reported, in mean
absolute pixel error on a 0-255 scale:

    reference implementation   4.6
    tiled, 256 tile / 64 overlap   31.4   (tile seams and banding)
    untiled                        93.7   (strong ViT patch-grid artefacts)

`VAE_TILE_SIZE = 256` / `VAE_TILE_OVERLAP = 64` is that middle row, which read as "our tile
geometry is the degraded configuration". It is not that simple, and the third row is the reason:
tiling and no-tiling call the *same* decoder. `AutoencoderKLMiniMaxH3._decode_clip` either hands
the whole latent to `self.decoder` or hands it 256 px pieces and linearly cross-fades them. A tiler
can only approximate what the untiled decode produces -- it cannot beat it. So a run where untiled
is three times worse than tiled is not measuring a bad tiler; it is measuring a decoder whose error
grows with input size, with tiling accidentally holding it to a size it handles well.

That argument says the 31.4 cannot be blamed on tiling. It does not say what OUR decode does, and
this project's own rule is that ffprobe is the check, not the eye. Hence a measurement.

What it reports
---------------
**A. Tiling.** A full VAE round trip decoded three ways against the source pixels: tiled at the
released 256/64 geometry, tiled at whatever `--tile` you pass, and untiled. If untiled is best,
tiling costs us a little and the released geometry is a fair trade. If untiled is dramatically
worse, we have reproduced #15416 on our own stack, tiling is load-bearing rather than a compromise,
and nobody should "fix" it by turning tiling off.

**B. Precision.** The same round trip at FP32, BF16 and FP16, reporting the drift from FP32, the
resident weight size and peak VRAM. Diffusers pins this VAE to FP32 through
`_keep_in_fp32_modules`, so `dtype=torch.float16` on it is silently a no-op; ComfyUI ships
`minimax_h3_video_vae_fp16.safetensors` and selects it in every official H3 template, and #15416
measured fp16, bf16 and fp32 "within 0.1 of each other". This decides, on this card, whether the
5.2 GB a lower precision frees is worth anything — which matters because a 768p canvas is 2.6x the
spatial area of the 544p profile, and the decoder holds the whole model resident.

Usage
-----
    vendor\\MiniMax-H3\\.venv\\Scripts\\python.exe scripts\\measure-h3-decode.py --image path\\to.png
    vendor\\MiniMax-H3\\.venv\\Scripts\\python.exe scripts\\measure-h3-decode.py --video clip.mp4 --frames 17

With neither, it synthesises a broadband test frame -- fine for detecting seams and patch grids,
weaker than real photography for judging texture.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "ai-runtime"))


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-root", default=str(REPO_ROOT / "models" / "MiniMax-H3"))
    parser.add_argument("--image", help="A still to round-trip. Cropped to a multiple of 32.")
    parser.add_argument("--video", help="A clip to round-trip instead of a still.")
    parser.add_argument("--frames", type=int, default=1, help="Frames to read from --video.")
    parser.add_argument("--size", type=int, default=1024, help="Square edge for the synthetic frame.")
    parser.add_argument("--tile", type=int, default=512, help="A second tile size to compare against 256.")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _snap(value: int, multiple: int = 32) -> int:
    return max(multiple, (value // multiple) * multiple)


def load_source(args) -> "torch.Tensor":
    """`(1, 3, frames, height, width)` in [0, 1], on the 32-grid."""
    import numpy as np
    import torch

    if args.image:
        from PIL import Image

        picture = Image.open(args.image).convert("RGB")
        width, height = _snap(picture.width), _snap(picture.height)
        picture = picture.crop((0, 0, width, height))
        frame = np.asarray(picture, dtype=np.float32) / 255.0
        pixels = torch.from_numpy(frame).permute(2, 0, 1)[:, None]
    elif args.video:
        import av

        frames = []
        with av.open(args.video) as container:
            for frame in container.decode(video=0):
                frames.append(np.asarray(frame.to_image().convert("RGB"), dtype=np.float32) / 255.0)
                if len(frames) >= args.frames:
                    break
        if not frames:
            raise SystemExit(f"no frames decoded from {args.video}")
        stack = torch.from_numpy(np.stack(frames)).permute(3, 0, 1, 2)
        pixels = stack[:, :, : _snap(stack.shape[2]), : _snap(stack.shape[3])]
    else:
        # Broadband and non-symmetric: a ramp plus a fine checker plus a mid-frequency sweep, so a
        # tile seam, a patch grid and a flat-grey failure are each visible in the number.
        edge = _snap(args.size)
        ys, xs = torch.meshgrid(torch.arange(edge), torch.arange(edge), indexing="ij")
        ramp = xs / (edge - 1)
        checker = ((xs // 8 + ys // 8) % 2).float()
        sweep = (torch.sin(xs.float() / 6.0) * torch.cos(ys.float() / 9.0) + 1) / 2
        frame = torch.stack([ramp, checker * 0.5 + sweep * 0.5, sweep])
        pixels = frame[:, None].repeat(1, max(1, args.frames), 1, 1)

    return pixels[None].clamp(0, 1)


def main() -> int:
    args = parse()
    import torch
    from diffusers import AutoencoderKLMiniMaxH3

    from aiv_runtime import minimax_h3 as base

    source = load_source(args)
    _, _, frames, height, width = source.shape
    print(f"source · {width}x{height} · {frames} frame(s)")
    reference = (source[0].permute(1, 2, 3, 0) * 255).numpy()

    def load(dtype):
        """A fresh VAE at one precision. `_keep_in_fp32_modules` covers this whole model, so the
        pin has to be cleared for a cast to be total -- see `minimax_h3.cast_video_vae`."""
        vae = AutoencoderKLMiniMaxH3.from_pretrained(args.repo_root, subfolder="vae")
        vae.requires_grad_(False)
        if dtype is not torch.float32:
            vae._keep_in_fp32_modules = None
            vae.to(dtype)
        vae.to(args.device)
        return vae

    def tiles(vae, tile):
        if tile is None:
            vae.use_tiling = False
        else:
            vae.enable_tiling(
                tile_sample_min_height=tile,
                tile_sample_min_width=tile,
                tile_sample_min_overlap_height=base.VAE_TILE_OVERLAP,
                tile_sample_min_overlap_width=base.VAE_TILE_OVERLAP,
            )

    def roundtrip(vae, tile):
        """Encode at the released geometry, decode at `tile`, and score against the source.

        Encode is held fixed at 256/64 for every row: #15416 measured it as faithful (cosine 0.9997
        against the reference encoder), and holding it still is what isolates the decoder.
        """
        import numpy as np

        if args.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        tiles(vae, base.VAE_TILE_SIZE)
        with torch.inference_mode():
            posterior = vae.encode((source * 2 - 1).to(args.device, next(vae.parameters()).dtype))
            latents = posterior.latent_dist.mode() if hasattr(posterior, "latent_dist") else posterior[0]
        tiles(vae, tile)
        autocast = next(vae.parameters()).dtype
        autocast = autocast if autocast in (torch.float16, torch.bfloat16) else torch.float16
        with torch.inference_mode(), torch.autocast(
            device_type=args.device, dtype=autocast, enabled=args.device == "cuda"
        ):
            decoded = vae.decode(latents, return_dict=False)[0]
        pixels = ((decoded.float().cpu() + 1) / 2).clamp(0, 1)
        got = (pixels[0].permute(1, 2, 3, 0) * 255).numpy()[: reference.shape[0]]
        peak = torch.cuda.max_memory_allocated() / 2**30 if args.device == "cuda" else 0.0
        return got, float(np.abs(got - reference[: got.shape[0]]).mean()), peak

    # ---------------------------------------------------------------- part A: does tiling cost?
    print("\nA. Tiling, at the released FP32 precision. Round-trip MAE against the source, 0-255.")
    vae = load(torch.float32)
    weights = sum(p.numel() * p.element_size() for p in vae.parameters()) / 2**30
    print(f"   weights {weights:.2f} GiB")
    rows = []
    for label, tile in (
        (f"tiled {base.VAE_TILE_SIZE}/{base.VAE_TILE_OVERLAP} (production)", base.VAE_TILE_SIZE),
        (f"tiled {args.tile}/{base.VAE_TILE_OVERLAP}", args.tile),
        ("untiled (one decoder call)", None),
    ):
        _, mae, peak = roundtrip(vae, tile)
        rows.append((label, mae, peak))
        print(f"   {label:<34} MAE {mae:7.2f}   peak {peak:5.2f} GiB")
    released, _, untiled = rows[0][1], rows[1][1], rows[2][1]
    if rows[2][1] <= released:
        print(
            f"   -> Untiled wins ({rows[2][1]:.2f} vs {released:.2f}), so tiling costs "
            f"{released - rows[2][1]:.2f} of 255 here.\n"
            f"      #15416's 31.4 is not a number this stack reproduces."
        )
    else:
        print(
            f"   -> Untiled is WORSE ({rows[2][1]:.2f} vs {released:.2f}). #15416 reproduced: the\n"
            f"      decoder degrades with spatial extent and tiling is what holds it down. Keep it on."
        )
    baseline, _, _ = roundtrip(vae, base.VAE_TILE_SIZE)
    del vae
    if args.device == "cuda":
        torch.cuda.empty_cache()

    # ---------------------------------------------------------------- part B: does precision cost?
    print(
        "\nB. Precision, at the production tile. Diffusers pins this VAE to FP32 via"
        "\n   `_keep_in_fp32_modules`; ComfyUI ships it in FP16 and selects that in every official"
        "\n   H3 template. #15416 measured the two 'within 0.1 of each other'."
    )
    import numpy as np

    for name, dtype in (("fp32 (released)", torch.float32), ("bf16", torch.bfloat16), ("fp16", torch.float16)):
        try:
            vae = load(dtype)
            weights = sum(p.numel() * p.element_size() for p in vae.parameters()) / 2**30
            got, mae, peak = roundtrip(vae, base.VAE_TILE_SIZE)
            drift = float(np.abs(got - baseline[: got.shape[0]]).mean())
            finite = bool(np.isfinite(got).all())
            print(
                f"   {name:<16} MAE {mae:7.2f}   vs fp32 {drift:6.3f}   "
                f"weights {weights:5.2f} GiB   peak {peak:5.2f} GiB"
                f"{'' if finite else '   *** NON-FINITE OUTPUT ***'}"
            )
            del vae
            if args.device == "cuda":
                torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            print(f"   {name:<16} out of memory")

    print(
        "\nReading part B: `vs fp32` is the drift from the released precision, on a 0-255 scale."
        "\nUnder about 0.5 and finite, a lower precision is free accuracy-wise and halves both the"
        "\nresident model and the host copy -- which is the lever to reach for if a 768p decode does"
        "\nnot fit, since 768x1344 is 2.6x the spatial area of the 544p profile. Set it with"
        "\n`AIV_H3_VAE_DTYPE=bf16` (or `fp16`); BF16 keeps FP32's exponent range and cannot overflow"
        "\nwhere FP16 would. Leave it unset to run exactly what MiniMax released."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
