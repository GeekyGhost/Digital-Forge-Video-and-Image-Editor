"""Locate the MiniMax-H3 colour fault by round-tripping a known image through the video VAE.

The 2026-08-25 FL2VA qualification run produced correct content -- a blacksmith, an anvil, a
hammer, a spark burst, all moving correctly -- in wrong colour: a blue/red inversion, a washed-out
range, and roughly six percent chroma speckle. The same "neon/negative" signature was recorded
against the portable-FP8 experiments, but it now reproduces on Comfy-Org's published INT8 ConvRot
weights that match their published SHA-256 exactly. So quantization is not the cause and the fault
is somewhere in encode -> denoise -> decode.

This bisects that chain without running the denoiser at all. It ImageNet-normalizes a known image
exactly as `encoders.encode_vae_condition` does, encodes it, decodes it back with the same
`_decoded_chunk_to_frames` the worker uses, and compares input pixels to output pixels.

Read the result like this:

    channels swap (input R mean ~= output B mean)   the VAE wrapper or the un-normalization has
                                                    the channel order wrong -- a post-processing
                                                    bug, fixable without touching sampling
    contrast collapses, hue holds                   a scale/normalization bug in the same place
    output ~= input                                 decode is sound; the fault is upstream, in the
                                                    denoise stage or the latents it produces
    output is speckled like the generation          the VAE itself is mis-loaded (dtype, tiling)

Usage:

    vendor\\MiniMax-H3\\.venv\\Scripts\\python.exe scripts\\bisect-h3-colour.py --image <png>

Defaults to the qualification colour-bar reference, which is the harshest case: saturated
primaries, a greyscale ramp and hard edges.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PACK = REPO_ROOT / "models" / "MiniMax-H3"
sys.path.insert(0, str(REPO_ROOT / "ai-runtime"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def say(message: str = "") -> None:
    print(message, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--image",
        default=str(REPO_ROOT / ".aiv" / "h3-qualification" / "refs" / "ref-image.png"),
    )
    parser.add_argument("--output", default=str(REPO_ROOT / ".aiv" / "h3-qualification" / "vae-roundtrip.png"))
    args = parser.parse_args()

    import numpy as np
    import torch
    from PIL import Image

    from aiv_runtime.minimax_h3 import (
        H3_PIXEL_MEAN,
        H3_PIXEL_STD,
        VAE_TILE_OVERLAP,
        VAE_TILE_SIZE,
        _decoded_chunk_to_frames,
    )

    source = Path(args.image)
    if not source.is_file():
        say(f"no such image: {source}")
        return 1

    # RGB explicitly: an RGBA source would otherwise carry a fourth plane into a three-channel
    # convention, and several of the project's own reference images are RGBA.
    image = Image.open(source).convert("RGB")
    # Keep the test cheap: the colour question is scale-invariant, and a small canvas still
    # crosses the 256px tile boundary, so tiling is exercised rather than skipped.
    scale = min(1.0, 480 / max(image.size))
    width, height = (max(32, int(value * scale) - (int(value * scale) % 32)) for value in image.size)
    image = image.resize((width, height), Image.LANCZOS)
    # np.array copies; np.asarray would hand torch a read-only buffer and warn.
    original = np.array(image, dtype=np.uint8)
    say(f"  input  {source.name} · {width}x{height}")

    from diffusers import AutoencoderKLMiniMaxH3

    vae = AutoencoderKLMiniMaxH3.from_pretrained(PACK, subfolder="vae", torch_dtype=torch.float16)
    vae.requires_grad_(False)
    vae.eval()
    vae.to("cuda")

    # A single frame cannot survive this VAE's temporal decode. `_decode` computes
    # `num_chunks = (tokens + token_drop + pad) // tokens_chunk_size - 1`, which is ZERO below
    # seven latent frames, and `torch.cat` then gets an empty list. Two encoder clips of
    # `clip_length` frames give 2*5 - 3 = 7 latent frames, the smallest stack that decodes.
    # Repeating one still image also removes temporal alignment from the question: every output
    # frame should equal the input, whatever the padding and cross-fade do.
    clip_length = int(vae.config.clip_length)
    stack = clip_length * 2
    say(f"  stacking the image {stack}x — one frame decodes to zero temporal chunks")

    # Match the worker's tiling exactly, so this measures the path the generation took.
    vae.enable_tiling(
        tile_sample_min_height=VAE_TILE_SIZE,
        tile_sample_min_width=VAE_TILE_SIZE,
        tile_sample_min_overlap_height=VAE_TILE_OVERLAP,
        tile_sample_min_overlap_width=VAE_TILE_OVERLAP,
    )

    frame = torch.from_numpy(original).permute(2, 0, 1)
    pixels = frame[None, :, None].repeat(1, 1, stack, 1, 1).to("cuda")
    pixel_mean = torch.tensor(H3_PIXEL_MEAN, device="cuda").view(1, -1, 1, 1, 1)
    pixel_std = torch.tensor(H3_PIXEL_STD, device="cuda").view(1, -1, 1, 1, 1)
    normalized = (pixels.to(torch.float32).div(255.0) - pixel_mean) / pixel_std

    with torch.inference_mode():
        posterior = vae.encode(normalized.to(torch.float16), return_dict=False)[0]
        latents = posterior.sample(generator=torch.Generator().manual_seed(42))
        say(f"  latents {tuple(latents.shape)} — {latents.shape[2]} latent frames")
        decoded = vae.decode(latents, return_dict=False)[0]
    say(f"  decoded {tuple(decoded.shape)}")

    # The single most diagnostic number in this script. The VAE returns ImageNet-normalized RGB,
    # so a DARK input must leave the decoder NEGATIVE (below the mean) and a bright one positive.
    # The 2026-08-25 generation implied +1.3 to +2.1 for a scene that should have been about -1.2,
    # so if this line reports the wrong sign for a dark image the fault is at or before the
    # decoder, and everything downstream is innocent.
    raw = decoded.float()
    say()
    say("  raw decoder output, BEFORE any denormalization (ImageNet space, 0 = the mean):")
    for index, channel in enumerate("RGB"):
        plane = raw[0, index, raw.shape[2] // 2]
        say(f"    {channel}: mean {plane.mean().item():+7.3f}  min {plane.min().item():+7.3f}  max {plane.max().item():+7.3f}")
    source_luma = original.mean() / 255.0
    implied = float(raw.mean())
    say(f"  source image mean luma {source_luma:.3f} of 1.0")
    if source_luma < 0.45 and implied > 0.3:
        say("    ^ a DARK source decoded to a POSITIVE mean. The sign is wrong at the decoder.")
    elif source_luma > 0.55 and implied < -0.3:
        say("    ^ a BRIGHT source decoded to a NEGATIVE mean. The sign is wrong at the decoder.")
    else:
        say("    ^ sign is consistent with the source; look downstream of the decoder.")

    frames = _decoded_chunk_to_frames(decoded.float())
    result = frames[len(frames) // 2]
    say(f"  {len(frames)} frames out; comparing the middle one, away from the cross-fade seams")
    Image.fromarray(result).save(args.output)
    say(f"  output {Path(args.output).name}")
    say()

    say("  channel means, input -> output")
    means = []
    for index, channel in enumerate("RGB"):
        before = float(original[:, :, index].mean())
        after = float(result[:, :, index].mean())
        means.append((before, after))
        say(f"    {channel}: {before:6.1f} -> {after:6.1f}   ({after - before:+6.1f})")

    say()
    straight = sum(abs(a - b) for b, a in means)
    swapped = abs(means[0][1] - means[2][0]) + abs(means[1][1] - means[1][0]) + abs(means[2][1] - means[0][0])
    say(f"  total drift keeping channel order:  {straight:6.1f}")
    say(f"  total drift with R and B exchanged: {swapped:6.1f}")
    if swapped < straight * 0.6:
        say("  VERDICT: the round trip EXCHANGES RED AND BLUE. The fault is in the VAE wrapper or")
        say("           the un-normalization, not in sampling -- fixable in post-processing.")
    elif straight < 12:
        say("  VERDICT: the round trip PRESERVES colour. Decode is sound; the fault is upstream,")
        say("           in the denoise stage or the latents it hands over.")
    else:
        say("  VERDICT: colour drifts without a clean channel exchange. Compare the saved image")
        say("           with the input by eye before concluding -- check contrast and speckle.")

    contrast_before = float(original.std())
    contrast_after = float(result.std())
    say()
    say(f"  contrast (stdev): {contrast_before:.1f} -> {contrast_after:.1f}")
    if contrast_after < contrast_before * 0.6:
        say("  The range also COLLAPSED, which is the washed-out half of the generation fault.")

    # Inversion is the specific shape the 2026-08-25 generation had: `1 - true` recovers a clean
    # photographic image, so whatever went wrong happened in pixel space at the very end rather
    # than to the latents, which a non-linear VAE could not have inverted so cleanly.
    inverted_drift = sum(abs((255.0 - a) - b) for b, a in means)
    say()
    say(f"  total drift if the output were INVERTED: {inverted_drift:6.1f}")
    if inverted_drift < straight * 0.5:
        say("  VERDICT: the round trip INVERTS. The fault is in decode or the un-normalization,")
        say("           and it is reproducible without the denoiser -- fix it here.")

    # The export half, isolated. `_decoded_chunk_to_frames` hands RGB uint8 to `export_to_video`;
    # if that writer inverts or swaps, the fault is downstream of every tensor operation and no
    # amount of reading the decode maths will find it.
    say()
    say("  export round trip (write the SOURCE image through the worker's own writer, read back):")
    try:
        import imageio.v3 as iio
        from diffusers.utils import export_to_video

        probe = Path(args.output).with_name("export-probe.mp4")
        export_to_video([original] * 8, str(probe), fps=24)  # the source, not the round trip
        back = iio.imread(str(probe), index=4)
        drift = [float(abs(back[:, :, i].astype(float) - original[:, :, i].astype(float)).mean()) for i in range(3)]
        say(f"    per-channel mean |delta| after encode/decode: R {drift[0]:.1f}  G {drift[1]:.1f}  B {drift[2]:.1f}")
        swap = float(abs(back[:, :, 0].astype(float) - original[:, :, 2].astype(float)).mean())
        if max(drift) > 40 and swap < 20:
            say("    VERDICT: the WRITER exchanges red and blue. Nothing above it is at fault.")
        elif max(drift) > 40:
            say("    VERDICT: the WRITER mangles colour. Compare export-probe.mp4 with the source.")
        else:
            say("    writer is faithful (small drift is just H.264 being lossy); look upstream.")
    except Exception as error:  # noqa: BLE001
        say(f"    export probe SKIPPED ({error})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
