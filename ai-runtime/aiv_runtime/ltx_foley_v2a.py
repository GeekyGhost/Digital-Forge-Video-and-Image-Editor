"""Video-conditioned audio (foley) for an existing clip, on a 16 GB GPU.

This is the "special pipeline" the upstream `DubItPipeline` implies but does not offer: it
generates audio that is *synchronised to the picture*, and nothing else.

Why not just call DubItPipeline? Because two thirds of what it does is wasted here. Reading its
`__call__`:

    s1_audio_latent = audio_state.latent.clone()      # stage 1 produces the audio
    upscaled_video_latent = self.upsampler(...)       # stage 2 is video-only
    video_state, _audio_unused = self.stage(          # audio frozen=True, result discarded
        ..., audio=ModalitySpec(..., frozen=True, initial_latent=s1_audio_latent))
    decoded_audio = self.audio_decoder(s1_audio_latent)   # <- stage 1's latent, unchanged

The audio it returns is decoded from the **stage 1** latent. Stage 2 freezes the audio and throws
its output away; it exists purely to upscale the picture. Since we keep the user's original picture
untouched, stage 2 and the spatial upsampler are dead weight — so this adapter runs stage 1 only,
decodes the audio, and muxes it onto the source video.

That removes the upsampler from the VRAM budget entirely and roughly halves the work.

What it does still need is the Foley V2A IC-LoRA, which is what carries the video-conditioning
behaviour (`append_ic_lora_reference_video_conditionings` and the reference downscale factor are
both read from it). LoRA fusion is not supported on the scaled-FP8 streaming path, so this adapter
uses the plain distilled checkpoint with an FP8 cast policy and block streaming instead.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

# The LTX adapters run under the vendored LTX virtualenv, where this package is not installed, and
# Python only puts the *script's own* directory on sys.path -- not the package root. Without this,
# `import aiv_runtime.…` fails with ModuleNotFoundError depending on which interpreter is used.
_PACKAGE_ROOT = str(Path(__file__).resolve().parent.parent)
if _PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, _PACKAGE_ROOT)

from aiv_runtime.ltx_attention import (  # noqa: E402
    ATTENTION_BACKENDS,
    DEFAULT_ATTENTION_BACKEND,
    describe_attention,
    install_stage_attention,
)
from aiv_runtime.ltx_offload import (  # noqa: E402
    OFFLOAD_PROFILES,
    cpu_slots_for,
    describe_profile,
    offload_mode_for,
)
from aiv_runtime.ltx_geometry import (  # noqa: E402
    AXIS_MULTIPLE,
    FRAMES_PER_LATENT,
    snap_axes,
    snap_canvas,
    snap_frames,
)
from aiv_runtime.oom import ltx_degradations, run_with_recovery

# The conditioning canvas. The picture is never generated here -- only audio is -- so this has to
# be large enough for the video conditioning to be meaningful and no larger.
NOMINAL_PIXELS = 768 * 512

# The grid and the frame rule now come from `ltx_geometry`, which is where the three LTX adapters'
# four different answers were reconciled. This file used to round axes to 64: that is the TWO-stage
# divisor, and its own docstring says there is no stage 2 here.
SPATIAL_MULTIPLE = AXIS_MULTIPLE
FRAME_MULTIPLE = FRAMES_PER_LATENT

# What the conditioning stage runs at, as a fraction of the layout. DubIt halves *because it
# upsamples afterwards*; this adapter has no stage 2, so halving just discards half the resolution
# of the only signal the audio branch has to sync against -- and the IC-LoRA's own
# `reference_downscale_factor` would divide again on top of it, where a LoRA declares one -- the
# shipped foley adapter does not, so today the halving is the whole loss. 0.5 is the default until
# the peak-VRAM print below has been read at 1.0 on a real clip; the plan's Task 3.5 is that
# measurement and the default change that follows it.
DEFAULT_STAGE_SCALE = 0.5


def force_utf8_io() -> None:
    """diffusers and transformers emit emoji in warnings; Windows consoles default to cp1252."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LTX 2.3 video-conditioned foley for 16 GB GPUs.")
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--gemma-root", required=True)
    parser.add_argument("--lora-path", required=True)
    parser.add_argument("--video-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-strength", type=float, default=1.0)
    parser.add_argument("--reference-strength", type=float, default=1.0)
    # Audio CFG is the control that matters here: below ~1.5 the prompt has little say over the
    # generated sound, so that is the default rather than the unguided 1.0.
    parser.add_argument("--audio-cfg-scale", type=float, default=1.5)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    # How strongly the SOURCE clip's existing audio anchors the result.
    #
    # DubIt derives its audio reference from the same clip as its video reference, because in
    # dubbing that reference IS the voice to imitate. Foley is the opposite task: the picture
    # should drive the sound and the source track is exactly what we are replacing. Inherited
    # unchanged, `AudioConditionByReferenceLatent(..., strength=1.0)` appends the clip's own audio
    # as CLEAN, attended tokens at negative RoPE positions -- a continuation prefix. On a clip with
    # music under it the model dutifully continues the music and the text prompt loses.
    #
    # `t2a_one_stage.py` is the reference for prompt-driven audio and passes
    # `ModalitySpec(context=...)` with no conditionings at all, so 0.0 skips the reference
    # entirely rather than appending denoised tokens. Raise it only to deliberately keep some of
    # the original bed.
    parser.add_argument("--audio-reference-strength", type=float, default=0.0)
    # CPU streaming, not disk. Disk streaming crashes this machine natively: six `0xC0000005`
    # faults on 2026-08-11/12, every one inside `torch_cpu.dll` at the SAME offset 0x8e8b949, and
    # every one on the disk path. The identical clip completes cleanly under `cpu`. The suspect is
    # the pinned-buffer recycling around `sft_loader`'s `non_blocking=True` copies, which only the
    # disk path exercises. `disk` is kept as an option for machines short of RAM.
    parser.add_argument("--offload", choices=OFFLOAD_PROFILES, default="cpu")
    # Sage is not the default here, unlike H3: on H3 the kernels were installed and
    # merely unselected, whereas nothing is installed in the LTX venv until
    # scripts\install-ltx-attention.ps1 has been run. Asking for a backend that is not
    # there costs a log line, not a run.
    parser.add_argument(
        "--attention-backend", choices=ATTENTION_BACKENDS, default=DEFAULT_ATTENTION_BACKEND
    )
    # See DEFAULT_STAGE_SCALE. 1.0 conditions on the full layout; 0.5 is the DubIt fossil.
    parser.add_argument("--stage-scale", type=float, default=DEFAULT_STAGE_SCALE)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def validate(args: argparse.Namespace) -> None:
    for value in (args.checkpoint_path, args.gemma_root, args.lora_path, args.video_path):
        if not Path(value).exists():
            raise FileNotFoundError(value)
    if not args.prompt.strip():
        raise ValueError("prompt cannot be empty")
    if Path(args.output_path).suffix.lower() != ".mp4":
        raise ValueError("foley output must be an MP4 file")
    if not 0 < args.lora_strength <= 2:
        raise ValueError("lora-strength must be in (0, 2]")
    if not 0 < args.reference_strength <= 2:
        raise ValueError("reference-strength must be in (0, 2]")
    if not 0 <= args.audio_reference_strength <= 1:
        raise ValueError("audio-reference-strength must be in [0, 1]")
    for name, value in (("cfg-scale", args.cfg_scale), ("audio-cfg-scale", args.audio_cfg_scale)):
        if not 1.0 <= value <= 10.0:
            raise ValueError(f"{name} must be between 1.0 (unguided) and 10.0")


def probe(path: str) -> dict:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", path,
        ],
        capture_output=True, text=True, check=True,
    )
    payload = json.loads(result.stdout)
    streams = payload.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"), {})
    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    numerator, _, denominator = (video.get("avg_frame_rate") or "24/1").partition("/")
    try:
        fps = float(numerator) / float(denominator or 1)
    except (ValueError, ZeroDivisionError):
        fps = 24.0
    return {
        "width": int(video.get("width") or 0),
        "height": int(video.get("height") or 0),
        "fps": fps if fps > 0 else 24.0,
        "duration": float((payload.get("format") or {}).get("duration") or 0.0),
        "hasAudio": has_audio,
    }


def plan(source: dict) -> dict:
    """Nominal geometry for the conditioning pass, snapped to the model's constraints."""
    width, height = source["width"], source["height"]
    if width <= 0 or height <= 0:
        raise ValueError("the source clip has no decodable video stream")
    target_width, target_height = snap_canvas(width, height, NOMINAL_PIXELS)
    # Nearest, not floor. Flooring planned audio shorter than the clip on every source whose frame
    # count was not already 8n+1, and the mux then reconciled the two by discarding whichever ran
    # out first. The mux now carries the source's own duration explicitly, so the nearest count is
    # both closer to the truth and safe.
    frames = snap_frames(max(1, int(round(source["duration"] * source["fps"]))))
    return {
        "width": target_width,
        "height": target_height,
        "frames": frames,
        "fps": source["fps"],
    }


def ensure_audio_track(video_path: str, source: dict, scratch: Path) -> str:
    """The reference-audio encoder raises on a silent file, so give a silent clip a silent track.

    The conditioning that matters is the picture; this only satisfies the audio reference input.
    """
    if source["hasAudio"]:
        return video_path
    padded = scratch / "with-silence.mp4"
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-i", video_path,
            "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
            "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac",
            # anullsrc never ends, so the picture is what governs the length here -- but say so
            # explicitly rather than relying on `-shortest`, which is what truncated the output
            # mux. The reference clip's frame count is conditioning signal; it must not move.
            "-t", f"{source['duration']:.6f}",
            str(padded),
        ],
        check=True,
    )
    print("[ltx-foley] source had no audio track; added silence for the reference input", file=sys.stderr)
    return str(padded)


def generate_audio(args: argparse.Namespace, layout: dict, reference_path: str, scratch: Path) -> Path:
    """Stage 1 only: condition on the reference video, denoise, decode the audio latent."""
    import torch
    from ltx_core.components.noisers import GaussianNoiser
    from ltx_core.conditioning import AudioConditionByReferenceLatent
    from ltx_core.loader import LoraPathStrengthAndSDOps
    from ltx_core.model.audio_vae import encode_audio as vae_encode_audio
    from aiv_runtime.safetensors_reader import describe_backend, install_pread_state_dict_loader
    from ltx_core.quantization.fp8_cast import build_policy as build_fp8_cast_policy
    from ltx_core.quantization.fp8_scaled_mm import build_policy as build_fp8_scaled_policy

    from aiv_runtime import scaled_fp8_lora
    from ltx_pipelines.iclora_utils import (
        append_ic_lora_reference_video_conditionings,
        read_lora_reference_downscale_factor,
        read_lora_reference_temporal_scale_factor,
    )
    from ltx_pipelines.utils.blocks import (
        AudioConditioner, AudioDecoder, DiffusionStage, ImageConditioner, PromptEncoder,
    )
    from ltx_pipelines.utils.constants import DISTILLED_SIGMAS
    from ltx_core.components.guiders import MultiModalGuider, MultiModalGuiderParams
    from ltx_pipelines.utils.denoisers import GuidedDenoiser, SimpleDenoiser
    from ltx_pipelines.utils.helpers import combined_image_conditionings, get_device
    from ltx_pipelines.utils.media_io import decode_audio_from_file, encode_audio
    from ltx_core.model.video_vae import TilingConfig
    from ltx_pipelines.dubit import patchify_dubit_audio_reference_latent
    from ltx_pipelines.utils.types import ModalitySpec, OffloadMode

    device = get_device()
    dtype = torch.bfloat16
    offload_mode = offload_mode_for(args.offload)
    cpu_slots = cpu_slots_for(args.offload)
    print(f"[ltx] {describe_profile(args.offload)}", file=sys.stderr)

    # Either checkpoint works. A scaled-FP8 file (per-tensor `.weight_scale` companions) goes
    # through `scaled_fp8_lora`, which supplies the companion-key write-back the stock streaming
    # provider lacks -- that is what lets the IC-LoRA ride on the 27.5 GiB FP8 checkpoint rather
    # than demanding the 46 GB BF16 one. A plain BF16 checkpoint still takes the fp8-cast path.
    ic_lora = LoraPathStrengthAndSDOps(args.lora_path, args.lora_strength, None)
    scaled_fp8 = scaled_fp8_lora.is_scaled_fp8(args.checkpoint_path)
    quantization = (
        build_fp8_scaled_policy(args.checkpoint_path)
        if scaled_fp8
        else build_fp8_cast_policy(args.checkpoint_path)
    )
    print(
        f"[ltx-foley] {'scaled-FP8' if scaled_fp8 else 'BF16 + fp8-cast'} checkpoint - "
        f"IC-LoRA strength {args.lora_strength:.2f}",
        file=sys.stderr,
    )

    prompt_encoder = PromptEncoder(
        args.checkpoint_path, args.gemma_root, dtype, device, offload_mode=offload_mode
    )
    # CFG needs an unconditional branch, so the negative prompt is encoded alongside the positive
    # one whenever either scale is above 1.0.
    guided = args.cfg_scale > 1.0 or args.audio_cfg_scale > 1.0
    prompts = [args.prompt, args.negative_prompt] if guided else [args.prompt]
    encoded = prompt_encoder(prompts, enhance_first_prompt=False, enhance_prompt_image=None)
    video_context = encoded[0].video_encoding.cpu()
    audio_context = encoded[0].audio_encoding.cpu()
    negative_video_context = encoded[1].video_encoding.cpu() if guided else None
    negative_audio_context = encoded[1].audio_encoding.cpu() if guided else None
    del encoded, prompt_encoder
    torch.cuda.empty_cache()

    image_conditioner = ImageConditioner(args.checkpoint_path, dtype, device)
    audio_conditioner = AudioConditioner(args.checkpoint_path, dtype, device)
    # Both factors, not just the spatial one. `append_ic_lora_reference_video_conditionings`
    # accepts a temporal factor and `temporal_subsample` implements it -- keep frame 0, then every
    # Nth -- and we were reading the spatial factor and letting the temporal one default to 1. If a
    # Foley LoRA ever declares one, the reference video would have been fed at the wrong temporal
    # density relative to training, which desynchronises the one thing foley is for.
    #
    # Measured on the shipped adapter, 2026-09-08: `ltx-2.3-22b-lora-foley-v2a-1.0.safetensors`
    # declares neither factor -- its `__metadata__` holds only `model_version` and the licence --
    # so both are 1 today and this changes nothing for it. That is exactly why it is worth wiring
    # rather than remembering: the asymmetry would have gone unnoticed until a LoRA that does
    # declare one arrived, and then presented as bad sync rather than as a missing argument.
    install_pread_state_dict_loader()
    print(f"[ltx-foley] checkpoint read: {describe_backend()}", file=sys.stderr)
    downscale_factor = read_lora_reference_downscale_factor(args.lora_path)
    temporal_scale_factor = read_lora_reference_temporal_scale_factor(args.lora_path)
    print(
        f"[ltx-foley] IC-LoRA reference factors - spatial {downscale_factor}, "
        f"temporal {temporal_scale_factor}",
        file=sys.stderr,
    )

    # The conditioning stage's own geometry. Snapped back onto the grid rather than divided
    # blindly: the layout is on a 32 grid now, and half of a 32-aligned axis is not always
    # 32-aligned (832 // 2 == 416 is fine, 480 // 2 == 240 is not).
    stage_scale = max(0.05, min(1.0, float(args.stage_scale)))
    stage_width, stage_height = snap_axes(
        max(1, round(layout["width"] * stage_scale)),
        max(1, round(layout["height"] * stage_scale)),
    )
    print(
        f"[ltx-foley] conditioning stage {stage_width}x{stage_height} "
        f"(scale {stage_scale:g} of {layout['width']}x{layout['height']}); the IC-LoRA's "
        f"IC-LoRA reference factors divide again on top of this where the adapter declares them",
        file=sys.stderr,
    )

    def build_conditionings(encoder):
        conditionings = combined_image_conditionings(
            images=[], height=stage_height, width=stage_width,
            video_encoder=encoder, dtype=dtype, device=device,
        )
        append_ic_lora_reference_video_conditionings(
            conditionings,
            [(reference_path, args.reference_strength)],
            height=stage_height,
            width=stage_width,
            num_frames=layout["frames"],
            video_encoder=encoder,
            dtype=dtype,
            device=device,
            reference_downscale_factor=downscale_factor,
            reference_temporal_scale_factor=temporal_scale_factor,
            conditioning_attention_strength=1.0,
            conditioning_attention_mask=None,
            tiling_config=TilingConfig.default(),
        )
        return conditionings

    video_conditionings = image_conditioner(build_conditionings)

    if args.audio_reference_strength > 0:
        reference_audio = decode_audio_from_file(reference_path, device)
        if reference_audio is None:
            raise ValueError("the reference clip has no decodable audio stream")
        audio_reference_latent = audio_conditioner(lambda enc: vae_encode_audio(reference_audio, enc, None))
        ref_patch, ref_pos = patchify_dubit_audio_reference_latent(
            audio_reference_latent, negative_positions=True, device=device
        )
        audio_conditionings = [
            AudioConditionByReferenceLatent(ref_patch, ref_pos, strength=args.audio_reference_strength)
        ]
    else:
        # No reference tokens at all, matching `t2a_one_stage`. This also skips the audio VAE
        # encode, so it is cheaper as well as more faithful to the prompt.
        audio_conditionings = []
    del image_conditioner, audio_conditioner
    torch.cuda.empty_cache()

    stage = (
        scaled_fp8_lora.build_streaming_stage(
            args.checkpoint_path, dtype, device, offload_mode, quantization,
            loras=(ic_lora,), cpu_slots_count=cpu_slots,
        )
        if scaled_fp8
        else DiffusionStage.from_checkpoint(
            args.checkpoint_path, dtype, device,
            loras=(ic_lora,), quantization=quantization, offload_mode=offload_mode,
        )
    )
    generator = torch.Generator(device=device).manual_seed(args.seed)
    noiser = GaussianNoiser(generator=generator)
    sigmas = DISTILLED_SIGMAS.to(dtype=torch.float32, device=device)
    video_context = video_context.to(device)
    audio_context = audio_context.to(device)

    if guided:
        negative_video_context = negative_video_context.to(device)
        negative_audio_context = negative_audio_context.to(device)
        denoiser = GuidedDenoiser(
            video_context,
            audio_context,
            video_guider=MultiModalGuider(
                params=MultiModalGuiderParams(cfg_scale=args.cfg_scale),
                negative_context=negative_video_context,
            ),
            audio_guider=MultiModalGuider(
                params=MultiModalGuiderParams(cfg_scale=args.audio_cfg_scale),
                negative_context=negative_audio_context,
            ),
        )
    else:
        denoiser = SimpleDenoiser(video_context, audio_context)

    print(
        f"[ltx-foley] stage 1 only · {stage_width}x{stage_height} · {layout['frames']} frames · "
        f"{layout['fps']:.2f} fps · audio CFG {args.audio_cfg_scale:.2f} · "
        f"LoRA {args.lora_strength:.2f} · video reference {args.reference_strength:.2f} · "
        f"audio reference {'off' if not audio_conditionings else f'{args.audio_reference_strength:.2f}'} · "
        f"no upsampler",
        file=sys.stderr,
    )
    # Configure the stage after construction, then attach to the real transformer inside its
    # lazy `run`: there are no attention modules to swap before that build happens.
    active_attention, attention_swapped, attention_reason = install_stage_attention(
        args.attention_backend,
        stage,
        log=lambda message: print(f"[ltx-foley] {message}", file=sys.stderr),
    )
    print(
        f"[ltx] {describe_attention(args.attention_backend, active_attention, attention_swapped, attention_reason)}",
        file=sys.stderr,
    )
    _video_state, audio_state = stage(
        denoiser=denoiser,
        sigmas=sigmas,
        noiser=noiser,
        width=stage_width,
        height=stage_height,
        frames=layout["frames"],
        fps=layout["fps"],
        video=ModalitySpec(context=video_context, conditionings=video_conditionings),
        audio=ModalitySpec(context=audio_context, conditionings=audio_conditionings),
    )
    audio_latent = audio_state.latent.clone()
    del stage, _video_state, audio_state
    torch.cuda.empty_cache()

    # The only decode we need. No video decoder, no upsampler.
    audio_decoder = AudioDecoder(args.checkpoint_path, dtype, device)
    audio = audio_decoder(audio_latent)
    del audio_decoder
    torch.cuda.empty_cache()

    wav_path = scratch / "foley.wav"
    encode_audio(audio=audio, output_path=str(wav_path))
    return wav_path


def mux(args: argparse.Namespace, wav_path: Path, source: dict) -> None:
    """Copy the original picture through, whole, and attach the generated audio.

    `-shortest` used to be here, and `plan()` floors the frame count to the model's 8n+1 grid
    *downward*, so the generated audio is always shorter than or equal to the source. What
    `-shortest` did with that short stream turned out to depend on the clip, and both outcomes are
    wrong:

    - on a 30-frame synthetic source it cut the copied video stream, which came out with 27 or 28
      frames while the container's duration metadata still read the original 1.250000 -- which is
      why nothing ever reported it;
    - on a real 124-frame H3 output it left the picture alone and ended the soundtrack at 5.013 s
      against a 5.167 s clip, so the last sixth of a second was silent.

    Which one you get comes down to muxer interleaving, so it is not something a frame-count check
    on one file can rule out. Both are the same root cause: the picture and the soundtrack were
    being reconciled by discarding whichever ran out first.

    So: pad the audio instead (`apad`) and end the output at the source's own probed duration
    (`-t`). Measured on that same real clip: 124 frames in, 124 frames out at 5.166667 s, audio
    5.167 s. `apad` with `-shortest` keeps the frames too but still leaves the audio about a tenth
    of a second short, because `-shortest` ends on an encoder frame boundary.
    """
    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-i", args.video_path, "-i", str(wav_path),
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "320k",
            "-af", "apad", "-t", f"{source['duration']:.6f}",
            str(output),
        ],
        check=True,
    )


def main() -> int:
    force_utf8_io()
    args = build_parser().parse_args()
    validate(args)
    source = probe(args.video_path)
    layout = plan(source)
    if args.dry_run:
        print(json.dumps({"ok": True, "mode": "foley-v2a", "stageTwo": False, "layout": layout}))
        return 0

    import torch

    with tempfile.TemporaryDirectory(prefix="aiv-foley-") as scratch_dir:
        scratch = Path(scratch_dir)
        # Only needed to satisfy the audio-reference encoder. With the reference off there is
        # nothing to decode, so skip the re-mux entirely.
        reference_path = (
            ensure_audio_track(args.video_path, source, scratch)
            if args.audio_reference_strength > 0
            else args.video_path
        )
        with torch.inference_mode():
            wav_path = run_with_recovery(
                lambda: generate_audio(args, layout, reference_path, scratch),
                args,
                ltx_degradations(),
                label="ltx-foley",
            )
        mux(args, wav_path, source)
    print(json.dumps({"ok": True, "outputPath": args.output_path, "mode": "foley-v2a"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
