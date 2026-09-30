from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# The LTX adapters run under the vendored LTX virtualenv, where this package is not installed, and
# Python only puts the *script's own* directory on sys.path -- not the package root. Without this,
# `import aiv_runtime.…` fails with ModuleNotFoundError depending on which interpreter is used.
_PACKAGE_ROOT = str(Path(__file__).resolve().parent.parent)
if _PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, _PACKAGE_ROOT)


from aiv_runtime import ltx_nag  # noqa: E402
from aiv_runtime.ltx_attention import (  # noqa: E402
    ATTENTION_BACKENDS,
    DEFAULT_ATTENTION_BACKEND,
    describe_attention,
    install_attention,
)
from aiv_runtime.ltx_offload import (  # noqa: E402
    OFFLOAD_PROFILES,
    cpu_slots_for,
    describe_profile,
    offload_mode_for,
)
from aiv_runtime.safetensors_header import has_key_suffix  # noqa: E402
from aiv_runtime.safetensors_reader import (  # noqa: E402
    describe_backend,
    install_pread_state_dict_loader,
)
from aiv_runtime.ltx_geometry import (  # noqa: E402
    AXIS_MULTIPLE,
    DEFAULT_MAX_FRAMES,
    DEFAULT_MAX_PIXELS,
    validate_axes,
    validate_num_frames,
)


def _multiple(value: int, factor: int, name: str) -> int:
    if value <= 0 or value % factor:
        raise argparse.ArgumentTypeError(f"{name} must be a positive multiple of {factor}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LTX distilled single-pass first/last-frame video generation without native upscalers."
    )
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--gemma-root", required=True)
    # `first-last` is the original, hand-tuned single-pass path and stays the default so nothing
    # that already worked changes behaviour.
    parser.add_argument("--mode", choices=("first-last", "text", "image"), default="first-last")
    parser.add_argument("--first-frame", default=None)
    parser.add_argument("--last-frame", default=None)
    parser.add_argument("--negative-prompt", default="")
    # Classifier-free guidance. 1.0 is unguided, which is what the distilled schedule assumes;
    # anything above it runs a second unconditional pass per step, so it costs roughly double.
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--audio-cfg-scale", type=float, default=1.0)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--prompt", required=True)
    # 32, not 64. `assert_resolution` in the pinned tree computes `64 if is_two_stage else 32`,
    # and this is the one-stage path -- 832x480 used to be retakeable but not generatable purely
    # because of this line. See `ltx_geometry`.
    parser.add_argument("--width", type=lambda value: _multiple(int(value), AXIS_MULTIPLE, "width"), default=768)
    parser.add_argument("--height", type=lambda value: _multiple(int(value), AXIS_MULTIPLE, "height"), default=512)
    parser.add_argument("--num-frames", type=int, default=121)
    parser.add_argument("--frame-rate", type=float, default=24.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guide-strength", type=float, default=0.8)
    parser.add_argument("--offload", choices=OFFLOAD_PROFILES, default="cpu")
    # Sage is not the default here, unlike H3: on H3 the kernels were installed and
    # merely unselected, whereas nothing is installed in the LTX venv until
    # scripts\install-ltx-attention.ps1 has been run. Asking for a backend that is not
    # there costs a log line, not a run.
    parser.add_argument(
        "--attention-backend", choices=ATTENTION_BACKENDS, default=DEFAULT_ATTENTION_BACKEND
    )
    # Normalized Attention Guidance: a negative prompt that works on a guidance-distilled
    # checkpoint, where CFG cannot. Off by default -- it has not been run against a picture yet,
    # and the point of shipping it inert is that the CFG defaults above stay the only thing
    # changing until someone deliberately turns this on. See `ltx_nag`.
    parser.add_argument("--nag-scale", type=float, default=0.0)
    parser.add_argument("--nag-tau", type=float, default=ltx_nag.DEFAULT_TAU)
    parser.add_argument("--nag-alpha", type=float, default=ltx_nag.DEFAULT_ALPHA)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def nag_requested(args: argparse.Namespace) -> bool:
    """Whether this run asked for NAG and gave it something to steer away from."""
    return args.nag_scale > 0.0 and bool(args.negative_prompt.strip())


# Which conditioning inputs each mode actually requires.
MODE_REQUIREMENTS = {
    "first-last": ("first_frame", "last_frame"),
    "image": ("first_frame",),
    "text": (),
}


def _refuse_a_negative_prompt_that_cannot_be_used(args: argparse.Namespace) -> None:
    """Do not accept a negative prompt and then quietly drop it.

    CFG needs an unconditional branch, so `_run_inference` only encodes the negative prompt when
    a scale is above 1.0. Both scales default to 1.0 here, which is the distilled schedule's
    native setting -- so a negative prompt typed into the panel was accepted, never encoded, and
    never mentioned. The user waits out a multi-minute generation to find their instruction had no
    effect at all, with nothing in the log to say why.

    Refusing rather than warning follows `kiwi_edit_16gb.py`, which does the same thing for the
    same reason: it detects a requested guidance the pipeline cannot honour and raises with an
    actionable message. This runs in `validate()`, before a byte of the checkpoint is read.

    This whole check is temporary. NAG makes negative prompts work on a guidance-distilled
    checkpoint without a surviving `cond - uncond` gap; while NAG is opt-in, asking for it is an
    exemption, and when it stops being opt-in this function retires entirely.
    """
    if not args.negative_prompt.strip():
        return
    if args.cfg_scale > 1.0 or args.audio_cfg_scale > 1.0:
        return
    if nag_requested(args):
        # NAG carries the negative prompt without an unconditional branch, so there is nothing to
        # refuse. This is the exemption the plan describes; the check itself retires when NAG
        # stops being opt-in.
        return
    raise ValueError(
        "a negative prompt was given but both guidance scales are 1.0, so there is no "
        "unconditional branch to apply it to and it would be silently ignored. Raise "
        "--cfg-scale (or --audio-cfg-scale) above 1.0 to use it -- each costs a second pass per "
        "step -- or clear the negative prompt."
    )


def validate(args: argparse.Namespace) -> None:
    for value in (args.checkpoint_path, args.gemma_root):
        if not Path(value).exists():
            raise FileNotFoundError(value)
    for attribute in MODE_REQUIREMENTS[args.mode]:
        value = getattr(args, attribute)
        if not value:
            raise ValueError(f"mode {args.mode!r} requires --{attribute.replace('_', '-')}")
        if not Path(value).exists():
            raise FileNotFoundError(value)
    if not args.prompt.strip():
        raise ValueError("prompt cannot be empty")
    _refuse_a_negative_prompt_that_cannot_be_used(args)
    # Generation had no frame cap at all, which is how `--num-frames 481` reached a 27 GB load
    # before failing. The shared module carries the cap retake already had.
    validate_num_frames(args.num_frames, DEFAULT_MAX_FRAMES)
    if not 0 < args.guide_strength <= 1:
        raise ValueError("guide-strength must be in (0, 1]")
    for name, value in (("cfg-scale", args.cfg_scale), ("audio-cfg-scale", args.audio_cfg_scale)):
        if not 1.0 <= value <= 10.0:
            raise ValueError(f"{name} must be between 1.0 (unguided) and 10.0")
    validate_axes(args.width, args.height, DEFAULT_MAX_PIXELS)


def _scaled_fp8_streaming_stage(
    checkpoint_path, dtype, device, offload_mode, quantization, cpu_slots_count=None
):
    """Build scaled-FP8 streaming without changing the installed LTX packages.

    LTX 2.3's block streamer already handles mixed-dtype block layouts, but its
    high-level factory currently rejects policies with companion scale tensors.
    With no LoRAs there is no companion-key fusion, so streaming the checkpoint's
    existing weight/input scales is safe. Keep those scales in FP32 while block
    weights remain FP8.
    """
    import ltx_core.block_streaming.builder as block_builder_module
    from ltx_core.block_streaming import DISK_CPU_SLOTS, StreamingModelBuilder
    from ltx_core.block_streaming.utils import FP8_DTYPES
    from ltx_core.loader.registry import DummyRegistry
    from ltx_core.model.transformer import LTXV_MODEL_COMFY_RENAMING_MAP, LTXModelConfigurator
    from ltx_pipelines.utils.blocks import DiffusionStage
    from ltx_pipelines.utils.types import OffloadMode

    def scale_safe_layout(tensors, requested_dtype=None):
        return {
            name: (
                tensor.shape,
                tensor.dtype
                if requested_dtype is None
                or tensor.dtype in FP8_DTYPES
                or name.endswith((".weight_scale", ".input_scale"))
                else requested_dtype,
            )
            for name, tensor in tensors.items()
        }

    block_builder_module.derive_layout = scale_safe_layout
    builder = StreamingModelBuilder(
        model_class_configurator=LTXModelConfigurator,
        model_path=checkpoint_path,
        model_sd_ops=LTXV_MODEL_COMFY_RENAMING_MAP,
        loras=(),
        registry=DummyRegistry(),
        fuse_rule=quantization.fuse_rule,
        blocks_attr="transformer_blocks",
        blocks_prefix="transformer_blocks",
        # An explicit count is a named profile's choice; falling back to the two extremes keeps
        # any caller that has not been taught about profiles behaving exactly as before.
        cpu_slots_count=(
            cpu_slots_count
            if cpu_slots_count is not None
            else (DISK_CPU_SLOTS if offload_mode == OffloadMode.DISK else None)
        ),
    )
    return DiffusionStage(builder, dtype, device, quantization=quantization)


def _run_inference(args: argparse.Namespace) -> None:
    import torch
    from ltx_core.components.noisers import GaussianNoiser
    from ltx_core.model.transformer import X0Model
    from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
    from ltx_core.quantization.fp8_cast import build_policy as build_fp8_cast_policy
    from ltx_core.quantization.fp8_scaled_mm import build_policy as build_fp8_scaled_policy
    from ltx_pipelines.utils.args import ImageConditioningInput
    from ltx_pipelines.utils.blocks import AudioDecoder, DiffusionStage, ImageConditioner, PromptEncoder, VideoDecoder
    from ltx_pipelines.utils.constants import DISTILLED_SIGMAS
    from ltx_core.components.guiders import MultiModalGuider, MultiModalGuiderParams
    from ltx_pipelines.utils.denoisers import GuidedDenoiser, SimpleDenoiser
    from ltx_pipelines.utils.helpers import (
        cleanup_memory,
        get_device,
        image_conditionings_by_adding_guiding_latent,
    )
    from ltx_pipelines.utils.media_io import encode_video
    from ltx_pipelines.utils.types import ModalitySpec, OffloadMode

    device = get_device()
    dtype = torch.bfloat16
    offload_mode = offload_mode_for(args.offload)
    cpu_slots = cpu_slots_for(args.offload)
    print(f"[ltx] {describe_profile(args.offload)}", file=sys.stderr)
    prompt_encoder = PromptEncoder(
        args.checkpoint_path, args.gemma_root, dtype, device, offload_mode=offload_mode
    )
    image_conditioner = ImageConditioner(args.checkpoint_path, dtype, device)
    supports_fp8 = device.type == "cuda" and torch.cuda.get_device_capability(device) >= (8, 9)
    if not supports_fp8:
        raise RuntimeError("The 16 GB profile requires an RTX GPU with FP8 support (compute capability 8.9+)")
    # The header, not a mapping. `safe_open` here charged the whole 27.5 GiB checkpoint against
    # the Windows commit limit to answer one key-suffix question -- see `safetensors_header`.
    scaled_fp8 = has_key_suffix(args.checkpoint_path, ".weight_scale")
    install_pread_state_dict_loader()
    print(f"[ltx] checkpoint read: {describe_backend()}", file=sys.stderr)
    quantization = (
        build_fp8_scaled_policy(args.checkpoint_path)
        if scaled_fp8
        else build_fp8_cast_policy(args.checkpoint_path)
    )
    stage = (
        _scaled_fp8_streaming_stage(
            args.checkpoint_path, dtype, device, offload_mode, quantization, cpu_slots
        )
        if scaled_fp8
        else DiffusionStage.from_checkpoint(
            args.checkpoint_path,
            dtype,
            device,
            loras=(),
            quantization=quantization,
            offload_mode=offload_mode,
        )
    )
    video_decoder = VideoDecoder(args.checkpoint_path, dtype, device)
    audio_decoder = AudioDecoder(args.checkpoint_path, dtype, device)

    generator = torch.Generator(device=device).manual_seed(args.seed)
    noiser = GaussianNoiser(generator=generator)

    # CFG needs an unconditional branch, which means encoding the negative prompt too. Skip that
    # work entirely when both scales are 1.0 and NAG is off, which is the distilled model's native
    # setting. NAG needs the same encoding for a different reason -- it attends against it rather
    # than differencing whole predictions -- so either one asks for it.
    guided = args.cfg_scale > 1.0 or args.audio_cfg_scale > 1.0
    nag_wanted = nag_requested(args)
    needs_negative = guided or nag_wanted
    prompts = [args.prompt, args.negative_prompt] if needs_negative else [args.prompt]
    encoded = prompt_encoder(
        prompts, enhance_first_prompt=False, enhance_prompt_image=args.first_frame or None
    )
    video_context = encoded[0].video_encoding.cpu()
    audio_context = encoded[0].audio_encoding.cpu()
    negative_video_context = encoded[1].video_encoding.cpu() if needs_negative else None
    negative_audio_context = encoded[1].audio_encoding.cpu() if needs_negative else None
    del encoded
    torch.cuda.empty_cache()

    if nag_wanted:
        ltx_nag.activate(
            scale=args.nag_scale,
            negative_video_context=negative_video_context,
            negative_audio_context=negative_audio_context,
            tau=args.nag_tau,
            alpha=args.nag_alpha,
        )
    else:
        ltx_nag.deactivate()
    print(f"[ltx] {ltx_nag.describe()}", file=sys.stderr)
    # Conditioning depends on the mode. Text-to-video supplies none, image-to-video pins frame 0,
    # and first-last pins both ends -- the original behaviour, unchanged.
    if args.mode == "text":
        images = []
    elif args.mode == "image":
        images = [ImageConditioningInput(args.first_frame, 0, args.guide_strength)]
    else:
        images = [
            ImageConditioningInput(args.first_frame, 0, args.guide_strength),
            ImageConditioningInput(args.last_frame, args.num_frames - 1, args.guide_strength),
        ]
    conditionings = (
        image_conditioner(
            lambda encoder: image_conditionings_by_adding_guiding_latent(
                images=images,
                height=args.height,
                width=args.width,
                video_encoder=encoder,
                dtype=dtype,
                device=device,
            )
        )
        if images
        else []
    )
    torch.cuda.empty_cache()
    video_context = video_context.to(device)
    audio_context = audio_context.to(device)
    if guided:
        # Both guiders are built even when only one modality is above 1.0. That is deliberate:
        # LTX-2 is a single joint audio-video transformer, and `_guided_denoise` appends ONE
        # unconditional pass if *either* modality needs it, carrying both v_neg and a_neg. Setting
        # a guider to None does not remove the second pass -- it only makes that pass reuse the
        # positive context for the absent modality, which changes what video contributes to the
        # joint unconditional prediction and can feed back into audio through cross-attention.
        # A guider at cfg_scale 1.0 contributes a zero delta ((cfg-1) * (cond - uncond)), so
        # keeping it costs nothing in the result and keeps the negative contexts symmetric.
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
        print(
            f"[ltx] CFG video {args.cfg_scale:.2f} · audio {args.audio_cfg_scale:.2f} "
            "(two passes per step)",
            file=sys.stderr,
        )
    else:
        denoiser = SimpleDenoiser(video_context, audio_context)
    sigmas = DISTILLED_SIGMAS.to(dtype=torch.float32, device=device)
    builder = stage._prepared_builder()
    streaming_wrapper = builder.build(device=device, dtype=dtype, gpu_slots_count=1)
    try:
        transformer = X0Model(streaming_wrapper).eval()
        # AFTER the transformer exists, never before: see `install_attention`.
        active_attention, attention_swapped, attention_reason = install_attention(
            args.attention_backend,
            transformer,
            log=lambda message: print(f"[ltx] {message}", file=sys.stderr),
        )
        print(
            f"[ltx] {describe_attention(args.attention_backend, active_attention, attention_swapped, attention_reason)}",
            file=sys.stderr,
        )
        video_state, audio_state = stage.run(
            transformer=transformer,
            denoiser=denoiser,
            sigmas=sigmas,
            noiser=noiser,
            width=args.width,
            height=args.height,
            frames=args.num_frames,
            fps=args.frame_rate,
            video=ModalitySpec(context=video_context, conditionings=conditionings),
            audio=ModalitySpec(context=audio_context),
            max_batch_size=1,
        )
    finally:
        # `teardown()` releases the non-memory resources -- forward hooks, the disk I/O worker,
        # open file handles -- and nothing else. The memory reclaim is `to("meta")`, and without
        # it the non-block weights and the GPU buffer slot stayed resident through the whole
        # decode: the video decoder, the audio decoder and the vocoder all build and run after
        # this block, which is the moment of peak decode pressure.
        #
        # This is exactly what the pinned tree's own `_streaming_model` context manager does
        # (`blocks.py`, AllocatorTrimStrategy.TRIM), and what `ltx_foley_v2a.py` gets for free by
        # going through it. Our two adapters disagreed; they no longer do.
        streaming_wrapper.teardown()
        streaming_wrapper.to("meta")
        cleanup_memory()
        torch.cuda.empty_cache()
    if torch.cuda.is_available():
        print(
            f"[ltx] VRAM after releasing the denoiser: "
            f"{torch.cuda.memory_allocated() / 1024**2:,.0f} MiB allocated, peak so far "
            f"{torch.cuda.max_memory_allocated() / 1024**2:,.0f} MiB",
            file=sys.stderr,
        )
        torch.cuda.reset_peak_memory_stats()
    tiling = TilingConfig.default()
    video = video_decoder(video_state.latent, tiling, generator=generator)
    audio = audio_decoder(audio_state.latent)
    if torch.cuda.is_available():
        print(
            f"[ltx] VRAM peak across the decode: "
            f"{torch.cuda.max_memory_allocated() / 1024**2:,.0f} MiB",
            file=sys.stderr,
        )
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    encode_video(
        video=video,
        fps=args.frame_rate,
        audio=audio,
        output_path=args.output_path,
        video_chunks_number=get_video_chunks_number(args.num_frames, tiling),
    )


def run(args: argparse.Namespace) -> None:
    import torch

    from aiv_runtime.oom import ltx_degradations, run_with_recovery

    with torch.inference_mode():
        run_with_recovery(lambda: _run_inference(args), args, ltx_degradations(), label="ltx")


def main() -> int:
    args = build_parser().parse_args()
    validate(args)
    if args.dry_run:
        print(json.dumps({
            "ok": True,
            "mode": args.mode,
            "spatialUpscaler": False,
            "temporalUpscaler": False,
            "offload": args.offload,
        }))
        return 0
    run(args)
    print(json.dumps({"ok": True, "outputPath": args.output_path}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
