"""Scaled-FP8 block streaming with LoRA fusion.

LTX's block streamer refuses LoRAs on a scaled-FP8 checkpoint. `DiffusionStage.from_checkpoint`
rejects the policy outright, and even reaching `WeightsProvider` directly is not enough, because
its fusion step is written for rules that emit exactly one tensor per weight:

    fused = rule(name, tensor, deltas, _EMPTY_STATE_DICT)
    tensor.copy_(fused[name])

Two things break there for scaled FP8:

1. `_EMPTY_STATE_DICT` means `_fp8_scaled_mm_fuse` cannot find the layer's `.weight_scale`, so it
   falls through to `bf16_fuse_rule` and adds a BF16 delta directly onto an FP8 weight. That does
   not raise -- it quietly corrupts the weight.
2. `_fp8_scaled_mm_fuse` returns TWO keys, the re-quantized weight and a fresh `.weight_scale`.
   Only the weight is copied back, so the scale keeps describing the pre-fusion weight and every
   fused layer comes out mis-scaled.

Neither is a missing algorithm. `ltx_core.quantization.fp8_scaled_mm._fp8_scaled_mm_fuse` already
implements dequantize -> add delta -> re-quantize, which is the same thing ComfyUI does per-layer
through `convert_weight`/`revert_weight` (Comfy-Org/ComfyUI#10267). ComfyUI's official LTX-2.3
IC-LoRA workflows run on `ltx-2.3-22b-distilled-fp8.safetensors` for exactly this reason.

So this module supplies what the streaming path is missing rather than reimplementing the maths:
the block's own tensors as the state dict, and write-back for every key the rule returns. That
removes the need for the 46 GB BF16 checkpoint on the LoRA paths.

The patch is applied to `WeightsProvider` in-process. The vendored LTX tree is left untouched so
`uv sync` stays reproducible.
"""

from __future__ import annotations

import torch

from aiv_runtime.ltx_offload import CPU_SLOTS_BY_PROFILE


def scale_safe_layout(tensors, requested_dtype=None):
    """Layout that preserves FP8 block weights and FP32 companion scales.

    `derive_layout` coerces every non-FP8 tensor to the requested dtype, which would drag the
    FP32 `.weight_scale` / `.input_scale` companions down to BF16 and destroy the precision the
    scaled matmul depends on. Keep FP8 weights as FP8, scales at their own dtype, everything else
    as asked.
    """
    from ltx_core.block_streaming.utils import FP8_DTYPES

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


_PATCHED = False


def patch_streaming_lora_fusion() -> None:
    """Teach `WeightsProvider` to fuse LoRAs into scaled-FP8 blocks. Idempotent."""
    global _PATCHED
    if _PATCHED:
        return

    from ltx_core.block_streaming.provider import WeightsProvider
    from ltx_core.loader.fuse_loras import aggregate_lora_products, device_fuse_rule
    from ltx_core.loader.primitives import StateDict

    def _fuse_block_loras(self, idx: int, weights: dict[str, torch.Tensor]) -> None:
        rule = device_fuse_rule(self._target_device, self._fuse_rule)

        # The block's own tensors, so the rule can resolve `<layer>.weight_scale`. These are live
        # views into the GPU slot, which is what makes companion write-back below work in place.
        block_sd = StateDict(
            sd=dict(weights),
            device=self._target_device,
            size=0,
            dtype={tensor.dtype for tensor in weights.values()},
        )

        for name, tensor in weights.items():
            if not name.endswith(".weight"):
                continue
            prefix = f"{self._blocks_prefix}.{idx}.{name}".removesuffix(".weight")
            products = (
                ab
                for ab in (
                    source.get_ab(prefix, device=self._target_device, dtype=rule.aggregation_dtype)
                    for source in self._lora_sources
                )
                if ab is not None
            )
            deltas = aggregate_lora_products(products, rule.aggregation_dtype)
            if deltas is None:
                continue

            fused = rule(name, tensor, deltas, block_sd)

            # Write back EVERY returned key, not just the weight. For scaled FP8 the rule also
            # returns a fresh `.weight_scale`; dropping it leaves the scale describing the
            # pre-fusion weight. Keys absent from this block's layout are skipped rather than
            # raising, so a rule that emits companions we do not stream stays harmless.
            for key, value in fused.items():
                target = weights.get(key)
                if target is None:
                    continue
                # copy_ keeps the pooled buffer's storage and dtype; the rule already produced
                # each tensor in its destination dtype (FP8 weight, FP32 scale).
                target.copy_(value)

    WeightsProvider._fuse_block_loras = _fuse_block_loras
    _PATCHED = True


def build_streaming_stage(
    checkpoint_path: str,
    dtype: torch.dtype,
    device: torch.device,
    offload_mode,
    quantization,
    loras: tuple = (),
    cpu_slots_count: int | None = None,
):
    """A `DiffusionStage` that streams a scaled-FP8 checkpoint, optionally fusing `loras`.

    Bypasses `DiffusionStage.from_checkpoint`, whose `_build_streaming_builder` raises for any
    policy whose `fuse_rule` is not `fp8_cast_fuse_rule`. That guard is about companion-key
    write-back, which `patch_streaming_lora_fusion` supplies, so constructing the builder directly
    is sound here -- provided the caller passes a scaled-FP8 policy and this module's patch is
    active, both of which this function ensures.
    """
    import ltx_core.block_streaming.builder as block_builder_module
    from ltx_core.block_streaming import DISK_CPU_SLOTS, StreamingModelBuilder
    from ltx_core.loader.registry import DummyRegistry
    from ltx_core.model.transformer import LTXV_MODEL_COMFY_RENAMING_MAP, LTXModelConfigurator
    from ltx_pipelines.utils.blocks import DiffusionStage
    from ltx_pipelines.utils.types import OffloadMode

    assert DISK_CPU_SLOTS == CPU_SLOTS_BY_PROFILE["disk"], (
        f"the pinned tree now uses {DISK_CPU_SLOTS} disk slots; the profile table says "
        f"{CPU_SLOTS_BY_PROFILE['disk']}"
    )
    block_builder_module.derive_layout = scale_safe_layout
    if loras:
        patch_streaming_lora_fusion()

    builder = StreamingModelBuilder(
        model_class_configurator=LTXModelConfigurator,
        model_path=checkpoint_path,
        model_sd_ops=LTXV_MODEL_COMFY_RENAMING_MAP,
        loras=tuple(loras),
        registry=DummyRegistry(),
        fuse_rule=quantization.fuse_rule,
        blocks_attr="transformer_blocks",
        blocks_prefix="transformer_blocks",
        cpu_slots_count=(
            cpu_slots_count
            if cpu_slots_count is not None
            else (DISK_CPU_SLOTS if offload_mode == OffloadMode.DISK else None)
        ),
    )
    return DiffusionStage(builder, dtype, device, quantization=quantization)


def is_scaled_fp8(checkpoint_path: str) -> bool:
    """True when the checkpoint carries per-tensor `.weight_scale` companions.

    Read from the header. `safe_open` answered the same question by mapping 27.5 GiB
    copy-on-write, which Windows charges in full against the commit limit at map time.
    """
    from aiv_runtime.safetensors_header import has_key_suffix

    return has_key_suffix(checkpoint_path, ".weight_scale")
