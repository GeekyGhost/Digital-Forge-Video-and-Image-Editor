"""Build the reproducible Digital Forge MiniMax-H3 Ref2VA Turbo checkpoint.

The original DF converter packed low-bit values into a safetensors file, but the
runtime expanded every quantized projection to BF16 before calling F.linear. It
was disk compression, not low-bit inference.

The default profile is deliberately not a new post-training quantization scheme.
It preserves the source checkpoint's FP8 payload and row scales bit-for-bit, but
serializes them as ordinary safetensors tensors so Diffusers group offload never
calls ``Tensor.to`` on a TorchAO tensor subclass. Requantized GemLite profiles
remain behind explicit experimental flags because end-to-end qualification
showed blur or severe color corruption; DiT quantization research predicts those
failures without calibration, channel balancing, and sensitivity allocation.

The converter produces:

* default ``source-fp8``: preserves every source FP8 payload and row scale
  bit-for-bit as ordinary tensors, avoiding TorchAO tensor-subclass offloading;
* optional ``w8-a16``: every Q/K/V/output and MLP projection uses W8A16;
* optional ``w4w8-a16``: Q/output/MLP projections use W4A16 and K/V use
  W8A16 (structurally valid, but not visually qualified on H3);
* optional ``experimental-w8a8``: all transformer projections use W8A8;
* optional ``experimental-w4a8``: Q/output/MLP projections use W4A8 and K/V
  use W8A8 (structurally valid, but intentionally not quality-qualified);
* every other tensor, including the sampled AdaLN curve and normalization path:
  BF16;
* the complete LightX2V Turbo LoRA is embedded in the checkpoint. Quantized
  projections keep its low-rank branch in BF16 so requantization cannot erase
  the distillation signal; BF16 projections are fused exactly.

The output contains only ordinary tensors and safetensors metadata. At runtime,
the default loader reconstructs portable FP8 projections and dequantizes only
the active projection for BF16 compute. Experimental profiles reconstruct
GemLiteLinearTriton modules without repacking their weights.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Mapping

import torch
from safetensors.torch import load_file, save_file


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACK = ROOT / "models" / "MiniMax-H3"
DEFAULT_SOURCE = DEFAULT_PACK / "quantized" / "MiniMax-H3-Ref2VA-FP8.pt"
DEFAULT_LORA = DEFAULT_PACK / "loras" / "minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors"
DEFAULT_OUTPUTS = {
    "source-fp8": DEFAULT_PACK
    / "quantized"
    / "MiniMax-H3-Ref2VA-DF-Turbo-Portable-FP8.safetensors",
    "w8-a16": DEFAULT_PACK
    / "quantized"
    / "MiniMax-H3-Ref2VA-DF-Turbo-GemLite-W8A16.safetensors",
    "w4w8-a16": DEFAULT_PACK
    / "quantized"
    / "MiniMax-H3-Ref2VA-DF-Turbo-GemLite-W4W8A16.safetensors",
    "experimental-w8a8": DEFAULT_PACK
    / "quantized"
    / "MiniMax-H3-Ref2VA-DF-Turbo-GemLite-W8A8-EXPERIMENTAL.safetensors",
    "experimental-w4a8": DEFAULT_PACK
    / "quantized"
    / "MiniMax-H3-Ref2VA-DF-Turbo-GemLite-W4A8-EXPERIMENTAL.safetensors",
}

FORMATS = {
    "source-fp8": "df_portable_source_fp8_v1",
    "w8-a16": "df_gemlite_w8_a16_v1",
    "w4w8-a16": "df_gemlite_w4w8_a16_v1",
    "experimental-w8a8": "df_gemlite_w8a8_experimental_v1",
    "experimental-w4a8": "df_gemlite_w4a8_experimental_v1",
}
LORA_A_SUFFIX = ".lora_A.default.weight"
LORA_B_SUFFIX = ".lora_B.default.weight"
LORA_ALPHA = 8.0

TIER1_SUFFIXES = (
    ".ff.net.0.proj.weight",
    ".ff.net.2.weight",
    ".attn.to_q.weight",
    ".attn.to_out.0.weight",
)
TIER2_SUFFIXES = (
    ".attn.to_k.weight",
    ".attn.to_v.weight",
)
QUANTIZED_SUFFIXES = TIER1_SUFFIXES + TIER2_SUFFIXES
GEMLITE_FIELDS = ("W_q", "scales", "zeros", "metadata", "orig_shape")
GEMLITE_LORA_FIELDS = ("lora_A", "lora_B", "lora_scale")


def tier_for(name: str, tensor: torch.Tensor, profile: str = "source-fp8") -> str:
    """Return the exact mixed-precision tier for one source tensor."""
    if profile == "source-fp8":
        from torchao.quantization import Float8Tensor

        return "source-fp8" if isinstance(tensor, Float8Tensor) else "bf16"
    if "transformer_blocks." not in name or tensor.ndim != 2:
        return "bf16"
    if profile == "experimental-w8a8" and name.endswith(QUANTIZED_SUFFIXES):
        return "w8a8"
    if profile == "w8-a16" and name.endswith(QUANTIZED_SUFFIXES):
        return "w8a16"
    if profile == "w4w8-a16":
        if name.endswith(TIER1_SUFFIXES):
            return "w4a16"
        if name.endswith(TIER2_SUFFIXES):
            return "w8a16"
        return "bf16"
    if profile != "experimental-w4a8":
        return "bf16"
    if name.endswith(TIER1_SUFFIXES):
        return "w4a8"
    if name.endswith(TIER2_SUFFIXES):
        return "w8a8"
    return "bf16"


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_lora_pairs(path: Path) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    if not path.is_file():
        raise FileNotFoundError(f"Turbo LoRA not found: {path}")
    state = load_file(str(path), device="cpu")
    pairs: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    a_keys = [key for key in state if key.endswith(LORA_A_SUFFIX)]
    for a_key in a_keys:
        stem = a_key[: -len(LORA_A_SUFFIX)]
        b_key = stem + LORA_B_SUFFIX
        if b_key not in state:
            raise ValueError(f"LoRA A tensor has no matching B tensor: {a_key}")
        pairs[stem + ".weight"] = (state[a_key], state[b_key])

    b_count = sum(key.endswith(LORA_B_SUFFIX) for key in state)
    if not pairs or len(pairs) != b_count:
        raise ValueError(
            f"incomplete LoRA pairing: {len(pairs)} A/B pairs from {len(a_keys)} A and {b_count} B tensors"
        )
    return pairs


def dequantize_source_tensor(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Materialize one source tensor on CUDA without expanding the complete model."""
    from torchao.quantization import Float8Tensor

    if isinstance(tensor, Float8Tensor):
        return tensor.to(device).dequantize().to(torch.bfloat16)
    if tensor.is_floating_point():
        return tensor.to(device=device, dtype=torch.bfloat16)
    return tensor.to(device=device)


def fuse_lora(
    weight: torch.Tensor,
    pair: tuple[torch.Tensor, torch.Tensor],
    *,
    alpha: float,
) -> torch.Tensor:
    """Fuse one PEFT pair in FP32, then return a BF16 continuous weight."""
    lora_a, lora_b = pair
    rank = int(lora_a.shape[0])
    if rank < 1 or lora_b.shape[1] != rank:
        raise ValueError(f"invalid LoRA pair shapes: A={tuple(lora_a.shape)}, B={tuple(lora_b.shape)}")
    if (lora_b.shape[0], lora_a.shape[1]) != tuple(weight.shape):
        raise ValueError(
            f"LoRA delta shape {(lora_b.shape[0], lora_a.shape[1])} does not match weight {tuple(weight.shape)}"
        )
    device = weight.device
    delta = torch.mm(
        lora_b.to(device=device, dtype=torch.float32),
        lora_a.to(device=device, dtype=torch.float32),
    )
    fused = weight.float().add_(delta, alpha=alpha / rank).to(torch.bfloat16)
    del delta
    return fused


def _refresh_gemlite_metadata(module) -> None:
    module.metadata.data = torch.tensor(
        module.get_meta_args(), device=module.W_q.device, dtype=torch.int32
    )


def pack_gemlite_w4a8(weight: torch.Tensor) -> dict[str, torch.Tensor]:
    """Pre-pack a matrix for GemLite's INT8-activation/INT4-weight kernel."""
    from gemlite import DType, GemLiteLinearTriton

    out_features, in_features = weight.shape
    group_size = 128
    if in_features % group_size:
        raise ValueError(f"W4A16 input width {in_features} is not divisible by {group_size}")
    values = weight.float().view(out_features, in_features // group_size, group_size)
    scales = (values.abs().amax(dim=2) / 7.0).clamp_min(1e-5).to(torch.bfloat16)
    qweight = torch.clamp(
        torch.round(values / scales.float().unsqueeze(2)), -8, 7
    ).add_(8).to(torch.uint8).view(out_features, in_features)
    del values

    module = GemLiteLinearTriton(
        4,
        group_size=group_size,
        in_features=in_features,
        out_features=out_features,
        input_dtype=DType.INT8,
        output_dtype=DType.BF16,
        scaled_activations=True,
    ).to(weight.device)
    module.pack(qweight, scales, torch.full_like(scales, 8), fma_mode=False)
    module.W_group_mode = 1
    module.channel_scale_mode = 3
    _refresh_gemlite_metadata(module)
    del qweight, scales
    return {
        key: value.detach().to("cpu").contiguous()
        for key, value in module.state_dict().items()
    }


def pack_gemlite_w8a8(weight: torch.Tensor) -> dict[str, torch.Tensor]:
    """Pre-pack a matrix for GemLite's INT8-by-INT8 kernel."""
    from gemlite import DType, GemLiteLinearTriton

    out_features, in_features = weight.shape
    values = weight.float()
    scales = (values.abs().amax(dim=1, keepdim=True) / 127.0).clamp_min(1e-5).to(torch.bfloat16)
    qweight = torch.clamp(torch.round(values / scales.float()), -128, 127).to(torch.int8)
    del values

    module = GemLiteLinearTriton(
        8,
        group_size=in_features,
        in_features=in_features,
        out_features=out_features,
        input_dtype=DType.INT8,
        output_dtype=DType.BF16,
        scaled_activations=True,
    ).to(weight.device)
    module.pack(qweight, scales, 0, fma_mode=False)
    module.W_group_mode = 1
    module.channel_scale_mode = 3
    _refresh_gemlite_metadata(module)
    del qweight, scales
    return {
        key: value.detach().to("cpu").contiguous()
        for key, value in module.state_dict().items()
    }


def pack_gemlite_w4a16(weight: torch.Tensor) -> dict[str, torch.Tensor]:
    """Pre-pack symmetric INT4 weights for BF16-activation inference."""
    from gemlite import DType, GemLiteLinearTriton

    out_features, in_features = weight.shape
    values = weight.float()
    scales = (values.abs().amax(dim=1, keepdim=True) / 7.0).clamp_min(1e-5).to(torch.bfloat16)
    qweight = torch.clamp(torch.round(values / scales.float()), -8, 7).add_(8).to(torch.uint8)
    del values
    module = GemLiteLinearTriton(
        4,
        group_size=in_features,
        in_features=in_features,
        out_features=out_features,
        input_dtype=DType.BF16,
        output_dtype=DType.BF16,
        scaled_activations=False,
    ).to(weight.device)
    module.pack(qweight, scales, 8, fma_mode=False)
    _refresh_gemlite_metadata(module)
    del qweight, scales
    return {key: value.detach().to("cpu").contiguous() for key, value in module.state_dict().items()}


def pack_gemlite_w8a16(weight: torch.Tensor) -> dict[str, torch.Tensor]:
    """Pre-pack symmetric INT8 weights for BF16-activation inference."""
    from gemlite import DType, GemLiteLinearTriton

    out_features, in_features = weight.shape
    values = weight.float()
    scales = (values.abs().amax(dim=1, keepdim=True) / 127.0).clamp_min(1e-5).to(torch.bfloat16)
    qweight = torch.clamp(torch.round(values / scales.float()), -128, 127).to(torch.int8)
    del values
    module = GemLiteLinearTriton(
        8,
        group_size=in_features,
        in_features=in_features,
        out_features=out_features,
        input_dtype=DType.BF16,
        output_dtype=DType.BF16,
        scaled_activations=False,
    ).to(weight.device)
    module.pack(qweight, scales, None, fma_mode=False)
    _refresh_gemlite_metadata(module)
    del qweight, scales
    return {key: value.detach().to("cpu").contiguous() for key, value in module.state_dict().items()}


def add_packed_state(
    output: dict[str, torch.Tensor],
    weight_name: str,
    state: Mapping[str, torch.Tensor],
) -> None:
    if not weight_name.endswith(".weight"):
        raise ValueError(f"packed projection is not a weight key: {weight_name}")
    prefix = weight_name[: -len(".weight")]
    missing = set(GEMLITE_FIELDS) - set(state)
    if missing:
        raise ValueError(f"GemLite state for {weight_name} is missing {sorted(missing)}")
    for field in GEMLITE_FIELDS:
        output[f"{prefix}.gemlite.{field}"] = state[field]


def add_embedded_lora_state(
    output: dict[str, torch.Tensor],
    weight_name: str,
    pair: tuple[torch.Tensor, torch.Tensor],
    *,
    alpha: float,
    namespace: str = "gemlite",
) -> None:
    """Embed an exact BF16 PEFT residual beside one packed base projection."""
    prefix = weight_name[: -len(".weight")]
    lora_a, lora_b = pair
    rank = int(lora_a.shape[0])
    output[f"{prefix}.{namespace}.lora_A"] = lora_a.to(torch.bfloat16).contiguous()
    output[f"{prefix}.{namespace}.lora_B"] = lora_b.to(torch.bfloat16).contiguous()
    output[f"{prefix}.{namespace}.lora_scale"] = torch.tensor(alpha / rank, dtype=torch.float32)


def add_portable_fp8_state(
    output: dict[str, torch.Tensor], weight_name: str, source_tensor: torch.Tensor
) -> None:
    """Serialize a TorchAO FP8 tensor's exact payload using ordinary tensors."""
    prefix = weight_name[: -len(".weight")]
    output[f"{prefix}.aiv_fp8.qdata"] = source_tensor.qdata.contiguous()
    output[f"{prefix}.aiv_fp8.scale"] = source_tensor.scale.contiguous()
    output[f"{prefix}.aiv_fp8.orig_shape"] = torch.tensor(
        tuple(source_tensor.shape), dtype=torch.int32
    )


def _validate_environment(profile: str) -> torch.device:
    if profile == "source-fp8":
        print(
            "Portable source-FP8 conversion: CPU serialization; GemLite and CUDA are not required",
            flush=True,
        )
        return torch.device("cpu")
    try:
        import gemlite
    except ImportError as error:
        raise RuntimeError(
            "GemLite is required. Repair the H3 runtime with scripts/setup-h3.ps1."
        ) from error
    if not torch.cuda.is_available():
        raise RuntimeError("the genuine W4A8 conversion requires a CUDA GPU")
    capability = torch.cuda.get_device_capability()
    if capability < (8, 0):
        raise RuntimeError(
            f"GemLite W4A8 requires SM80 or newer, found SM{capability[0]}{capability[1]}"
        )
    print(
        f"CUDA target: {torch.cuda.get_device_name()} · SM{capability[0]}{capability[1]} · "
        f"GemLite {gemlite.__version__}",
        flush=True,
    )
    return torch.device("cuda")


def convert(
    source: Path,
    lora: Path,
    output: Path,
    *,
    profile: str,
    overwrite: bool,
    hash_sources: bool,
) -> dict:
    if not source.is_file():
        raise FileNotFoundError(f"source checkpoint not found: {source}")
    if output.exists() and not overwrite:
        raise FileExistsError(
            f"output already exists (pass --overwrite to replace it): {output}"
        )

    device = _validate_environment(profile)
    started = time.perf_counter()
    print(f"Loading source checkpoint with mmap: {source}", flush=True)
    document = torch.load(source, map_location="cpu", weights_only=False, mmap=True)
    source_state = document.get("state_dict") if isinstance(document, Mapping) else None
    if not isinstance(source_state, Mapping):
        raise TypeError("source checkpoint does not contain a state_dict mapping")
    source_format = str(document.get("format", "unsloth_prequant_transformer_state_dict_v1"))
    source_metadata = dict(document.get("metadata", {}))
    del document

    print(f"Loading and validating Turbo LoRA: {lora}", flush=True)
    lora_pairs = load_lora_pairs(lora)
    missing_from_source = sorted(set(lora_pairs) - set(source_state))
    if missing_from_source:
        raise ValueError(f"LoRA targets are absent from source: {missing_from_source[:3]}")

    output_state: dict[str, torch.Tensor] = {}
    handled_lora: set[str] = set()
    embedded_lora = 0
    bf16_fused_lora = 0
    counts = {
        "source_fp8": 0,
        "w4a16": 0,
        "w8a16": 0,
        "w4a8": 0,
        "w8a8": 0,
        "bf16": 0,
    }
    total = len(source_state)

    for index, (name, source_tensor) in enumerate(source_state.items(), start=1):
        tier = tier_for(name, source_tensor, profile)
        # Portable conversion copies qdata/scale directly and therefore never
        # materializes its FP8 base weights. Experimental profiles still
        # requantize one projection at a time on CUDA.
        weight = None if tier == "source-fp8" else dequantize_source_tensor(source_tensor, device)
        pair = lora_pairs.get(name)

        if tier == "source-fp8":
            add_portable_fp8_state(output_state, name, source_tensor)
            if pair is not None:
                add_embedded_lora_state(
                    output_state, name, pair, alpha=LORA_ALPHA, namespace="aiv_fp8"
                )
                handled_lora.add(name)
                embedded_lora += 1
            counts["source_fp8"] += 1
        elif tier == "w4a8":
            packed = pack_gemlite_w4a8(weight)
            add_packed_state(output_state, name, packed)
            if pair is not None:
                add_embedded_lora_state(output_state, name, pair, alpha=LORA_ALPHA)
                handled_lora.add(name)
                embedded_lora += 1
            counts["w4a8"] += 1
            del packed
        elif tier == "w8a8":
            packed = pack_gemlite_w8a8(weight)
            add_packed_state(output_state, name, packed)
            if pair is not None:
                add_embedded_lora_state(output_state, name, pair, alpha=LORA_ALPHA)
                handled_lora.add(name)
                embedded_lora += 1
            counts["w8a8"] += 1
            del packed
        elif tier == "w4a16":
            packed = pack_gemlite_w4a16(weight)
            add_packed_state(output_state, name, packed)
            if pair is not None:
                add_embedded_lora_state(output_state, name, pair, alpha=LORA_ALPHA)
                handled_lora.add(name)
                embedded_lora += 1
            counts["w4a16"] += 1
            del packed
        elif tier == "w8a16":
            packed = pack_gemlite_w8a16(weight)
            add_packed_state(output_state, name, packed)
            if pair is not None:
                add_embedded_lora_state(output_state, name, pair, alpha=LORA_ALPHA)
                handled_lora.add(name)
                embedded_lora += 1
            counts["w8a16"] += 1
            del packed
        else:
            if pair is not None:
                weight = fuse_lora(weight, pair, alpha=LORA_ALPHA)
                handled_lora.add(name)
                bf16_fused_lora += 1
            output_state[name] = (
                weight.to(dtype=torch.bfloat16).to("cpu").contiguous()
                if weight.is_floating_point()
                else weight.to("cpu").contiguous()
            )
            counts["bf16"] += 1

        if weight is not None:
            del weight
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if index % 25 == 0 or index == total:
            print(
                f"[{index:>3}/{total}] FP8={counts['source_fp8']} · "
                f"W4A16={counts['w4a16']} · W8A16={counts['w8a16']} · "
                f"W4A8={counts['w4a8']} · W8A8={counts['w8a8']} · "
                f"BF16={counts['bf16']} · LoRA={len(handled_lora)} · "
                f"CUDA={torch.cuda.memory_allocated() / 1024**3:.2f} GiB",
                flush=True,
            )

    unfused = sorted(set(lora_pairs) - handled_lora)
    if unfused:
        raise RuntimeError(
            f"conversion left {len(unfused)} LoRA pairs unfused: {unfused[:3]}"
        )
    if not any(counts[key] for key in ("source_fp8", "w4a16", "w8a16", "w4a8", "w8a8")):
        raise RuntimeError(f"invalid precision mapping: {counts}")
    if profile == "experimental-w4a8" and counts["w4a8"] == 0:
        raise RuntimeError(f"experimental W4A8 profile did not select W4 layers: {counts}")

    source_hash = sha256_file(source) if hash_sources else "not-computed"
    lora_hash = sha256_file(lora) if hash_sources else "not-computed"
    metadata = {
        "format": FORMATS[profile],
        "architecture": "MiniMax-H3-Ref2VA",
        "profile": profile,
        "quality_status": (
            "runtime-qualified-visual-pending"
            if profile == "source-fp8"
            else "experimental-unqualified"
        ),
        "quantization": (
            "exact source row-wise FP8 payloads + BF16 compute + BF16 residuals"
            if profile == "source-fp8"
            else (
                "GemLite per-channel symmetric W8A16 + BF16"
                if profile == "w8-a16"
                else (
                    "GemLite group-128 symmetric W4A16 + per-channel W8A16 + BF16"
                    if profile == "w4w8-a16"
                    else (
                        "GemLite per-channel symmetric W8A8 + BF16"
                        if profile == "experimental-w8a8"
                        else "GemLite per-channel symmetric W4A8 + W8A8 + BF16"
                    )
                )
            )
        ),
        "activation_quantization": (
            "none (BF16 activations)"
            if profile in ("source-fp8", "w8-a16", "w4w8-a16")
            else "dynamic per-token INT8"
        ),
        "tier1": (
            "exact source FP8 payload and row scale for every source Float8Tensor"
            if profile == "source-fp8"
            else (
                "unused; every transformer projection is W8A16"
                if profile == "w8-a16"
                else (
                    "INT4 weights / BF16 activations: transformer Q, output, and MLP projections"
                    if profile == "w4w8-a16"
                    else (
                        "unused"
                        if profile == "experimental-w8a8"
                        else "INT4 weights / INT8 activations: transformer Q, output, and MLP projections"
                    )
                )
            )
        ),
        "tier2": (
            "BF16 matmul; only the active FP8 weight is dequantized"
            if profile == "source-fp8"
            else (
                "INT8 weights / BF16 activations: all transformer projections"
                if profile == "w8-a16"
                else (
                    "INT8 weights / BF16 activations: transformer K/V projections"
                    if profile == "w4w8-a16"
                    else (
                        "INT8 weights / INT8 activations: all transformer projections"
                        if profile == "experimental-w8a8"
                        else "INT8 weights / INT8 activations: transformer K/V projections"
                    )
                )
            )
        ),
        "tier3": "BF16: all remaining tensors",
        "tier1_w4a8_layers": str(counts["w4a8"]),
        "tier2_w8a8_layers": str(counts["w8a8"]),
        "w4a16_layers": str(counts["w4a16"]),
        "w8a16_layers": str(counts["w8a16"]),
        "portable_source_fp8_tensors": str(counts["source_fp8"]),
        "tier3_bf16_tensors": str(counts["bf16"]),
        "is_turbo_baked": "True",
        "turbo_lora_pairs_handled": str(len(handled_lora)),
        "turbo_lora_pairs_embedded_bf16": str(embedded_lora),
        "turbo_lora_pairs_fused_into_bf16": str(bf16_fused_lora),
        "turbo_strategy": (
            "embedded BF16 residual beside every portable source-FP8 projection"
            if profile == "source-fp8"
            else "embedded BF16 residual for packed layers; exact fusion for BF16 layers"
        ),
        "turbo_alpha": str(LORA_ALPHA),
        "turbo_recommended_steps": "4",
        "turbo_video_shift": "12.0",
        "turbo_audio_shift": "3.0",
        "source_checkpoint": source.name,
        "source_sha256": source_hash,
        "lora_checkpoint": lora.name,
        "lora_sha256": lora_hash,
        "source_format": source_format,
        "family": str(source_metadata.get("family", "minimax-h3")),
        "adaln_form": str(source_metadata.get("adaln_form", "curve")),
        "curve_dim": str(source_metadata.get("curve_dim", 8)),
        "curve_grid": str(source_metadata.get("curve_grid", 1025)),
        "adaln_out_dtype": str(source_metadata.get("adaln_out_dtype", "bfloat16")),
        "gemlite_version": (
            "not-required" if profile == "source-fp8" else __import__("gemlite").__version__
        ),
        "torch_version": torch.__version__,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".partial")
    if temporary.exists():
        temporary.unlink()
    print(f"Writing {len(output_state)} tensors atomically to {output}", flush=True)
    save_file(output_state, str(temporary), metadata=metadata)
    del output_state, source_state, lora_pairs
    gc.collect()

    os.replace(temporary, output)
    output_hash = sha256_file(output) if hash_sources else "not-computed"
    report = {
        "ok": True,
        "output": str(output),
        "bytes": output.stat().st_size,
        "gib": output.stat().st_size / 1024**3,
        "sha256": output_hash,
        "counts": counts,
        "handled_lora_pairs": len(handled_lora),
        "embedded_lora_pairs": embedded_lora,
        "bf16_fused_lora_pairs": bf16_fused_lora,
        "elapsed_seconds": time.perf_counter() - started,
        "metadata": metadata,
    }
    report_path = output.with_suffix(output.suffix + ".conversion.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    result.add_argument("--turbo-lora", type=Path, default=DEFAULT_LORA)
    result.add_argument(
        "--profile",
        choices=tuple(DEFAULT_OUTPUTS),
        default="source-fp8",
        help="source-fp8 preserves source numerics and avoids TorchAO offload; requantized profiles are experimental.",
    )
    result.add_argument(
        "--output",
        type=Path,
        help="Output path. Defaults to the canonical filename for the selected profile.",
    )
    result.add_argument("--overwrite", action="store_true")
    result.add_argument(
        "--no-hash",
        action="store_true",
        help="Skip SHA-256 provenance hashes.",
    )
    return result


def main() -> int:
    args = parser().parse_args()
    output = (args.output or DEFAULT_OUTPUTS[args.profile]).resolve()
    try:
        report = convert(
            args.source.resolve(),
            args.turbo_lora.resolve(),
            output,
            profile=args.profile,
            overwrite=args.overwrite,
            hash_sources=not args.no_hash,
        )
    except Exception as error:
        print(f"conversion failed: {error}", file=sys.stderr, flush=True)
        raise
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
