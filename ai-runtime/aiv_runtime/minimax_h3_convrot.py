"""Direct loader for Comfy-Org's published MiniMax-H3 INT8 ConvRot checkpoints.

This module deliberately does not import ComfyUI.  It maps the released single-file checkpoint
into Diffusers' MiniMaxH3Transformer3DModel and calls Comfy-Org's small ``comfy-kitchen`` kernel
package only for the quantized linear projections.  All parameters remain ordinary tensors, so
Diffusers group offloading can move them without TorchAO tensor-subclass storage aliasing.
"""

from __future__ import annotations

import json
import types
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import init_empty_weights
from aiv_runtime.safetensors_reader import open_tensors

# `MiniMaxH3Transformer3DModel` is imported inside `load_transformer`, not here, and must stay
# there. The regression suite runs under the project's own `.venv` (diffusers 0.39, which predates
# the H3 classes), while this module only ever *runs* under `vendor/MiniMax-H3/.venv` (0.40).
# Hoisting the import makes the whole module unimportable in the test environment, which takes the
# pure mapping logic -- `native_to_diffusers` and `_mapped_tensor`, the part most worth testing --
# down with it. `minimax_h3.py` defers every diffusers import for the same reason. The annotations
# below are safe because of `from __future__ import annotations`.


LORA_A_SUFFIX = ".lora_A.default.weight"
LORA_B_SUFFIX = ".lora_B.default.weight"


def native_to_diffusers(native_key: str, native_shape: tuple[int, ...]) -> list[dict]:
    """Map one released native H3 tensor to the Diffusers component tree."""
    if native_key == "rope.inv_freq":
        return []
    # The four fp32 compute islands upstream pins -- `video_patch_proj`, `audio_patch_proj`,
    # `final_layer.video_out` and `final_layer.audio_out` -- carry no `cast_dtype` here because
    # they do not need one. Both production ConvRot blobs store all four as F32, and the loader
    # runs with `assign=True`, so they land in fp32 from the file. Measured from the safetensors
    # headers on 2026-09-08 (header read only, no mapping). `final_layer.adaln_proj` is the
    # exception -- F16 in the blob against fp32 parameters in the native model -- which is why it
    # is the one entry below that casts.
    simple_prefixes = {
        "video_patch_proj.": "proj_in.",
        "audio_patch_proj.": "audio_proj_in.",
        "condition_proj.": "context_embedder.",
        "final_layer.norm.": "norm_out.norm.",
        "final_layer.adaln_proj.": "norm_out.",
        "final_layer.video_out.": "proj_out.",
        "final_layer.audio_out.": "audio_proj_out.",
    }
    if native_key == "adaln_t_table":
        return [{"target": "time_embedder.table", "rows": None}]
    for source, target in simple_prefixes.items():
        if native_key.startswith(source):
            mapping = {"target": target + native_key[len(source) :], "rows": None}
            # Curve-form checkpoints store the tiny basis coefficients in FP16, but
            # the native model constructs every curve AdaLN projection as FP32 and
            # load_state_dict copies those values into the FP32 parameters. Mirror
            # that compute island rather than preserving the storage dtype via
            # Diffusers' assign=True loader.
            if source == "final_layer.adaln_proj.":
                mapping["cast_dtype"] = "float32"
            return [mapping]

    if native_key.startswith("token_refiner.blocks."):
        mapped = "token_refiner.refiner_blocks." + native_key[len("token_refiner.blocks.") :]
    elif native_key.startswith("blocks."):
        mapped = "transformer_blocks." + native_key[len("blocks.") :]
    elif native_key == "token_refiner.final_norm.weight":
        return [{"target": native_key, "rows": None}]
    else:
        raise ValueError(f"unmapped native MiniMax-H3 tensor: {native_key}")

    if mapped.endswith(".attn.qkv_proj.weight"):
        if len(native_shape) != 2 or native_shape[0] % 3:
            raise ValueError(f"invalid fused QKV shape for {native_key}: {native_shape}")
        rows = native_shape[0] // 3
        prefix = mapped[: -len("qkv_proj.weight")]
        return [
            {"target": f"{prefix}to_{role}.weight", "rows": [index * rows, (index + 1) * rows]}
            for index, role in enumerate(("q", "k", "v"))
        ]

    # Native H3's fused SwiGLU projection stores [gate; value]. Diffusers'
    # SwiGLU module reads [value; gate], so this is a row permutation rather
    # than a key-only conversion. Apply the same permutation to ConvRot's
    # per-output-row scales below; otherwise the INT8 codes and scales no
    # longer describe the same rows.
    transform = "swap_halves" if mapped.endswith(".mlp.fc1.weight") else None
    cast_dtype = "float32" if ".adaln_proj.linear." in mapped else None

    mapped = (
        mapped.replace(".attn.q_norm.", ".attn.norm_q.")
        .replace(".attn.k_norm.", ".attn.norm_k.")
        .replace(".attn.out_proj.", ".attn.to_out.0.")
        .replace(".mlp.fc1.", ".ff.net.0.proj.")
        .replace(".mlp.fc2.", ".ff.net.2.")
    )
    return [
        {
            "target": mapped,
            "rows": None,
            "transform": transform,
            "cast_dtype": cast_dtype,
        }
    ]


def _mapped_tensor(tensor: torch.Tensor, mapping: dict) -> torch.Tensor:
    """Apply one source-to-Diffusers row transform without changing values."""
    rows = mapping.get("rows")
    if rows is not None:
        tensor = tensor[rows[0] : rows[1]]
    transform = mapping.get("transform")
    if transform is None:
        tensor = tensor.contiguous()
    elif transform == "swap_halves":
        if tensor.shape[0] % 2:
            raise ValueError(f"cannot swap uneven fused SwiGLU rows: {tuple(tensor.shape)}")
        gate, value = tensor.chunk(2, dim=0)
        tensor = torch.cat((value, gate), dim=0).contiguous()
    else:
        raise ValueError(f"unsupported MiniMax-H3 tensor transform: {transform}")
    cast_dtype = mapping.get("cast_dtype")
    if cast_dtype is not None:
        dtype = getattr(torch, cast_dtype, None)
        if not isinstance(dtype, torch.dtype):
            raise ValueError(f"unsupported MiniMax-H3 mapping dtype: {cast_dtype}")
        tensor = tensor.to(dtype=dtype)
    return tensor


class ConvRotLinear(nn.Module):
    """One published INT8 ConvRot projection backed by ordinary Parameters."""

    def __init__(
        self,
        qdata: torch.Tensor,
        scale: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if qdata.ndim != 2 or qdata.dtype != torch.int8:
            raise ValueError(f"ConvRot weight must be INT8 [out,in], got {qdata.shape} {qdata.dtype}")
        if tuple(scale.shape) != (qdata.shape[0], 1) or scale.dtype != torch.float32:
            raise ValueError(f"ConvRot scale must be FP32 [out,1], got {scale.shape} {scale.dtype}")
        self.in_features = int(qdata.shape[1])
        self.out_features = int(qdata.shape[0])
        self.qdata = nn.Parameter(qdata, requires_grad=False)
        self.scale = nn.Parameter(scale, requires_grad=False)
        self.bias = nn.Parameter(bias, requires_grad=False) if bias is not None else None
        self.register_buffer("_dtype_anchor", torch.empty(0, dtype=torch.bfloat16), persistent=False)

    @property
    def weight(self) -> torch.Tensor:
        # Diffusers inspects ``weight.dtype`` while preparing mixed video/audio inputs.
        return self._dtype_anchor

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        import comfy_kitchen

        return comfy_kitchen.int8_linear(
            inputs.contiguous(),
            self.qdata,
            self.scale,
            self.bias,
            out_dtype=inputs.dtype,
            convrot=True,
            convrot_groupsize=256,
        )


class TurboLinear(nn.Module):
    """Released base projection plus an unfused LightX2V Turbo residual."""

    def __init__(
        self,
        base: nn.Module,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
        scale: float,
    ) -> None:
        super().__init__()
        in_features = int(getattr(base, "in_features"))
        out_features = int(getattr(base, "out_features"))
        if tuple(lora_a.shape)[1] != in_features:
            raise ValueError("Turbo LoRA A input width does not match its H3 projection")
        if tuple(lora_b.shape) != (out_features, lora_a.shape[0]):
            raise ValueError("Turbo LoRA B shape does not match its H3 projection")
        self.base = base
        self.in_features = in_features
        self.out_features = out_features
        self.lora_A = nn.Parameter(lora_a.to(torch.bfloat16), requires_grad=False)
        self.lora_B = nn.Parameter(lora_b.to(torch.bfloat16), requires_grad=False)
        self.lora_scale = float(scale)

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        result = self.base(inputs)
        residual = F.linear(F.linear(inputs.to(self.lora_A.dtype), self.lora_A), self.lora_B)
        return result.add(residual.to(result.dtype), alpha=self.lora_scale)


def _replace_module(model: nn.Module, name: str, replacement: nn.Module) -> None:
    parent_name, _, child_name = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    setattr(parent, child_name, replacement)


def _install_curve_adaln(
    transformer: MiniMaxH3Transformer3DModel,
    table: torch.Tensor,
) -> None:
    """Restore the release checkpoint's sampled AdaLN curve architecture."""
    curve_grid, curve_dim = (int(value) for value in table.shape)

    class CurveTimestepEmbedding(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer(
                "table", torch.empty((curve_grid, curve_dim), device="meta", dtype=table.dtype)
            )

        @property
        def linear_1(self):
            return self

        @property
        def weight(self):
            return self.table

        def forward(self, timestep):
            values = timestep.reshape(-1).to(dtype=torch.float32)
            position = values.clamp(0.0, 1.0) * (self.table.shape[0] - 1)
            lower = position.floor().long().clamp(max=self.table.shape[0] - 2)
            fraction = (position - lower).unsqueeze(1).to(dtype=self.table.dtype)
            return torch.lerp(self.table[lower], self.table[lower + 1], fraction)

    def block_forward(module, temb):
        projected = module.linear(temb.to(module.linear.weight.dtype)).to(torch.bfloat16)
        return projected.view(-1, 6 * module.hidden_size).chunk(6, dim=-1)

    def output_forward(module, hidden_states, temb, timestep_indices):
        # Native curve AdaLN leaves the final modulation in FP32. That promotes
        # the normalized block output before the checkpoint's FP32 output heads.
        shift, scale = module.linear(temb.to(module.linear.weight.dtype)).chunk(2, dim=-1)
        hidden_states = module.norm(hidden_states)
        return hidden_states * (1.0 + scale.index_select(0, timestep_indices)) + shift.index_select(
            0, timestep_indices
        )

    transformer.time_proj = nn.Identity()
    transformer.time_embedder = CurveTimestepEmbedding()
    for block in transformer.transformer_blocks:
        output_size = 6 * block.adaln_proj.hidden_size * 3
        block.adaln_proj.linear = nn.Linear(
            curve_dim, output_size, bias=True, device="meta", dtype=torch.float32
        )
        block.adaln_proj.forward = types.MethodType(block_forward, block.adaln_proj)
    final_size = 2 * transformer.norm_out.norm.normalized_shape[0]
    transformer.norm_out.linear = nn.Linear(
        curve_dim, final_size, bias=True, device="meta", dtype=torch.float32
    )
    transformer.norm_out.forward = types.MethodType(output_forward, transformer.norm_out)


def _read_lora(path: Path, alpha: int) -> dict[str, tuple[torch.Tensor, torch.Tensor, float]]:
    pairs: dict[str, tuple[torch.Tensor, torch.Tensor, float]] = {}
    with open_tensors(path, framework="pt", device="cpu") as checkpoint:
        keys = set(checkpoint.keys())
        a_keys = sorted(key for key in keys if key.endswith(LORA_A_SUFFIX))
        if not a_keys or len(keys) != 2 * len(a_keys):
            raise ValueError(f"{path.name} is not a pure paired Diffusers Turbo LoRA")
        for a_key in a_keys:
            stem = a_key[: -len(LORA_A_SUFFIX)]
            b_key = stem + LORA_B_SUFFIX
            if b_key not in keys:
                raise ValueError(f"Turbo LoRA A has no B tensor: {a_key}")
            lora_a = checkpoint.get_tensor(a_key)
            lora_b = checkpoint.get_tensor(b_key)
            rank = int(lora_a.shape[0])
            pairs[stem] = (lora_a, lora_b, float(alpha) / rank)
    return pairs


def _validate_quant_descriptor(raw: torch.Tensor) -> None:
    descriptor = json.loads(bytes(raw.tolist()).decode("utf-8"))
    expected = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}
    if descriptor != expected:
        raise ValueError(f"unsupported Comfy quantization descriptor: {descriptor}")


def load_transformer(
    *,
    repo_root: Path,
    subfolder: str,
    checkpoint_path: Path,
    turbo_lora_path: Path | None,
    turbo_lora_alpha: int,
) -> MiniMaxH3Transformer3DModel:
    """Load one official/community H3 partition without materializing a second state dict."""
    from diffusers import MiniMaxH3Transformer3DModel

    config = MiniMaxH3Transformer3DModel.load_config(repo_root, subfolder=subfolder)
    with init_empty_weights():
        transformer = MiniMaxH3Transformer3DModel.from_config(config)

    with open_tensors(checkpoint_path, framework="pt", device="cpu") as checkpoint:
        keys = set(checkpoint.keys())
        descriptor_keys = sorted(key for key in keys if key.endswith(".comfy_quant"))
        if not descriptor_keys:
            raise ValueError(f"{checkpoint_path.name} is not an INT8 ConvRot checkpoint")
        for key in descriptor_keys:
            _validate_quant_descriptor(checkpoint.get_tensor(key))

        table = checkpoint.get_tensor("adaln_t_table")
        _install_curve_adaln(transformer, table)
        model_keys = sorted(
            key for key in keys if not key.endswith(".comfy_quant") and not key.endswith(".weight_scale")
        )
        full_state: dict[str, torch.Tensor] = {}
        quant_specs: list[tuple[str, str, dict]] = []
        for native_key in model_keys:
            shape = tuple(checkpoint.get_slice(native_key).get_shape())
            mappings = native_to_diffusers(native_key, shape)
            quantized = native_key.endswith(".weight") and (
                native_key[: -len(".weight")] + ".comfy_quant" in keys
            )
            if quantized:
                quant_specs.extend(
                    (mapping["target"][: -len(".weight")], native_key, mapping)
                    for mapping in mappings
                )
                continue
            if mappings:
                tensor = checkpoint.get_tensor(native_key)
                for mapping in mappings:
                    full_state[mapping["target"]] = _mapped_tensor(tensor, mapping)

        incompatible = transformer.load_state_dict(full_state, strict=False, assign=True)
        expected_missing = {name + ".weight" for name, _, _ in quant_specs}
        if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                "official ConvRot bridge mismatch: "
                f"missing={sorted(set(incompatible.missing_keys) - expected_missing)[:3]} "
                f"unexpected={incompatible.unexpected_keys[:3]}"
            )

        for target, native_key, mapping in quant_specs:
            prefix = native_key[: -len(".weight")]
            qdata = _mapped_tensor(checkpoint.get_tensor(native_key), mapping)
            scale = _mapped_tensor(checkpoint.get_tensor(prefix + ".weight_scale"), mapping)
            _replace_module(transformer, target, ConvRotLinear(qdata, scale))

    lora_pairs = {}
    lora_rank = 0
    lora_scale = 0.0
    if turbo_lora_path is not None:
        lora_pairs = _read_lora(turbo_lora_path, turbo_lora_alpha)
        # Every pair in a LightX2V release shares one rank, and `_read_lora` derives the scale from
        # it. Surface the resolved numbers rather than the requested alpha: a silent rank change
        # would otherwise move the applied strength with nothing in the log to show it.
        lora_scale = next(iter(lora_pairs.values()))[2] if lora_pairs else 0.0
        lora_rank = round(turbo_lora_alpha / lora_scale) if lora_scale else 0
        for target, (lora_a, lora_b, scale) in lora_pairs.items():
            base = transformer.get_submodule(target)
            if not isinstance(base, (nn.Linear, ConvRotLinear)):
                raise TypeError(f"Turbo target is not a supported linear projection: {target}")
            _replace_module(transformer, target, TurboLinear(base, lora_a, lora_b, scale))

    meta_parameters = [name for name, value in transformer.named_parameters() if value.is_meta]
    meta_buffers = [name for name, value in transformer.named_buffers() if value.is_meta]
    if meta_parameters or meta_buffers:
        raise RuntimeError(
            f"ConvRot loader left meta tensors: params={meta_parameters[:3]} buffers={meta_buffers[:3]}"
        )
    transformer._aiv_fp8_mode = "convrot-int8"
    transformer._is_turbo_baked = turbo_lora_path is not None
    transformer._aiv_convrot_report = {
        "int8_projections": len(quant_specs),
        "full_precision_tensors": len(full_state),
        "turbo_pairs": len(lora_pairs),
        "turbo_rank": lora_rank,
        "turbo_alpha": turbo_lora_alpha if lora_pairs else 0,
        "turbo_scale": lora_scale,
    }
    transformer.requires_grad_(False)
    transformer.eval()
    return transformer
