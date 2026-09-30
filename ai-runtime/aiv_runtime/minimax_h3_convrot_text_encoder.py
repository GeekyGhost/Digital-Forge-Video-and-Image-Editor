"""Direct loader for Comfy-Org's published MiniMax-H3 Qwen3-VL conditioner.

The checkpoint is a pruned Qwen3-VL-32B model: it contains the first 50 decoder
layers because MiniMax-H3 consumes the unnormalised hidden state immediately
after layer 50.  The language projections use Comfy-Org's INT8 ConvRot format;
the vision tower and embeddings remain BF16.  This bridge does not import
ComfyUI and keeps every weight as an ordinary tensor so Diffusers leaf
offloading can move it safely.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from accelerate import init_empty_weights
from aiv_runtime.safetensors_reader import open_tensors
from transformers import AutoConfig, Qwen3VLForConditionalGeneration

from .minimax_h3_convrot import ConvRotLinear, _replace_module, _validate_quant_descriptor


PUBLISHED_METADATA = {
    "num_hidden_layers": 50,
    "output": "unnormalized_hidden_after_layer_50",
}


class _IdentityDecoderLayer(nn.Module):
    """Keep Transformers' 51-layer hidden-state indexing without layer 51 weights."""

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return hidden_states


def native_to_transformers(native_key: str) -> str:
    """Map the single-file Comfy-Org names to Transformers' Qwen3-VL tree."""
    if native_key.startswith("visual."):
        return "model." + native_key
    if native_key.startswith("model."):
        return "model.language_model." + native_key[len("model.") :]
    raise ValueError(f"unmapped Qwen3-VL conditioner tensor: {native_key}")


def _validate_metadata(metadata: dict[str, str] | None) -> None:
    raw = (metadata or {}).get("minimax_h3_te")
    if raw is None:
        raise ValueError("conditioner safetensor has no minimax_h3_te metadata")
    if json.loads(raw) != PUBLISHED_METADATA:
        raise ValueError(f"unsupported MiniMax-H3 conditioner metadata: {raw}")


def load_text_encoder(
    *,
    config_path: Path,
    checkpoint_path: Path,
) -> Qwen3VLForConditionalGeneration:
    """Load the published pruned conditioner without materialising a second model."""
    if checkpoint_path.suffix.lower() != ".safetensors":
        raise ValueError("MiniMax-H3 conditioner weights must be a .safetensors file")

    config = AutoConfig.from_pretrained(config_path)
    if config.text_config.num_hidden_layers < 51:
        raise ValueError(
            "MiniMax-H3's Qwen3-VL config must expose at least 51 layers so hidden_states[50] "
            "is the unnormalised output of the 50 released decoder layers"
        )
    # The upstream config describes the full Qwen model. Construct only the 51-layer indexing
    # shell needed for the published 50-layer H3 conditioner.
    config.text_config.num_hidden_layers = 51
    with init_empty_weights():
        model = Qwen3VLForConditionalGeneration(config)

    # hidden_states[50] is captured before this identity layer.  Replacing the
    # unused final layer and final norm avoids inventing weights absent from the
    # deliberately pruned release while retaining Diffusers' required indexing.
    model.model.language_model.layers[50] = _IdentityDecoderLayer()
    model.model.language_model.norm = nn.Identity()
    model.lm_head = nn.Identity()

    with open_tensors(checkpoint_path, framework="pt", device="cpu") as checkpoint:
        _validate_metadata(checkpoint.metadata())
        keys = set(checkpoint.keys())
        descriptor_keys = sorted(key for key in keys if key.endswith(".comfy_quant"))
        if len(descriptor_keys) != 350:
            raise ValueError(
                f"published conditioner must contain 350 ConvRot projections, got {len(descriptor_keys)}"
            )
        for key in descriptor_keys:
            _validate_quant_descriptor(checkpoint.get_tensor(key))

        quant_native = {
            key[: -len(".comfy_quant")] + ".weight" for key in descriptor_keys
        }
        scale_keys = {key[: -len(".comfy_quant")] + ".weight_scale" for key in descriptor_keys}
        missing_scales = sorted(scale_keys - keys)
        if missing_scales:
            raise ValueError(f"conditioner projection has no weight scale: {missing_scales[0]}")

        full_state: dict[str, torch.Tensor] = {}
        for native_key in sorted(keys - set(descriptor_keys) - scale_keys - quant_native):
            full_state[native_to_transformers(native_key)] = checkpoint.get_tensor(native_key)

        incompatible = model.load_state_dict(full_state, strict=False, assign=True)
        expected_missing = {native_to_transformers(key) for key in quant_native}
        if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                "published conditioner bridge mismatch: "
                f"missing={sorted(set(incompatible.missing_keys) - expected_missing)[:3]} "
                f"unexpected={incompatible.unexpected_keys[:3]}"
            )

        for native_weight in sorted(quant_native):
            native_prefix = native_weight[: -len(".weight")]
            target = native_to_transformers(native_prefix)
            qdata = checkpoint.get_tensor(native_weight)
            scale = checkpoint.get_tensor(native_prefix + ".weight_scale")
            _replace_module(model, target, ConvRotLinear(qdata, scale))

    meta_parameters = [name for name, value in model.named_parameters() if value.is_meta]
    meta_buffers = [name for name, value in model.named_buffers() if value.is_meta]
    if meta_parameters or meta_buffers:
        raise RuntimeError(
            "conditioner loader left meta tensors: "
            f"params={meta_parameters[:3]} buffers={meta_buffers[:3]}"
        )

    model._aiv_convrot_conditioner = True
    model._aiv_convrot_report = {
        "int8_projections": len(quant_native),
        "full_precision_tensors": len(full_state),
        "real_decoder_layers": PUBLISHED_METADATA["num_hidden_layers"],
    }
    model.requires_grad_(False)
    model.eval()
    return model
