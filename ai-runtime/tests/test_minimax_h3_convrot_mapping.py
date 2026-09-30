import json
import struct
from pathlib import Path

import pytest
import torch

from aiv_runtime.minimax_h3_convrot import _mapped_tensor, native_to_diffusers

# The four compute islands upstream keeps in fp32. The bridge gives them no `cast_dtype`, so the
# blob's own storage dtype is what they land in.
FP32_PROJECTIONS = (
    "video_patch_proj.weight",
    "audio_patch_proj.weight",
    "final_layer.video_out.weight",
    "final_layer.audio_out.weight",
)

PRODUCTION_DENOISERS = (
    "models/MiniMax-H3/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
    "models/MiniMax-H3/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors",
)


def _safetensors_header(path: Path) -> dict:
    """The header alone: an 8-byte little-endian length, then that many bytes of JSON.

    Deliberately not `safe_open`. On Windows that maps the whole file copy-on-write and charges
    every byte of it against the commit limit -- 20 GiB here, 27.5 GiB for the LTX checkpoint --
    to read a few kilobytes of metadata.
    """
    with path.open("rb") as blob:
        length = struct.unpack("<Q", blob.read(8))[0]
        return json.loads(blob.read(length))


def test_native_swiglu_halves_are_swapped_for_diffusers():
    mapping = native_to_diffusers("blocks.0.mlp.fc1.weight", (8, 3))[0]
    native = torch.arange(24).view(8, 3)

    converted = _mapped_tensor(native, mapping)

    assert mapping["target"] == "transformer_blocks.0.ff.net.0.proj.weight"
    assert torch.equal(converted[:4], native[4:])
    assert torch.equal(converted[4:], native[:4])


def test_native_swiglu_scale_rows_follow_the_weight_permutation():
    mapping = native_to_diffusers("blocks.0.mlp.fc1.weight", (8, 3))[0]
    scales = torch.arange(8).view(8, 1)

    converted = _mapped_tensor(scales, mapping)

    assert converted.flatten().tolist() == [4, 5, 6, 7, 0, 1, 2, 3]


def test_published_grouped_qkv_is_split_into_contiguous_roles():
    mappings = native_to_diffusers("blocks.0.attn.qkv_proj.weight", (18, 4))
    native = torch.arange(72).view(18, 4)

    converted = [_mapped_tensor(native, mapping) for mapping in mappings]

    assert [mapping["target"] for mapping in mappings] == [
        "transformer_blocks.0.attn.to_q.weight",
        "transformer_blocks.0.attn.to_k.weight",
        "transformer_blocks.0.attn.to_v.weight",
    ]
    assert all(torch.equal(value, native[index * 6 : (index + 1) * 6]) for index, value in enumerate(converted))


def test_curve_adaln_coefficients_are_promoted_to_native_compute_dtype():
    block_mapping = native_to_diffusers("blocks.0.adaln_proj.linear.weight", (12, 8))[0]
    final_mapping = native_to_diffusers("final_layer.adaln_proj.linear.bias", (12,))[0]

    block_value = _mapped_tensor(torch.ones(12, 8, dtype=torch.float16), block_mapping)
    final_value = _mapped_tensor(torch.ones(12, dtype=torch.float16), final_mapping)

    assert block_value.dtype == torch.float32
    assert final_value.dtype == torch.float32


@pytest.mark.parametrize("relative", PRODUCTION_DENOISERS)
def test_the_projections_the_bridge_does_not_cast_are_stored_fp32(relative):
    """Why `video_patch_proj` and friends carry no `cast_dtype` while `adaln_proj` does.

    Upstream locks these four to fp32 and the bridge loads with `assign=True`, which also bypasses
    `_keep_in_fp32_modules` -- so had the blob stored them at a lower precision they would have
    landed there silently. It does not: all four are F32 on disk. `final_layer.adaln_proj` is F16
    and is the one entry the bridge casts.
    """
    checkpoint = Path(__file__).resolve().parents[2] / relative
    if not checkpoint.is_file():
        pytest.skip(f"{relative} is not installed on this machine")

    header = _safetensors_header(checkpoint)

    for key in FP32_PROJECTIONS:
        assert header[key]["dtype"] == "F32", f"{key} is {header[key]['dtype']}, not F32"
    assert header["final_layer.adaln_proj.linear.weight"]["dtype"] == "F16"
