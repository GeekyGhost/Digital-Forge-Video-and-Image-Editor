"""Verify a Digital Forge portable-FP8 checkpoint and its runtime projection math."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Mapping

import torch
from safetensors import safe_open


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = (
    ROOT
    / "models"
    / "MiniMax-H3"
    / "quantized"
    / "MiniMax-H3-Ref2VA-DF-Turbo-Portable-FP8.safetensors"
)
DEFAULT_SOURCE = ROOT / "models" / "MiniMax-H3" / "quantized" / "MiniMax-H3-Ref2VA-FP8.pt"
DEFAULT_LORA = (
    ROOT
    / "models"
    / "MiniMax-H3"
    / "loras"
    / "minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors"
)
sys.path.insert(0, str(ROOT / "ai-runtime"))

from aiv_runtime.minimax_h3 import EmbeddedLoraLinear, PortableFP8Linear  # noqa: E402
from aiv_runtime.minimax_h3_ref import validate as validate_ref2va  # noqa: E402


def verify_checkpoint(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    with safe_open(path, framework="pt", device="cpu") as checkpoint:
        metadata = checkpoint.metadata()
        keys = list(checkpoint.keys())
        qdata_keys = [key for key in keys if key.endswith(".aiv_fp8.qdata")]
        scale_keys = [key for key in keys if key.endswith(".aiv_fp8.scale")]
        shape_keys = [key for key in keys if key.endswith(".aiv_fp8.orig_shape")]
        lora_a_keys = [key for key in keys if key.endswith(".aiv_fp8.lora_A")]
        lora_b_keys = [key for key in keys if key.endswith(".aiv_fp8.lora_B")]
        lora_scale_keys = [key for key in keys if key.endswith(".aiv_fp8.lora_scale")]

        assert metadata.get("format") == "df_portable_source_fp8_v1", metadata
        assert len(qdata_keys) == len(scale_keys) == len(shape_keys) == 313
        assert len(lora_a_keys) == len(lora_b_keys) == len(lora_scale_keys) == 312
        assert all(checkpoint.get_tensor(key).dtype == torch.float8_e4m3fn for key in qdata_keys)
        assert all(checkpoint.get_tensor(key).dtype == torch.bfloat16 for key in lora_a_keys)
        assert all(checkpoint.get_tensor(key).dtype == torch.bfloat16 for key in lora_b_keys)


def verify_runtime_math() -> None:
    qdata = torch.tensor(
        [[1.0, -2.0, 0.5], [3.0, 1.0, -1.0]], dtype=torch.float8_e4m3fn
    )
    scale = torch.tensor([[0.25], [0.5]], dtype=torch.float32)
    base = PortableFP8Linear(qdata, scale, torch.tensor([2, 3], dtype=torch.int32))
    lora_a = torch.tensor([[1.0, 0.0, -1.0]], dtype=torch.bfloat16)
    lora_b = torch.tensor([[2.0], [-1.0]], dtype=torch.bfloat16)
    module = EmbeddedLoraLinear(base, lora_a, lora_b, torch.tensor(0.125))
    inputs = torch.tensor([[2.0, -1.0, 4.0]], dtype=torch.bfloat16)

    expected_weight = (qdata.float() * scale.float()).to(torch.bfloat16)
    expected_base = torch.nn.functional.linear(inputs, expected_weight)
    expected_lora = torch.nn.functional.linear(
        torch.nn.functional.linear(inputs, lora_a), lora_b
    )
    torch.testing.assert_close(module(inputs), expected_base + expected_lora * 0.125)
    assert module.weight.dtype == torch.bfloat16
    assert type(base.qdata.data) is torch.Tensor


def verify_self_contained(checkpoint_path: Path) -> None:
    with tempfile.TemporaryDirectory() as directory:
        encoder = Path(directory) / "text_encoder_int4"
        encoder.mkdir()
        args = argparse.Namespace(
            width=960,
            height=544,
            num_frames=124,
            prompt="verification",
            transformer_path=str(checkpoint_path),
            turbo_lora_path=str(Path(directory) / "deliberately-absent.safetensors"),
            text_encoder_path=str(encoder),
            vae_tile=256,
            stage="all",
            embeds_path=None,
            latents_path=None,
            references_json='[{"kind":"image"}]',
        )
        validate_ref2va(args)


def verify_exact_sources(checkpoint_path: Path, source_path: Path, lora_path: Path) -> None:
    """Read every low-bit/residual value and prove that conversion was lossless."""
    document = torch.load(source_path, map_location="cpu", weights_only=False, mmap=True)
    source_state = document.get("state_dict") if isinstance(document, Mapping) else None
    if not isinstance(source_state, Mapping):
        raise TypeError("source checkpoint does not contain a state_dict mapping")

    fp8_count = 0
    with safe_open(checkpoint_path, framework="pt", device="cpu") as checkpoint:
        for name, tensor in source_state.items():
            if not hasattr(tensor, "qdata") or not hasattr(tensor, "scale"):
                continue
            prefix = name.removesuffix(".weight") + ".aiv_fp8."
            assert torch.equal(checkpoint.get_tensor(prefix + "qdata"), tensor.qdata), name
            assert torch.equal(checkpoint.get_tensor(prefix + "scale"), tensor.scale), name
            assert tuple(checkpoint.get_tensor(prefix + "orig_shape").tolist()) == tuple(tensor.shape)
            fp8_count += 1
    assert fp8_count == 313

    lora_count = 0
    with safe_open(lora_path, framework="pt", device="cpu") as lora, safe_open(
        checkpoint_path, framework="pt", device="cpu"
    ) as checkpoint:
        for a_key in lora.keys():
            suffix = ".lora_A.default.weight"
            if not a_key.endswith(suffix):
                continue
            stem = a_key[: -len(suffix)]
            b_key = stem + ".lora_B.default.weight"
            prefix = stem + ".aiv_fp8."
            assert torch.equal(checkpoint.get_tensor(prefix + "lora_A"), lora.get_tensor(a_key))
            assert torch.equal(checkpoint.get_tensor(prefix + "lora_B"), lora.get_tensor(b_key))
            assert float(checkpoint.get_tensor(prefix + "lora_scale").item()) == 0.0625
            lora_count += 1
    assert lora_count == 312


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, nargs="?", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--deep", action="store_true", help="Compare every FP8 and LoRA value to its source")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--turbo-lora", type=Path, default=DEFAULT_LORA)
    args = parser.parse_args()
    verify_checkpoint(args.checkpoint.resolve())
    verify_runtime_math()
    verify_self_contained(args.checkpoint.resolve())
    if args.deep:
        verify_exact_sources(
            args.checkpoint.resolve(), args.source.resolve(), args.turbo_lora.resolve()
        )
    print(f"portable-FP8 verification passed: {args.checkpoint.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
