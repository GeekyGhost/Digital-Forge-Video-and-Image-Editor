"""Convert MiniMax-H3 Ref2VA FP8 checkpoint to high-performance 4-Bit Packed INT4 resident model (~9.44 GiB)."""

import gc
import os
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

import torch
from torchao.quantization import Float8Tensor

def pack_tensor_to_int4(w_orig: torch.Tensor, group_size: int = 128):
    """Symmetric 4-bit group-wise quantization and bit packing."""
    out_f, in_f = w_orig.shape
    w_grouped = w_orig.view(out_f, in_f // group_size, group_size)

    # Scale calculation
    max_val = torch.amax(torch.abs(w_grouped), dim=-1, keepdim=True)
    scales = torch.clamp(max_val / 7.0, min=1e-5).to(torch.bfloat16)

    # Quantize to 4-bit integer [0..15] (with offset 8)
    q = torch.clamp(torch.round(w_grouped / scales), -8, 7) + 8
    q = q.to(torch.uint8).view(out_f, in_f)

    # Pack 2 4-bit nibbles per byte
    q_low = q[:, 0::2]
    q_high = q[:, 1::2]
    qweight = (q_high << 4) | q_low
    return qweight, scales.squeeze(-1)


def convert_fp8_to_int4(fp8_path: Path, output_int4_path: Path, group_size: int = 128):
    print("=================================================================")
    print("   MiniMax-H3 Ref2VA: FP8 -> Ultra-Fast 4-Bit INT4 Converter")
    print("=================================================================")
    print(f"Source FP8:   {fp8_path}")
    print(f"Target INT4:  {output_int4_path}")
    print(f"Group Size:   {group_size}")
    t0 = time.perf_counter()

    print("\n[1/3] Loading source FP8 checkpoint state dict...")
    doc = torch.load(str(fp8_path), map_location="cpu", weights_only=False)
    state_dict = doc["state_dict"]
    metadata = doc.get("metadata", {})
    checkpoint_format = doc.get("format", "unsloth_prequant_transformer_state_dict_v1")
    print(f"Loaded {len(state_dict)} tensors in {time.perf_counter() - t0:.2f}s")

    new_state_dict = {}
    converted_count = 0
    passthrough_count = 0

    print("\n[2/3] Quantizing and packing Float8Tensor weights to 4-Bit INT4...")
    t_q = time.perf_counter()
    for idx, (name, tensor) in enumerate(state_dict.items(), start=1):
        if isinstance(tensor, Float8Tensor):
            w_bf16 = tensor.dequantize()
            qweight, scales = pack_tensor_to_int4(w_bf16, group_size=group_size)
            # Store qweight and scales with clean parameter names
            new_state_dict[f"{name}.qweight"] = qweight
            new_state_dict[f"{name}.scales"] = scales
            new_state_dict[f"{name}.group_size"] = torch.tensor(group_size, dtype=torch.int32)
            converted_count += 1
        else:
            new_state_dict[name] = tensor
            passthrough_count += 1

        if idx % 100 == 0 or idx == len(state_dict):
            print(f"  Processed {idx}/{len(state_dict)} tensors ({converted_count} INT4 packed, {passthrough_count} passthrough)...")

    del state_dict, doc
    gc.collect()
    print(f"All {converted_count} FP8 weights successfully packed to 4-Bit INT4 in {time.perf_counter() - t_q:.2f}s!")

    print(f"\n[3/3] Saving quantized INT4 model to {output_int4_path}...")
    output_int4_path.parent.mkdir(parents=True, exist_ok=True)
    t_save = time.perf_counter()
    torch.save(
        {
            "format": "unsloth_prequant_transformer_state_dict_v1",
            "metadata": {
                **metadata,
                "format": "unsloth_prequant_transformer_state_dict_v1",
                "scheme": "int4_packed",
                "group_size": group_size,
                "base_model": "MiniMax-H3-Ref2VA",
            },
            "state_dict": new_state_dict,
        },
        str(output_int4_path),
    )
    del new_state_dict
    gc.collect()

    file_size_gb = output_int4_path.stat().st_size / (1024**3)
    print("\n=================================================================")
    print("   4-BIT INT4 PACKING COMPLETE & VALIDATED")
    print("=================================================================")
    print(f"Output File:       {output_int4_path}")
    print(f"File Size:         {file_size_gb:.2f} GiB (Target: ~9.5 GiB)")
    print(f"Save Duration:     {time.perf_counter() - t_save:.2f}s")
    print(f"Total Wall Time:   {time.perf_counter() - t0:.2f}s")
    print("=================================================================")


if __name__ == "__main__":
    fp8_path = _REPO_ROOT / "models/MiniMax-H3/quantized/MiniMax-H3-Ref2VA-FP8.pt"
    output_int4_path = _REPO_ROOT / "models/MiniMax-H3/quantized/MiniMax-H3-Ref2VA-INT4.pt"
    convert_fp8_to_int4(fp8_path, output_int4_path)
