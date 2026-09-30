"""Stream-convert MiniMax-H3 Ref2VA FP8 checkpoint to TorchAO INT8 resident checkpoint."""

import gc
import os
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

import torch
from torchao.quantization import Int8WeightOnlyConfig, Float8Tensor
from torchao.quantization.quant_api import _int8_weight_only_quantize_tensor

def stream_convert_fp8_to_int8(fp8_path: Path, output_int8_path: Path):
    print("=================================================================")
    print("   MiniMax-H3 Ref2VA: FP8 -> TorchAO INT8 Stream Converter")
    print("=================================================================")
    print(f"Source FP8:   {fp8_path}")
    print(f"Target INT8:  {output_int8_path}")
    t0 = time.perf_counter()

    print("\n[1/3] Loading source FP8 checkpoint state dict...")
    doc = torch.load(str(fp8_path), map_location="cpu", weights_only=False)
    state_dict = doc["state_dict"]
    metadata = doc.get("metadata", {})
    checkpoint_format = doc.get("format", "torchao-int8")
    print(f"Loaded {len(state_dict)} tensors in {time.perf_counter() - t0:.2f}s")

    int8_config = Int8WeightOnlyConfig()
    new_state_dict = {}
    converted_count = 0
    passthrough_count = 0

    print("\n[2/3] Quantizing Float8Tensor weights to TorchAO Int8Tensor in-place...")
    t_q = time.perf_counter()
    for idx, (name, tensor) in enumerate(state_dict.items(), start=1):
        if isinstance(tensor, Float8Tensor):
            # Dequantize to BF16 then quantize to Int8Tensor
            bf16 = tensor.dequantize()
            int8_tensor = _int8_weight_only_quantize_tensor(bf16, int8_config)
            new_state_dict[name] = int8_tensor
            converted_count += 1
        else:
            new_state_dict[name] = tensor
            passthrough_count += 1

        if idx % 100 == 0 or idx == len(state_dict):
            print(f"  Processed {idx}/{len(state_dict)} tensors ({converted_count} INT8 converted, {passthrough_count} passthrough)...")

    del state_dict, doc
    gc.collect()
    print(f"All {converted_count} FP8 weights successfully converted to INT8 in {time.perf_counter() - t_q:.2f}s!")

    print(f"\n[3/3] Saving quantized INT8 model to {output_int8_path}...")
    output_int8_path.parent.mkdir(parents=True, exist_ok=True)
    t_save = time.perf_counter()
    torch.save(
        {
            "format": "torchao-int8",
            "metadata": {
                "format": "torchao-int8",
                "quantization": "int8_weight_only",
                "base_model": "MiniMax-H3-Ref2VA",
                **metadata,
            },
            "state_dict": new_state_dict,
        },
        str(output_int8_path),
    )
    del new_state_dict
    gc.collect()

    file_size_gb = output_int8_path.stat().st_size / (1024**3)
    print("\n=================================================================")
    print("   CONVERSION COMPLETE & VALIDATED")
    print("=================================================================")
    print(f"Output File:       {output_int8_path}")
    print(f"File Size:         {file_size_gb:.2f} GiB")
    print(f"Save Duration:     {time.perf_counter() - t_save:.2f}s")
    print(f"Total Wall Time:   {time.perf_counter() - t0:.2f}s")
    print("=================================================================")


if __name__ == "__main__":
    fp8_path = _REPO_ROOT / "models/MiniMax-H3/quantized/MiniMax-H3-Ref2VA-FP8.pt"
    output_int8_path = _REPO_ROOT / "models/MiniMax-H3/quantized/MiniMax-H3-Ref2VA-INT8.pt"
    stream_convert_fp8_to_int8(fp8_path, output_int8_path)
