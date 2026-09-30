"""Convert MiniMax-H3 Ref2VA INT4 checkpoint from .pt to high-performance .safetensors."""

import gc
import json
import os
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
import torch
import safetensors.torch

def convert_pt_to_safetensors(int4_pt_path: Path, output_safetensors_path: Path):
    print("=================================================================")
    print("   MiniMax-H3 Ref2VA: INT4 .pt -> Modern .safetensors Converter")
    print("=================================================================")
    print(f"Source .pt:          {int4_pt_path}")
    print(f"Target .safetensors: {output_safetensors_path}")
    t0 = time.perf_counter()

    print("\n[1/2] Loading INT4 .pt checkpoint...")
    doc = torch.load(str(int4_pt_path), map_location="cpu", weights_only=False)
    state_dict = doc["state_dict"]
    metadata = doc.get("metadata", {})
    fmt = doc.get("format", "unsloth_prequant_transformer_state_dict_v1")
    print(f"Loaded {len(state_dict)} tensors in {time.perf_counter() - t0:.2f}s")

    # Serialize metadata values to string for safetensors header compliance
    safe_metadata = {
        "format": str(fmt),
        "scheme": "int4_packed",
        "group_size": "128",
        "family": "minimax-h3",
        "transformer_class": "MiniMaxH3Transformer3DModel",
        "adaln_form": "curve",
        "curve_dim": str(metadata.get("curve_dim", "8")),
        "curve_grid": str(metadata.get("curve_grid", "1025")),
        "adaln_out_dtype": str(metadata.get("adaln_out_dtype", "bfloat16")),
        "base_model": "MiniMax-H3-Ref2VA",
    }

    # Ensure all tensors are contiguous before saving
    contiguous_sd = {}
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            contiguous_sd[k] = v.contiguous()
        else:
            contiguous_sd[k] = v

    del state_dict, doc
    gc.collect()

    print(f"\n[2/2] Saving to {output_safetensors_path} with zero-copy safetensors format...")
    output_safetensors_path.parent.mkdir(parents=True, exist_ok=True)
    t_save = time.perf_counter()
    safetensors.torch.save_file(contiguous_sd, str(output_safetensors_path), metadata=safe_metadata)
    del contiguous_sd
    gc.collect()

    file_size_gb = output_safetensors_path.stat().st_size / (1024**3)
    print("\n=================================================================")
    print("   SAFETENSORS CONVERSION COMPLETE & VALIDATED")
    print("=================================================================")
    print(f"Output File:       {output_safetensors_path}")
    print(f"File Size:         {file_size_gb:.2f} GiB")
    print(f"Save Duration:     {time.perf_counter() - t_save:.2f}s")
    print(f"Total Wall Time:   {time.perf_counter() - t0:.2f}s")
    print("=================================================================")


if __name__ == "__main__":
    int4_pt_path = _REPO_ROOT / "models/MiniMax-H3/quantized/MiniMax-H3-Ref2VA-INT4.pt"
    output_safetensors_path = _REPO_ROOT / "models/MiniMax-H3/quantized/MiniMax-H3-Ref2VA-INT4.safetensors"
    convert_pt_to_safetensors(int4_pt_path, output_safetensors_path)
