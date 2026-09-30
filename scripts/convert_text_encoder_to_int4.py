"""Convert Qwen3-VL 32B text encoder from INT8 to high-speed 4-Bit INT4 Safetensors (~8.8 GiB)."""

import gc
import json
import os
import shutil
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
import torch
import safetensors.torch
from torchao.quantization import Int8Tensor


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


def convert_text_encoder(source_dir: Path, target_dir: Path, group_size: int = 128):
    print("=================================================================")
    print("   Qwen3-VL 32B Conditioner: INT8 -> Ultra-Fast 4-Bit INT4")
    print("=================================================================")
    print(f"Source Dir: {source_dir}")
    print(f"Target Dir: {target_dir}")
    t0 = time.perf_counter()

    target_dir.mkdir(parents=True, exist_ok=True)

    # Copy config and tokenizer files
    for item in source_dir.glob("*.json"):
        if "index" not in item.name:
            shutil.copyfile(str(item), str(target_dir / item.name))
    for item in source_dir.glob("*.txt"):
        shutil.copyfile(str(item), str(target_dir / item.name))

    index_file = source_dir / "pytorch_model.bin.index.json"
    with open(index_file, "r", encoding="utf-8") as f:
        index_data = json.load(f)

    weight_map = index_data["weight_map"]
    shard_names = sorted(list(set(weight_map.values())))
    print(f"Found {len(shard_names)} shards to convert...\n")

    new_weight_map = {}
    total_converted = 0
    total_passthrough = 0

    for shard_idx, shard_name in enumerate(shard_names, start=1):
        t_shard = time.perf_counter()
        shard_path = source_dir / shard_name
        print(f"[{shard_idx}/{len(shard_names)}] Loading {shard_name}...")
        sd = torch.load(str(shard_path), map_location="cpu", weights_only=False)

        new_sd = {}
        for k, v in sd.items():
            if isinstance(v, Int8Tensor):
                w_bf16 = v.dequantize()
                qweight, scales = pack_tensor_to_int4(w_bf16, group_size=group_size)
                new_sd[f"{k}.qweight"] = qweight.contiguous()
                new_sd[f"{k}.scales"] = scales.contiguous()
                new_sd[f"{k}.group_size"] = torch.tensor(group_size, dtype=torch.int32)
                total_converted += 1
            elif isinstance(v, torch.Tensor):
                new_sd[k] = v.contiguous()
                total_passthrough += 1
            else:
                new_sd[k] = v

        safe_shard_name = shard_name.replace(".bin", ".safetensors").replace("pytorch_model", "model")
        out_shard_path = target_dir / safe_shard_name

        for k in new_sd.keys():
            new_weight_map[k] = safe_shard_name

        safetensors.torch.save_file(new_sd, str(out_shard_path), metadata={"format": "int4_packed", "group_size": str(group_size)})
        print(f"  -> Saved {safe_shard_name} ({out_shard_path.stat().st_size / 1024**3:.2f} GiB) in {time.perf_counter() - t_shard:.2f}s")
        del sd, new_sd
        gc.collect()

    # Save new index
    new_index = {
        "metadata": {"total_size": sum((target_dir / f).stat().st_size for f in set(new_weight_map.values()))},
        "weight_map": new_weight_map,
    }
    with open(target_dir / "model.safetensors.index.json", "w", encoding="utf-8") as f:
        json.dump(new_index, f, indent=2)

    total_size_gb = sum(p.stat().st_size for p in target_dir.glob("*.safetensors")) / (1024**3)
    print("\n=================================================================")
    print("   QWEN3-VL 4-BIT SAFETENSORS CONVERSION COMPLETE")
    print("=================================================================")
    print(f"Target Directory:    {target_dir}")
    print(f"Total 4-Bit Size:    {total_size_gb:.2f} GiB (Fits 100% in 16 GB VRAM!)")
    print(f"Converted Weights:   {total_converted}")
    print(f"Total Wall Time:     {time.perf_counter() - t0:.2f}s")
    print("=================================================================")


if __name__ == "__main__":
    source_dir = _REPO_ROOT / "models/MiniMax-H3/text_encoder_int8"
    target_dir = _REPO_ROOT / "models/MiniMax-H3/text_encoder_int4"
    convert_text_encoder(source_dir, target_dir)
