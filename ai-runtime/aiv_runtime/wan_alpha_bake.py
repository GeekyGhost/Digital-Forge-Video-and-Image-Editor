"""Bake Wan-Alpha v2.0 into a plain FP8 Wan2.1-T2V-14B DiT, one tensor at a time.

Upstream (WeChatCV/Wan-Alpha, `WanModel_dora_lightx2v`) loads the fp32 base DiT (~57 GB), merges
the Wan-Alpha DoRA into it (`load_dora`), then merges the LightX2V 4-step LoRA
(`WanLoraWrapper.apply_lora`). After that the model is an ordinary Wan2.1 14B DiT with different
weights -- nothing adapter-shaped survives to inference.

So the merge is done once, here, streaming: each base tensor is read, merged with the same
arithmetic as upstream (same order, same dtypes), and written straight out. Linear/conv weights
that Diffusers' layerwise casting would store in FP8 are written as `float8_e4m3fn`; everything
else keeps upstream's fp32. Peak memory is one input tensor plus one output shard, instead of the
57 GB fp32 model plus its state_dict copy upstream needs.

The resulting file is exactly what the spritesheet maker's 16 GB pattern would hold in RAM after
`enable_layerwise_casting(storage_dtype=float8_e4m3fn)`, so the runtime just loads it and streams
blocks.

Every read goes through `safetensors_reader` (pread, no copy-on-write mapping on Windows).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

#: Written next to the baked shards. Bump when the bake's output changes meaning.
BAKE_SCHEMA = 1

#: Modules left out of FP8 storage. Diffusers' own Wan transformer skips
#: `["patch_embedding", "condition_embedder", "norm"]`; upstream's module names for the condition
#: embedder are `text_embedding`, `time_embedding` and `time_projection`. `head` is added because
#: upstream runs it (and the time embedders) inside `autocast(dtype=float32)`, which CUDA treats as
#: "autocast off": a BF16-computed weight there meets an fp32 input and the matmul fails.
#: The bake and the runtime both pass this, so they cannot disagree about which tensors are FP8.
SKIP_LAYERWISE_CASTING = ("patch_embedding", "text_embedding", "time_embedding", "time_projection", "norm", "^head")
MANIFEST = "wan-alpha-bake.json"


def dora_targets(keys) -> dict[str, dict[str, str]]:
    """Upstream `load_dora.get_name_dict`, verbatim in behaviour.

    `diffusion_model.blocks.0.self_attn.q.lora_B.default.weight` targets
    `blocks.0.self_attn.q.weight`, with its `lora_A` and `lora_magnitude_vector` siblings.
    """
    targets: dict[str, dict[str, str]] = {}
    for key in keys:
        if ".lora_B." not in key:
            continue
        parts = key.split(".")
        b_index = parts.index("lora_B")
        if len(parts) > b_index + 2:
            parts.pop(b_index + 1)  # adapter name, e.g. "default"
        parts.pop(b_index)
        if parts[0] == "diffusion_model":
            parts.pop(0)
        target = ".".join(parts)
        magnitude = key.replace(".lora_B.", ".lora_magnitude_vector.").replace("lora_B", "lora_magnitude_vector")
        targets[target] = {"lora_B": key, "lora_A": key.replace(".lora_B.", ".lora_A."), "magnitude": magnitude}
    return targets


@dataclass
class LoraPlan:
    """Upstream `WanLoraWrapper._apply_lora_weights`, as a lookup by target parameter name."""

    pairs: dict[str, tuple[str, str]] = field(default_factory=dict)  # target -> (A key, B key)
    diffs: dict[str, str] = field(default_factory=dict)  # target -> diff key


def lora_plan(keys) -> LoraPlan:
    keys = list(keys)
    key_set = set(keys)
    plan = LoraPlan()
    for prefix in ("", "diffusion_model."):
        for key in keys:
            if not key.startswith(prefix):
                continue
            for suffix_a, suffix_b in (("lora_A.weight", "lora_B.weight"), ("lora_down.weight", "lora_up.weight")):
                if key.endswith(suffix_a):
                    target = key[len(prefix):].replace(suffix_a, "weight")
                    partner = key.replace(suffix_a, suffix_b)
                    if partner in key_set:
                        plan.pairs[target] = (key, partner)
            for suffix, replacement in (("diff", "weight"), ("diff_b", "bias"), ("diff_m", "modulation")):
                if key.endswith(suffix):
                    plan.diffs[key[len(prefix):].replace(suffix, replacement)] = key
    return plan


def apply_dora(weight, lora_b, lora_a, magnitude, ratio: float):
    """Upstream DoRA merge. `weight` is the fp32 base; the delta is computed in the LoRA's own
    dtype exactly as upstream does (`alpha * torch.mm(B, A)` on the stored tensors)."""
    import torch

    if lora_b.dim() == 4:
        delta = ratio * torch.mm(lora_b.squeeze(3).squeeze(2), lora_a.squeeze(3).squeeze(2))
        delta = delta.unsqueeze(2).unsqueeze(3)
    else:
        delta = ratio * torch.mm(lora_b, lora_a)
    merged = weight + delta
    if merged.dim() == 4:
        norm = torch.linalg.norm(merged, dim=tuple(range(1, merged.dim())), keepdim=True)
    else:
        norm = torch.linalg.norm(merged, dim=1, keepdim=True)
    norm = torch.clamp(norm, min=1e-8)
    return (magnitude.view_as(norm) / norm) * merged


def apply_lightx2v(name: str, param, plan: LoraPlan, lora: dict, alpha: float = 1.0) -> tuple[object, bool]:
    """Upstream LightX2V merge for one parameter. LoRA tensors are cast to the parameter's dtype
    first, as upstream casts them to `model.dtype` (fp32). Returns (param, applied)."""
    import torch

    if name in plan.pairs:
        a_key, b_key = plan.pairs[name]
        lora_a = lora[a_key].to(param.dtype)
        lora_b = lora[b_key].to(param.dtype)
        if param.shape == (lora_b.shape[0], lora_a.shape[1]):
            return param + torch.matmul(lora_b, lora_a) * alpha, True
        return param, False
    if name in plan.diffs:
        diff = lora[plan.diffs[name]].to(param.dtype)
        if param.shape == diff.shape:
            return param + diff * alpha, True
    return param, False


def fp8_parameter_names(model) -> set[str]:
    """The parameters Diffusers' `enable_layerwise_casting(float8_e4m3fn)` would store in FP8.

    Asked of Diffusers itself on a meta-device model, so the bake can never disagree with the
    runtime about which tensors are FP8.
    """
    import torch

    model.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16,
                                   skip_modules_pattern=SKIP_LAYERWISE_CASTING)
    names = {name for name, param in model.named_parameters() if param.dtype == torch.float8_e4m3fn}
    if not names:
        raise RuntimeError("layerwise casting selected no parameters; the Diffusers API has changed")
    return names


def shard_files(base_dir: Path) -> list[Path]:
    index = base_dir / "diffusion_pytorch_model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        return [base_dir / name for name in sorted(set(weight_map.values()))]
    single = base_dir / "diffusion_pytorch_model.safetensors"
    if single.is_file():
        return [single]
    raise FileNotFoundError(f"no diffusion_pytorch_model*.safetensors in {base_dir}")


def bake(base_dir: Path, dora_path: Path, lightx2v_path: Path, out_dir: Path, fp8_names: set[str],
         dora_ratio: float = 1.0, log=print) -> dict:
    """Stream-merge every base shard into `out_dir`. Returns the manifest it writes."""
    import torch
    from safetensors.torch import save_file

    from .safetensors_reader import describe_backend, load_state_dict, open_tensors

    dora = load_state_dict(dora_path)
    targets = dora_targets(dora.keys())
    lightx2v = load_state_dict(lightx2v_path)
    plan = lora_plan(lightx2v.keys())
    log(f"bake: {len(targets)} DoRA targets, {len(plan.pairs)} LightX2V pairs, {len(plan.diffs)} diffs")

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": BAKE_SCHEMA, "files": [], "doraApplied": 0, "lightx2vApplied": 0,
        "fp8Tensors": 0, "otherTensors": 0, "bytes": 0,
        "sources": {"base": [p.name for p in shard_files(base_dir)], "dora": dora_path.name, "lightx2v": lightx2v_path.name},
        "doraRatio": dora_ratio, "fp8Dtype": "float8_e4m3fn",
    }
    seen_dora: set[str] = set()
    for index, shard in enumerate(shard_files(base_dir)):
        out: dict[str, torch.Tensor] = {}
        with open_tensors(shard) as handle:
            for name in handle.keys():
                # Upstream loads the base in its stored dtype (fp32) and merges in that dtype.
                param = handle.get_tensor(name)
                if name in targets:
                    info = targets[name]
                    param = apply_dora(param, dora[info["lora_B"]], dora[info["lora_A"]], dora[info["magnitude"]], dora_ratio)
                    seen_dora.add(name)
                    manifest["doraApplied"] += 1
                param, applied = apply_lightx2v(name, param, plan, lightx2v)
                manifest["lightx2vApplied"] += int(applied)
                if name in fp8_names:
                    out[name] = param.to(torch.float8_e4m3fn).contiguous()
                    manifest["fp8Tensors"] += 1
                else:
                    out[name] = param.contiguous()
                    manifest["otherTensors"] += 1
                manifest["bytes"] += out[name].numel() * out[name].element_size()
        target = out_dir / f"wan-alpha-v2-dit-fp8-{index + 1:05d}.safetensors"
        save_file(out, str(target), metadata={"format": "pt", "aiv": "wan-alpha-bake"})
        manifest["files"].append(target.name)
        log(f"bake: wrote {target.name} ({len(out)} tensors)")
        del out
    missing = sorted(set(targets) - seen_dora)
    if missing:
        raise RuntimeError(f"{len(missing)} DoRA targets have no base tensor, e.g. {missing[:3]}")
    if manifest["lightx2vApplied"] == 0:
        raise RuntimeError("no LightX2V weight was applied (upstream asserts applied_count > 0)")
    manifest["readBackend"] = describe_backend()
    (out_dir / MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def baked_files(out_dir: Path) -> list[Path] | None:
    """The baked shards, or None if the bake is missing, partial or from another schema."""
    manifest_path = out_dir / MANIFEST
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != BAKE_SCHEMA:
        return None
    files = [out_dir / name for name in manifest.get("files", [])]
    return files if files and all(path.is_file() for path in files) else None
