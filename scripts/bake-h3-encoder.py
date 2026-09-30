"""Bake MiniMax-H3's Qwen3-VL-32B conditioner to int8 without exhausting Windows commit.

The conditioner is too large to quantize safely in one long-lived process on a 64 GiB Windows
machine. PyTorch's allocator and safetensors mappings keep enough commit charged that the next
4.5 GiB shard eventually fails with WinError 1455 even when Transformers uses low-memory loading.

This script therefore treats every source safetensors shard as a separate transaction:

* a child process opens one bf16 shard, quantizes its decoder projections, writes one .bin shard,
  and exits so Windows releases every mapping and allocator arena;
* the parent records completed shards in ``text_encoder_int8.partial/.bake-state.json`` so an
  interrupted bake resumes instead of starting over;
* a fresh child process loads the completed checkpoint and checks every expected int8 weight;
* only then is the partial directory published as ``text_encoder_int8`` and, with ``--prune``, the
  bf16 source removed.

Run it after ``scripts/models.py install minimax-h3``::

    vendor\\MiniMax-H3\\.venv\\Scripts\\python.exe scripts\\bake-h3-encoder.py --prune
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
H3_PYTHON = ROOT / "vendor" / "MiniMax-H3" / ".venv" / "Scripts" / "python.exe"
TRANSFORMERS_PIN = "4.57.6"
CONDITIONING_LAYER = 50
BAKE_STATE_VERSION = 1

MODULES_TO_NOT_CONVERT = [
    "model.visual",
    "model.language_model.embed_tokens",
    "model.language_model.norm",
    "lm_head",
]

_LAYER_KEY = re.compile(r"^model\.language_model\.layers\.(\d+)\.")
_QUANTIZED_WEIGHT = re.compile(
    r"^model\.language_model\.layers\.\d+\."
    r"(?:mlp\.(?:down_proj|gate_proj|up_proj)|"
    r"self_attn\.(?:k_proj|o_proj|q_proj|v_proj))\.weight$"
)
_SOURCE_WEIGHT = re.compile(r"^model-\d+-of-\d+\.safetensors$")


def human(bytes_: int) -> str:
    return f"{bytes_ / 1024 ** 3:.1f} GiB"


def directory_size(path: Path) -> int:
    if not path.is_dir():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def write_json_atomic(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def keep_key(name: str, keep_layers: int) -> bool:
    match = _LAYER_KEY.match(name)
    return not match or not keep_layers or int(match.group(1)) < keep_layers


def is_quantized_weight(name: str) -> bool:
    return _QUANTIZED_WEIGHT.fullmatch(name) is not None


def read_weight_map(source: Path) -> dict[str, str]:
    index_path = source / "model.safetensors.index.json"
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index["weight_map"]
    except (OSError, KeyError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read the source weight index at {index_path}: {error}") from error
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"The source weight index at {index_path} is empty")
    return weight_map


def build_plan(source: Path, keep_layers: int) -> list[dict]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for key, shard in read_weight_map(source).items():
        if keep_key(key, keep_layers):
            grouped[shard].append(key)

    source_shards = sorted(grouped)
    plan = []
    total = len(source_shards)
    for ordinal, source_shard in enumerate(source_shards, start=1):
        keys = sorted(grouped[source_shard])
        plan.append(
            {
                "source": source_shard,
                "output": f"pytorch_model-{ordinal:05d}-of-{total:05d}.bin",
                "keys": keys,
                "quantized": sum(is_quantized_weight(key) for key in keys),
            }
        )
    return plan


def quantization_config_dict() -> dict:
    # This is TorchAoConfig(Int8WeightOnlyConfig(version=2), ...).to_dict() under the pinned
    # environment. Keeping it data-only lets the coordinator avoid importing torch at all.
    return {
        "quant_method": "torchao",
        "quant_type": {
            "default": {
                "_type": "Int8WeightOnlyConfig",
                "_version": 2,
                "_data": {
                    "group_size": None,
                    "granularity": {
                        "_type": "PerRow",
                        "_version": 1,
                        "_data": {"dim": -1},
                    },
                    "set_inductor_config": True,
                },
            }
        },
        "modules_to_not_convert": MODULES_TO_NOT_CONVERT,
        "quant_type_kwargs": {},
        "include_input_output_embeddings": False,
        "untie_embedding_weights": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantize the H3 conditioner to int8 on disk.")
    parser.add_argument("--pack", default=str(ROOT / "models" / "MiniMax-H3"))
    parser.add_argument(
        "--keep-layers",
        type=int,
        default=CONDITIONING_LAYER + 1,
        help=f"Keep this many decoder layers. H3 reads hidden_states[{CONDITIONING_LAYER}], so "
        f"the minimum safe value is {CONDITIONING_LAYER + 1}. Use 0 to keep all layers.",
    )
    parser.add_argument(
        "--prune",
        action="store_true",
        help="Delete the bf16 shards only after a fresh process verifies the int8 copy.",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Validate and print the shard plan without reading a weight tensor.",
    )

    # Internal worker protocol. The coordinator is the only caller.
    parser.add_argument("--worker-mode", choices=("shard", "verify"), help=argparse.SUPPRESS)
    parser.add_argument("--source-shard", help=argparse.SUPPRESS)
    parser.add_argument("--output-shard", help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint-dir", help=argparse.SUPPRESS)
    return parser.parse_args()


def require_environment() -> None:
    try:
        transformers_version = importlib.metadata.version("transformers")
        importlib.metadata.version("torchao")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError(
            f"Missing {error.name}. Run this under the MiniMax-H3 environment:\n\n"
            f"  {H3_PYTHON} scripts\\bake-h3-encoder.py --prune"
        ) from error
    if transformers_version != TRANSFORMERS_PIN:
        raise RuntimeError(
            f"This resumable bake requires transformers=={TRANSFORMERS_PIN}, but the H3 "
            f"environment has {transformers_version}. Repair it with:\n\n"
            f"  uv pip install --python \"{H3_PYTHON}\" \"transformers=={TRANSFORMERS_PIN}\""
        )


def shard_worker(args: argparse.Namespace) -> int:
    require_environment()
    if not args.source_shard or not args.output_shard:
        raise ValueError("the shard worker requires --source-shard and --output-shard")

    import torch
    from safetensors import safe_open
    from torchao.quantization.granularity import PerRow
    from torchao.quantization.quantize_.workflows.int8.int8_tensor import Int8Tensor

    source = Path(args.pack) / "text_encoder"
    source_shard = source / args.source_shard
    output_shard = Path(args.output_shard)
    weight_map = read_weight_map(source)
    keys = sorted(
        key
        for key, shard in weight_map.items()
        if shard == args.source_shard and keep_key(key, args.keep_layers)
    )
    if not keys:
        raise ValueError(f"No retained weights belong to {args.source_shard}")

    print(
        f"  {args.source_shard}: {len(keys)} weights, "
        f"{sum(is_quantized_weight(key) for key in keys)} int8",
        flush=True,
    )
    state_dict = {}
    with torch.no_grad(), safe_open(source_shard, framework="pt", device="cpu") as checkpoint:
        available = set(checkpoint.keys())
        missing = [key for key in keys if key not in available]
        if missing:
            raise ValueError(f"{source_shard.name} is missing indexed weight {missing[0]}")
        for key in keys:
            tensor = checkpoint.get_tensor(key)
            if is_quantized_weight(key):
                tensor = Int8Tensor.from_hp(tensor, granularity=PerRow())
            state_dict[key] = tensor

    output_shard.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_shard.with_name(output_shard.name + ".tmp")
    temporary.unlink(missing_ok=True)
    try:
        torch.save(state_dict, temporary)
        os.replace(temporary, output_shard)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"  wrote {output_shard.name} ({human(output_shard.stat().st_size)})", flush=True)
    return 0


def verify_worker(args: argparse.Namespace) -> int:
    require_environment()
    if not args.checkpoint_dir:
        raise ValueError("the verify worker requires --checkpoint-dir")

    import torch
    from transformers import Qwen3VLForConditionalGeneration

    checkpoint = Path(args.checkpoint_dir)
    print(f"Loading {checkpoint} in a fresh process for verification", flush=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        checkpoint,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    layers = getattr(getattr(model.config, "text_config", model.config), "num_hidden_layers", 0)
    if layers <= CONDITIONING_LAYER:
        raise ValueError(
            f"The int8 copy has {layers} layers, but H3 reads hidden_states[{CONDITIONING_LAYER}]"
        )

    parameters = 0
    quantized = 0
    meta = []
    for name, parameter in model.named_parameters():
        parameters += parameter.numel()
        if parameter.device.type == "meta":
            meta.append(name)
        if is_quantized_weight(name):
            if not hasattr(parameter, "qdata") or parameter.qdata.dtype != torch.int8:
                raise TypeError(f"Expected an Int8Tensor at {name}, got {type(parameter).__name__}")
            quantized += 1
    if meta:
        raise ValueError(f"The checkpoint left {len(meta)} parameters on meta, starting with {meta[0]}")
    expected_quantized = layers * 7
    if quantized != expected_quantized:
        raise ValueError(f"Loaded {quantized} int8 projections; expected {expected_quantized}")
    if parameters < 1_000_000_000:
        raise ValueError(f"The checkpoint holds only {parameters:,} logical parameters")

    print(
        f"Verified {parameters:,} parameters, {quantized} int8 projections, {layers} decoder layers",
        flush=True,
    )
    return 0


def source_identity(source: Path, keep_layers: int) -> dict:
    index = source / "model.safetensors.index.json"
    stat = index.stat()
    return {
        "version": BAKE_STATE_VERSION,
        "source": str(source.resolve()),
        "sourceIndexSize": stat.st_size,
        "sourceIndexMtimeNs": stat.st_mtime_ns,
        "keepLayers": keep_layers,
    }


def open_bake_state(staging: Path, identity: dict) -> tuple[Path, dict]:
    manifest = staging / ".bake-state.json"
    if staging.exists():
        if not manifest.is_file():
            if any(staging.iterdir()):
                raise RuntimeError(
                    f"{staging} exists without a bake manifest. Move or remove it before retrying."
                )
            state = {**identity, "completed": []}
            write_json_atomic(manifest, state)
            return manifest, state
        state = json.loads(manifest.read_text(encoding="utf-8"))
        for key, value in identity.items():
            if state.get(key) != value:
                raise RuntimeError(
                    f"{staging} belongs to a different source or layer plan. Move or remove it "
                    "before retrying."
                )
        state.setdefault("completed", [])
        return manifest, state

    staging.mkdir(parents=True)
    state = {**identity, "completed": []}
    write_json_atomic(manifest, state)
    return manifest, state


def copy_support_files(pack: Path, source: Path, staging: Path) -> None:
    for item in source.iterdir():
        if not item.is_file() or item.stat().st_size >= 64 * 1024 * 1024:
            continue
        if item.name == "model.safetensors.index.json" or _SOURCE_WEIGHT.fullmatch(item.name):
            continue
        shutil.copy2(item, staging / item.name)

    for name in ("tokenizer", "processor"):
        folder = pack / name
        if not folder.is_dir():
            continue
        for item in folder.iterdir():
            if item.is_file() and item.stat().st_size < 64 * 1024 * 1024:
                shutil.copy2(item, staging / item.name)


def write_checkpoint_metadata(
    source: Path, staging: Path, plan: list[dict], keep_layers: int
) -> None:
    copy_support_files(source.parent, source, staging)

    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    text_config = config.get("text_config", config)
    if keep_layers:
        text_config["num_hidden_layers"] = keep_layers
    config["quantization_config"] = quantization_config_dict()
    config["transformers_version"] = TRANSFORMERS_PIN
    write_json_atomic(staging / "config.json", config)

    weight_map = {}
    for item in plan:
        for key in item["keys"]:
            weight_map[key] = item["output"]
    total_size = sum((staging / item["output"]).stat().st_size for item in plan)
    write_json_atomic(
        staging / "pytorch_model.bin.index.json",
        {"metadata": {"total_size": total_size}, "weight_map": weight_map},
    )


def run_worker(command: list[str], description: str) -> None:
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if completed.returncode:
        raise RuntimeError(f"{description} failed with exit code {completed.returncode}")


def coordinator(args: argparse.Namespace) -> int:
    require_environment()
    pack = Path(args.pack)
    source = pack / "text_encoder"
    destination = pack / "text_encoder_int8"
    staging = pack / "text_encoder_int8.partial"

    if not source.is_dir():
        print(f"No bf16 conditioner at {source}.", file=sys.stderr)
        if destination.is_dir():
            print(f"An int8 conditioner already exists at {destination} -- nothing to do.")
            return 0
        print(
            "Install the pack first: scripts\\models.py install minimax-h3 --accept-licenses",
            file=sys.stderr,
        )
        return 2
    if destination.exists():
        print(f"{destination} already exists. The source was not touched.", file=sys.stderr)
        return 2
    if args.keep_layers and args.keep_layers <= CONDITIONING_LAYER:
        print(
            f"--keep-layers must be more than {CONDITIONING_LAYER}; H3 reads "
            f"hidden_states[{CONDITIONING_LAYER}].",
            file=sys.stderr,
        )
        return 6

    plan = build_plan(source, args.keep_layers)
    kept_keys = sum(len(item["keys"]) for item in plan)
    quantized_keys = sum(item["quantized"] for item in plan)
    expected_quantized = (args.keep_layers or 64) * 7
    if quantized_keys != expected_quantized:
        raise ValueError(
            f"The source has {quantized_keys} quantizable decoder projections; expected "
            f"{expected_quantized}. Refusing to bake an unfamiliar layout."
        )

    source_bytes = directory_size(source)
    excluded_gib = 5.0
    kept = 1.0 if not args.keep_layers else min(1.0, args.keep_layers / 64)
    estimate_gib = (source_bytes / 1024**3 * kept - excluded_gib) / 2 + excluded_gib
    partial_gib = directory_size(staging) / 1024**3
    free_gib = shutil.disk_usage(pack).free / 1024**3
    remaining_gib = max(0.0, estimate_gib - partial_gib)
    print(
        f"bf16 source {human(source_bytes)} | int8 estimate {estimate_gib:.1f} GiB | "
        f"disk free {free_gib:.1f} GiB"
    )
    print(
        f"Plan: {len(plan)} source shards -> {len(plan)} process-isolated output shards | "
        f"{kept_keys} weights | {quantized_keys} int8 projections"
    )
    if args.plan_only:
        for item in plan:
            print(
                f"  {item['source']} -> {item['output']}: {len(item['keys'])} weights, "
                f"{item['quantized']} int8"
            )
        return 0
    if free_gib < remaining_gib + 8:
        print(
            f"Refusing to start: about {remaining_gib:.1f} GiB remains to write plus 8 GiB of "
            f"headroom, but only {free_gib:.1f} GiB is free.",
            file=sys.stderr,
        )
        return 5

    manifest_path, state = open_bake_state(
        staging, source_identity(source, args.keep_layers)
    )
    completed_shards = set(state["completed"])
    script = str(Path(__file__).resolve())
    for ordinal, item in enumerate(plan, start=1):
        output_path = staging / item["output"]
        if item["source"] in completed_shards and output_path.is_file():
            print(f"[{ordinal}/{len(plan)}] already complete: {item['source']}")
            continue
        print(f"[{ordinal}/{len(plan)}] baking {item['source']}", flush=True)
        run_worker(
            [
                sys.executable,
                script,
                "--pack",
                str(pack),
                "--keep-layers",
                str(args.keep_layers),
                "--worker-mode",
                "shard",
                "--source-shard",
                item["source"],
                "--output-shard",
                str(output_path),
            ],
            f"worker for {item['source']}",
        )
        completed_shards.add(item["source"])
        state["completed"] = sorted(completed_shards)
        write_json_atomic(manifest_path, state)

    write_checkpoint_metadata(source, staging, plan, args.keep_layers)
    print(f"Wrote {human(directory_size(staging))} of int8 shards. Verifying in a fresh process.")
    run_worker(
        [
            sys.executable,
            script,
            "--pack",
            str(pack),
            "--worker-mode",
            "verify",
            "--checkpoint-dir",
            str(staging),
        ],
        "int8 verification",
    )

    manifest_path.unlink(missing_ok=True)
    staging.replace(destination)
    freed = 0
    if args.prune:
        freed = directory_size(source)
        shutil.rmtree(source)

    marker = pack / "aiv-encoder.json"
    write_json_atomic(
        marker,
        {
            "encoder": "text_encoder_int8",
            "quantization": "torchao-int8-v2",
            "decoderLayers": args.keep_layers or 64,
            "conditioningLayer": CONDITIONING_LAYER,
        },
    )
    if args.prune:
        print(f"Published the verified int8 encoder and deleted bf16, freeing {human(freed)}.")
    else:
        print("Published the verified int8 encoder. The bf16 source was kept.")
    return 0


def main() -> int:
    args = parse_args()
    if args.worker_mode == "shard":
        return shard_worker(args)
    if args.worker_mode == "verify":
        return verify_worker(args)
    return coordinator(args)


if __name__ == "__main__":
    raise SystemExit(main())
