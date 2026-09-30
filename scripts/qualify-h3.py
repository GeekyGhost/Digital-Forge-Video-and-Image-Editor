"""MiniMax-H3 production qualification gate.

Runs the qualification gate in order and stops at the first stage that fails. Nothing here
generates video: stages 1-4 are the cheap, deterministic checks that have to pass *before* a
960x544/124-frame run is worth waiting for, and the script ends by printing the exact commands
for that run.

    vendor\\MiniMax-H3\\.venv\\Scripts\\python.exe scripts\\qualify-h3.py --stage all

Stages
------
hashes   Published SHA-256 for the two denoisers and the conditioner. Streamed, resumable across
         invocations via a cache keyed on (size, mtime) -- 69 GiB is not worth re-hashing to answer
         the same question twice.
mapping  Every native tensor in each checkpoint maps onto a real Diffusers target, the quantized
         set is exactly the set marked `comfy_quant`, every descriptor is the one the bridge
         supports, and every Turbo LoRA stem resolves to a module. Header-only: seconds, no weights.
kernel   One *real* quantized projection lifted out of the checkpoint, moved CPU->CUDA->CPU->CUDA
         and executed twice, then cross-checked against comfy-kitchen's independent eager backend.
         This is the check the TorchAO tensor-subclass path failed.
bridge   The full `load_transformer`, asserting no meta tensors survive. Materializes ~20 GiB of
         host RAM per partition, so it is last and runs one partition at a time.

`--partition` selects `fl2va`, `ref2va` or `both` (default). `--stage all` runs all four.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PACK = REPO_ROOT / "models" / "MiniMax-H3"
CACHE = REPO_ROOT / ".aiv" / "h3-qualification"

sys.path.insert(0, str(REPO_ROOT / "ai-runtime"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

#: Published Comfy-Org SHA-256. These are the whole-file hashes, i.e. the LFS oids on
#: huggingface.co/Comfy-Org/MiniMax-H3. A mirror that disagrees is a corrupt mirror.
PARTITIONS = {
    "fl2va": {
        "subfolder": "transformer",
        "checkpoint": PACK / "diffusion_models" / "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
        "sha256": "e889202c41dafb67b10d67b97f0d8541508036a6090af23425a5c2615d03c47a",
        "size": 20_970_379_616,
        "lora": PACK / "loras" / "minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors",
    },
    "ref2va": {
        "subfolder": "transformer_ref",
        "checkpoint": PACK / "diffusion_models" / "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
        "sha256": "9255f52b6677845ad238f20dfaafa94727053694127ab7f255c048f0f9365779",
        "size": 20_970_379_616,
        "lora": PACK / "loras" / "minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors",
    },
}
CONDITIONER = {
    "checkpoint": PACK / "text_encoders" / "qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
    "sha256": "bc2ced0fbea64757fa9acddccfc0b3f4819d1dcf1da6c124d690d368be283923",
    "size": 27_141_342_152,
}

EXPECTED_DESCRIPTOR = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}


class Failure(Exception):
    """A qualification stage that did not pass. The message is the finding."""


def say(message: str = "") -> None:
    print(message, flush=True)


def rule(title: str) -> None:
    say()
    say(f"=== {title} " + "=" * max(0, 74 - len(title)))


def read_header(path: Path) -> dict:
    """The safetensors header of `path`, without mapping a byte of tensor data."""
    import struct

    with path.open("rb") as handle:
        (length,) = struct.unpack("<Q", handle.read(8))
        if not 0 < length <= 100 * 1024 * 1024:
            raise Failure(f"{path.name}: implausible safetensors header length {length}")
        return json.loads(handle.read(length))


# --------------------------------------------------------------------------------------------
# Stage 1 - published hashes
# --------------------------------------------------------------------------------------------


def _cache_path() -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    return CACHE / "hashes.json"


def _load_cache() -> dict:
    try:
        return json.loads(_cache_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _identity(path: Path) -> str:
    stat = path.stat()
    return f"{stat.st_size}:{int(stat.st_mtime)}"


def sha256_of(path: Path, label: str) -> str:
    """SHA-256 of `path`, reusing a cached answer while size and mtime are unchanged."""
    cache = _load_cache()
    identity = _identity(path)
    entry = cache.get(str(path))
    if entry and entry.get("identity") == identity:
        say(f"  {label}: cached {entry['sha256'][:16]}...")
        return entry["sha256"]

    total = path.stat().st_size
    digest = hashlib.sha256()
    read = 0
    started = last = time.time()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
            read += len(chunk)
            if time.time() - last > 5:
                rate = read / max(time.time() - started, 1e-9) / 1024**2
                say(f"  {label}: {read / total:5.1%}  {rate:6.0f} MiB/s")
                last = time.time()
    value = digest.hexdigest()
    cache[str(path)] = {"identity": identity, "sha256": value}
    _cache_path().write_text(json.dumps(cache, indent=2), encoding="utf-8")
    elapsed = time.time() - started
    say(f"  {label}: {value[:16]}...  ({elapsed:.0f}s)")
    return value


def stage_hashes(names: list[str]) -> None:
    rule("1. published SHA-256")
    targets = [(name, PARTITIONS[name]) for name in names]
    targets.append(("conditioner", CONDITIONER))
    for label, spec in targets:
        path = Path(spec["checkpoint"])
        if not path.is_file():
            raise Failure(f"{label}: {path} is missing")
        size = path.stat().st_size
        if size != spec["size"]:
            raise Failure(
                f"{label}: {path.name} is {size} bytes, published size is {spec['size']}. "
                "The download is truncated or resumed against a different revision; re-fetch it "
                "rather than hashing an incomplete file."
            )
        actual = sha256_of(path, label)
        if actual != spec["sha256"]:
            raise Failure(
                f"{label}: SHA-256 mismatch.\n    expected {spec['sha256']}\n    actual   {actual}\n"
                "  Byte count matched, so this is corruption in transit, not a partial download."
            )
        say(f"  {label}: OK, matches the published hash")


# --------------------------------------------------------------------------------------------
# Stage 2 - structural mapping, header only
# --------------------------------------------------------------------------------------------


def _meta_transformer(subfolder: str, table_shape: tuple[int, int]):
    """The Diffusers module tree the bridge will load into, on meta, with curve AdaLN installed."""
    import torch
    from accelerate import init_empty_weights
    from diffusers import MiniMaxH3Transformer3DModel

    from aiv_runtime.minimax_h3_convrot import _install_curve_adaln

    config = MiniMaxH3Transformer3DModel.load_config(PACK, subfolder=subfolder)
    with init_empty_weights():
        transformer = MiniMaxH3Transformer3DModel.from_config(config)
    _install_curve_adaln(transformer, torch.empty(table_shape, device="meta", dtype=torch.float32))
    return transformer


def stage_mapping(names: list[str]) -> None:
    rule("2. structural mapping (header only)")
    from aiv_runtime.minimax_h3_convrot import (
        LORA_A_SUFFIX,
        LORA_B_SUFFIX,
        native_to_diffusers,
    )

    for name in names:
        spec = PARTITIONS[name]
        checkpoint = Path(spec["checkpoint"])
        if not checkpoint.is_file():
            raise Failure(f"{name}: {checkpoint} is missing")
        header = read_header(checkpoint)
        header.pop("__metadata__", None)
        keys = set(header)

        descriptors = sorted(key for key in keys if key.endswith(".comfy_quant"))
        if not descriptors:
            raise Failure(f"{name}: {checkpoint.name} carries no `comfy_quant` markers")

        # Every descriptor, not a sample: one projection published under a different scheme would
        # otherwise be dequantized by a kernel configured for the wrong rotation.
        import struct

        with checkpoint.open("rb") as handle:
            (length,) = struct.unpack("<Q", handle.read(8))
            base = 8 + length
            for key in descriptors:
                start, end = header[key]["data_offsets"]
                handle.seek(base + start)
                raw = json.loads(handle.read(end - start).decode("utf-8"))
                if raw != EXPECTED_DESCRIPTOR:
                    raise Failure(f"{name}: {key} declares {raw}, bridge supports {EXPECTED_DESCRIPTOR}")

        quantized_stems = {key[: -len(".comfy_quant")] for key in descriptors}
        scales = {key[: -len(".weight_scale")] for key in keys if key.endswith(".weight_scale")}
        if quantized_stems != scales:
            missing = sorted(quantized_stems - scales)[:3]
            extra = sorted(scales - quantized_stems)[:3]
            raise Failure(f"{name}: quantized markers and scales disagree ({missing=} {extra=})")

        for stem in sorted(quantized_stems):
            weight = header.get(stem + ".weight")
            scale = header[stem + ".weight_scale"]
            if weight is None or weight["dtype"] != "I8":
                raise Failure(f"{name}: {stem}.weight is not INT8")
            if scale["dtype"] != "F32" or scale["shape"] != [weight["shape"][0], 1]:
                raise Failure(
                    f"{name}: {stem}.weight_scale is {scale['dtype']}{scale['shape']}, "
                    f"expected F32[{weight['shape'][0]}, 1]"
                )
            groupsize = EXPECTED_DESCRIPTOR["convrot_groupsize"]
            if weight["shape"][1] % groupsize:
                raise Failure(
                    f"{name}: {stem} has in_features {weight['shape'][1]}, not a multiple of the "
                    f"declared ConvRot group size {groupsize}. The online rotation cannot apply."
                )

        table = header.get("adaln_t_table")
        if table is None:
            raise Failure(f"{name}: no `adaln_t_table`; this is not a curve-AdaLN release")
        transformer = _meta_transformer(spec["subfolder"], tuple(table["shape"]))
        model_keys = set(transformer.state_dict())

        expected_missing: set[str] = set()
        mapped = 0
        for native_key in sorted(keys):
            if native_key.endswith((".comfy_quant", ".weight_scale")):
                continue
            shape = tuple(header[native_key]["shape"])
            for mapping in native_to_diffusers(native_key, shape):
                target = mapping["target"]
                if target not in model_keys:
                    raise Failure(f"{name}: {native_key} maps to {target}, which the model has no slot for")
                mapped += 1
                if native_key[: -len(".weight")] in quantized_stems:
                    expected_missing.add(target)

        unmapped = model_keys - {
            mapping["target"]
            for native_key in keys
            if not native_key.endswith((".comfy_quant", ".weight_scale"))
            for mapping in native_to_diffusers(native_key, tuple(header[native_key]["shape"]))
        }
        if unmapped:
            raise Failure(
                f"{name}: the model has {len(unmapped)} slots no published tensor fills, "
                f"e.g. {sorted(unmapped)[:3]}"
            )

        lora = Path(spec["lora"])
        lora_note = "no adapter installed"
        if lora.is_file():
            lora_header = read_header(lora)
            metadata = lora_header.pop("__metadata__", None) or {}
            stems = {
                key[: -len(LORA_A_SUFFIX)] for key in lora_header if key.endswith(LORA_A_SUFFIX)
            }
            for stem in sorted(stems):
                if stem + LORA_B_SUFFIX not in lora_header:
                    raise Failure(f"{name}: Turbo LoRA {stem} has an A tensor with no B")
                try:
                    transformer.get_submodule(stem)
                except AttributeError as error:
                    raise Failure(f"{name}: Turbo LoRA targets {stem}, which does not exist") from error
            ranks = {
                lora_header[key + LORA_A_SUFFIX]["shape"][0] for key in stems
            }
            if len(ranks) != 1:
                raise Failure(f"{name}: Turbo LoRA mixes ranks {sorted(ranks)}")
            rank = ranks.pop()
            alpha = metadata.get("alpha")
            source = "declared" if alpha is not None else "filename table"
            from aiv_runtime.minimax_h3 import turbo_lora_alpha

            resolved = turbo_lora_alpha(lora)
            lora_note = (
                f"{len(stems)} pairs · rank {rank} · alpha {resolved} ({source}) · "
                f"scale {resolved / rank:g}"
            )
            if resolved / rank > 1.0 + 1e-6:
                raise Failure(
                    f"{name}: Turbo LoRA would apply at scale {resolved / rank:g}, above unity. "
                    "A distillation adapter above its trained strength destroys the output."
                )

        say(
            f"  {name}: {mapped} tensors map cleanly · {len(quantized_stems)} INT8 ConvRot "
            f"projections · {len(expected_missing)} filled by the kernel"
        )
        say(f"          turbo: {lora_note}")


# --------------------------------------------------------------------------------------------
# Stage 3 - one real ConvRot projection, moved and executed
# --------------------------------------------------------------------------------------------


def _implementation_name(comfy_kitchen, call: dict) -> str:
    """Which `int8_linear` would serve `call` right now, as `module.function`.

    Reported rather than assumed so the cross-check below is falsifiable: the registry picks by
    priority and skips disabled backends, but that is its behaviour, not a guarantee this call is
    covered by more than one of them.
    """
    try:
        implementation = comfy_kitchen.registry.get_implementation("int8_linear", kwargs=call)
    except Exception:  # noqa: BLE001
        return "unresolved"
    module = getattr(implementation, "__module__", "?").removeprefix("comfy_kitchen.backends.")
    return f"{module}.{getattr(implementation, '__name__', 'int8_linear')}"


def stage_kernel(names: list[str]) -> None:
    rule("3. real ConvRot projection, CPU->CUDA->CPU->CUDA")
    import torch
    from safetensors import safe_open

    if not torch.cuda.is_available():
        raise Failure("CUDA is not available; this stage is the point of the gate and cannot be skipped")

    import comfy_kitchen

    from aiv_runtime.minimax_h3_convrot import ConvRotLinear

    for name in names:
        spec = PARTITIONS[name]
        checkpoint = Path(spec["checkpoint"])
        stem = "blocks.0.attn.qkv_proj"
        with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
            qdata = handle.get_tensor(stem + ".weight")
            scale = handle.get_tensor(stem + ".weight_scale")
        projection = ConvRotLinear(qdata.contiguous(), scale.contiguous())
        say(f"  {name}: {stem} {tuple(qdata.shape)} INT8, scale {tuple(scale.shape)} FP32")

        torch.manual_seed(0)
        inputs = torch.randn(2, 37, projection.in_features, dtype=torch.bfloat16)

        # The move pattern group offload actually performs. The TorchAO tensor subclass raised a
        # CPU/CUDA storage-alias error on the second hop; ordinary Parameters must not.
        projection.to("cuda")
        first = projection(inputs.to("cuda"))
        projection.to("cpu")
        projection.to("cuda")
        second = projection(inputs.to("cuda"))

        if not torch.isfinite(first).all():
            raise Failure(f"{name}: the first ConvRot forward produced non-finite values")
        if not torch.equal(first, second):
            delta = (first.float() - second.float()).abs().max().item()
            raise Failure(
                f"{name}: the projection changed after a CPU round trip (max |delta| {delta:.3e}). "
                "Offloading this transformer would silently alter every step."
            )
        say(f"  {name}: bit-identical across the round trip · |out| max {first.abs().max().item():.3f}")

        # An independent implementation of the same contract. comfy-kitchen's eager backend shares
        # no code path with the CUDA kernel, so agreement here is evidence the bridge hands both of
        # them the same thing, rather than evidence that one kernel is self-consistent.
        # Name the implementation that actually served each call. Disabling a backend and trusting
        # that a *different* one answered is how a cross-check quietly becomes a comparison of one
        # kernel with itself -- which would report perfect agreement and prove nothing.
        call = {
            "x": inputs.to("cuda").contiguous(),
            "weight": projection.qdata,
            "weight_scale": projection.scale,
            "bias": None,
            "out_dtype": torch.bfloat16,
            "convrot": True,
            "convrot_groupsize": 256,
            "input_act": None,
        }
        primary = _implementation_name(comfy_kitchen, call)

        reference = None
        secondary = None
        try:
            comfy_kitchen.disable_backend("cuda")
            secondary = _implementation_name(comfy_kitchen, call)
            reference = comfy_kitchen.int8_linear(**call)
        except Exception as error:  # noqa: BLE001
            # No second backend can take this shape on this machine. That is a fact about the
            # install, not about the checkpoint, so it must not fail the gate -- but it must not
            # pass silently either, because the round-trip check above only proves the CUDA kernel
            # agrees with itself.
            say(f"  {name}: cross-check SKIPPED, no independent backend took it ({error})")
        finally:
            comfy_kitchen.enable_backend("cuda")

        if reference is not None:
            if secondary == primary:
                say(
                    f"  {name}: cross-check SKIPPED, disabling `cuda` still selected {primary}; "
                    "comparing it with itself would prove nothing"
                )
            else:
                magnitude = first.abs().mean().clamp(min=1e-6)
                divergence = ((first.float() - reference.float()).abs().mean() / magnitude).item()
                say(f"  {name}: {primary} vs {secondary} · {divergence:.3e} relative mean error")
                # Exact agreement is the expected answer, not a suspicious one: activations are
                # rotated and row-quantized deterministically, the GEMM accumulates in INT32, and
                # bfloat16 has eight mantissa bits to round into. A *large* divergence is the
                # finding here.
                if divergence > 5e-2:
                    raise Failure(
                        f"{name}: {primary} and {secondary} disagree by {divergence:.3e}. One of "
                        "them is not reading the published layout the way the bridge assumes."
                    )
        del projection, first, second, reference
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------------------------
# Stage 4 - the full bridge
# --------------------------------------------------------------------------------------------


def stage_bridge(names: list[str]) -> None:
    rule("4. full bridge load, no meta tensors")
    import torch

    from aiv_runtime.minimax_h3 import turbo_lora_alpha
    from aiv_runtime.minimax_h3_convrot import load_transformer

    for name in names:
        spec = PARTITIONS[name]
        lora = Path(spec["lora"])
        if not lora.is_file():
            raise Failure(
                f"{name}: {lora.name} is missing. The bridge-load stage exercises the published "
                "Turbo residual as well as Base; install the complete pack again."
            )
        started = time.time()
        transformer = load_transformer(
            repo_root=PACK,
            subfolder=spec["subfolder"],
            checkpoint_path=Path(spec["checkpoint"]),
            turbo_lora_path=lora,
            turbo_lora_alpha=turbo_lora_alpha(lora),
        )
        report = transformer._aiv_convrot_report
        say(
            f"  {name}: loaded in {time.time() - started:.0f}s · "
            f"{report['int8_projections']} INT8 projections · "
            f"{report['full_precision_tensors']} released-precision tensors · "
            f"{report['turbo_pairs']} Turbo pairs at scale {report['turbo_scale']:g}"
        )
        # `load_transformer` raises on meta tensors itself; assert it here too so this stage is a
        # real check rather than a restatement of the loader's own contract.
        stragglers = [
            name for name, value in transformer.named_parameters() if value.is_meta
        ] + [name for name, value in transformer.named_buffers() if value.is_meta]
        if stragglers:
            raise Failure(f"{name}: meta tensors survived the load: {stragglers[:3]}")
        del transformer
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------------------------


def windows(path: Path) -> str:
    """`path` as the user would type it: relative to the repo root, backslash separated."""
    try:
        return str(path.resolve().relative_to(REPO_ROOT)).replace("/", chr(92))
    except ValueError:
        return str(path).replace("/", chr(92))


def cmd_json(value: object) -> str:
    """A JSON argument that survives `cmd.exe`.

    The command-line parser strips one level of quoting before argv reaches Python, so every
    double quote inside the argument has to arrive backslash-escaped. Printing the raw JSON inside
    plain quotes produces a command that looks right and parses as several broken arguments -- the
    kind of instruction that costs an hour before anyone suspects the instruction itself.
    """
    return json.dumps(value, separators=(",", ":")).replace('"', chr(92) + '"')


def next_steps(names: list[str]) -> None:
    rule("5. the generation gate -- run these by hand")
    say("  960x544, 124 frames, native SDPA. Check finite latents, recognizable colour, and")
    say("  non-static audio that stays in sync.")
    say()
    if "fl2va" in names:
        say("  FL2VA text-to-video first, even when Ref2VA is the goal. It exercises the whole")
        say("  shared stack -- conditioner, denoiser, both VAEs, scheduler -- without the")
        say("  reference decode. If this works and Ref2VA does not, the fault is in the reference")
        say("  path; if neither works, it is in the shared stack. That split is worth one run.")
        say()
        lora = windows(PARTITIONS["fl2va"]["lora"])
        say("    vendor\\MiniMax-H3\\.venv\\Scripts\\python.exe ai-runtime\\aiv_runtime\\minimax_h3.py ^")
        say("      --repo-root models\\MiniMax-H3 --mode text --sampling-mode turbo-8 --steps 8 ^")
        # `validate` refuses any turbo mode without its adapter, and the flag defaults to None.
        say(f"      --turbo-lora-path {lora} ^")
        say("      --width 960 --height 544 --num-frames 124 --seed 42 --attention-backend native ^")
        say(f"      --offload balanced --output-path {windows(CACHE)}\\fl2va.mp4 ^")
        say('      --prompt "<your prompt>"')
        say()
    if "ref2va" in names:
        refs = CACHE / "refs"
        image, video, audio = refs / "ref-image.png", refs / "ref-video.mp4", refs / "ref-audio.wav"
        have_refs = all(path.is_file() for path in (image, video, audio))
        say("  Ref2VA Base at 20 NFE. Start with ONE case, not the four-case matrix -- the first")
        say("  run is about whether the pipeline produces anything at all.")
        say()
        if have_refs:
            say(f"  A synthetic reference set is already at {windows(refs)}\\ .")
            say("  Colour bars and tones: good for spotting a colour-space or INT8 outlier fault,")
            say("  useless for judging identity fidelity. Swap in real media before believing a")
            say("  quality verdict.")
        else:
            say("  No reference set found. Make one (any 2-15 s media works):")
            say(f"    ffmpeg -y -f lavfi -i testsrc2=size=960x544:rate=1 -frames:v 1 {windows(image)}")
        say()
        references = cmd_json(
            [{"kind": "image", "path": windows(image), "includeEmbeddedAudio": True}]
        )
        say("    vendor\\MiniMax-H3\\.venv\\Scripts\\python.exe ai-runtime\\aiv_runtime\\minimax_h3_ref.py ^")
        say("      --repo-root models\\MiniMax-H3 ^")
        say("      --transformer-path models\\MiniMax-H3\\diffusion_models\\minimax_h3_ref2va_pruned_int8_convrot.safetensors ^")
        say("      --text-encoder-path models\\MiniMax-H3\\text_encoders\\qwen3vl_32b_minimax_h3_int8_convrot.safetensors ^")
        say("      --sampling-mode base --steps 20 ^")
        say(f"      --references-json \"{references}\" ^")
        say("      --prompt \"<six-field Ref2V prompt>\" ^")
        say(f"      --output-path {windows(CACHE)}\\ref2va.mp4 --audio-path {windows(CACHE)}\\ref2va.wav ^")
        say("      --width 960 --height 544 --num-frames 124 --seed 42 ^")
        say("      --attention-backend native --offload balanced")
        say()
        if have_refs:
            say("  Then the full four-case reference matrix, once one case works:")
            say("    powershell -NoProfile -ExecutionPolicy Bypass -File .\\scripts\\benchmark-h3-ref2va.ps1 ^")
            say(f"      -Image {windows(image)} -Video {windows(video)} ^")
            say(f"      -Audio {windows(audio)} ^")
            say("      -SamplingMode base -Steps 20 -Attention native -Width 960 -Height 544 -Frames 124")
            say()
    say("  Then measure rather than eyeball the result. The MP4 carries no audio stream by design,")
    say("  so a silent preview is expected -- the sidecar WAV is the audio, and a foley failure")
    say("  looks like a WAV that is flat, not a video that is quiet:")
    say(f"    ffprobe -v error -show_entries stream=codec_type,width,height,nb_frames,duration ^")
    say(f"      -of default=noprint_wrappers=1 {windows(CACHE)}\\ref2va.mp4")
    say(f"    ffprobe -v error -f lavfi -i \"amovie={windows(CACHE)}\\ref2va.wav,astats=metadata=1\" ^")
    say("      -show_entries frame_tags=lavfi.astats.Overall.RMS_level -of csv=p=0 | sort -u | head")
    say()
    say("  A run that produces finite latents but static noise is a *different* failure from one")
    say("  that produces nothing; keep the worker log either way -- it now states the resolved")
    say("  Turbo rank, alpha, provenance and scale on the ConvRot path.")


STAGES = {
    "hashes": stage_hashes,
    "mapping": stage_mapping,
    "kernel": stage_kernel,
    "bridge": stage_bridge,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=(*STAGES, "all"), default="all")
    parser.add_argument("--partition", choices=("fl2va", "ref2va", "both"), default="both")
    args = parser.parse_args()

    names = ["fl2va", "ref2va"] if args.partition == "both" else [args.partition]
    order = list(STAGES) if args.stage == "all" else [args.stage]

    say(f"MiniMax-H3 qualification · partitions: {', '.join(names)} · stages: {', '.join(order)}")
    say(f"pack: {PACK}")

    for stage in order:
        try:
            STAGES[stage](names)
        except Failure as failure:
            say()
            say(f"FAILED at stage `{stage}`:")
            say(f"  {failure}")
            say()
            say("Later stages were not run. Fix this before spending a generation on it.")
            return 1
    if args.stage == "all":
        next_steps(names)
    say()
    say("All requested stages passed.")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("DIFFUSERS_OFFLINE", "1")
    raise SystemExit(main())
