"""Stage an approved dataset and run the pinned official Diffusers Z-Image trainer."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from aiv_runtime.zimage_training import lora_filename, stage_dataset, trigger_phrase

# Inherited by the trainer subprocess. Transformers v5's threaded weight materialiser has faulted
# in torch_cpu.dll on this machine; loading on one thread avoids it (see the caption worker).
os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")

# Attention-only LoRA (the Diffusers default) leaves the feed-forward blocks, where most of a DiT's
# visual detail is stored, untouched. AI Toolkit trains every linear layer of the Z-Image
# transformer; attention + SwiGLU feed-forward is the capacity that matters for identity and design.
DEFAULT_LORA_LAYERS = "to_q,to_k,to_v,to_out.0,feed_forward.w1,feed_forward.w2,feed_forward.w3"
LEGACY_LORA_LAYERS = "to_k,to_q,to_v,to_out.0"
# fp8 (torchao) keeps the 6B transformer's weights in bf16 (~12.3 GB) and only casts for the matmul,
# so it saves no weight memory on a 16 GB card and is slower without torch.compile. NF4 (QLoRA)
# stores the frozen weights in 4 bits (~3.5 GB), which is what actually fits a 16 GB GPU.
DEFAULT_PRECISION = "nf4"
PRECISIONS = {"nf4", "fp8", "bf16"}
# Z-Image Base ships shift=6.0, which puts ~86% of uniformly sampled training steps above sigma 0.5.
# Shift 3.0 (Turbo's value, and AI Toolkit's Z-Image training value) gives the low-noise steps that
# carry fine detail about twice the share.
DEFAULT_TIMESTEP_SHIFT = 3.0
# "base" trains on the undistilled Z-Image. "turbo" trains on Z-Image-Turbo with ostris' training
# adapter held active and frozen, so the result runs on Turbo at normal strength with 8-9 steps.
TRAINING_MODELS = {"base", "turbo"}


def model_label(config: dict[str, Any]) -> str:
    return "Z-Image-Turbo" if config.get("trainingModel") == "turbo" else "Z-Image"


def inference_hints(config: dict[str, Any]) -> dict[str, Any]:
    if config.get("trainingModel") == "turbo":
        return {"turbo": {"steps": 9, "guidanceScale": 0.0, "strength": 1.0}}
    # Trained on Base. On Turbo the adapter lands weaker and usually needs ~1.5-2.5 strength.
    return {
        "base": {"steps": 36, "guidanceScale": 4.0, "strength": 1.0},
        "turbo": {"steps": 9, "guidanceScale": 0.0, "strength": 1.75},
    }


def emit(event: str, **payload: Any) -> None:
    print(json.dumps({"event": event, **payload}, ensure_ascii=False), flush=True)


def resolve_training_settings(config: dict[str, Any]) -> dict[str, Any]:
    """Fill defaults for settings the UI does not expose yet, keeping resumes consistent.

    A resume must reproduce the adapter layout and precision of the run that wrote the checkpoint,
    otherwise the optimizer state no longer matches. Jobs started before these settings existed
    were attention-only fp8 runs at the model's own shift, so a resume without them keeps that.
    """
    legacy_resume = bool(str(config.get("resumeFromCheckpoint") or "").strip()) and "loraLayers" not in config
    if legacy_resume:
        config.setdefault("loraLayers", LEGACY_LORA_LAYERS)
        config.setdefault("precision", "fp8")
        config.setdefault("timestepShift", None)
    config.setdefault("loraLayers", DEFAULT_LORA_LAYERS)
    config.setdefault("precision", DEFAULT_PRECISION)
    config.setdefault("timestepShift", DEFAULT_TIMESTEP_SHIFT)
    config.setdefault("trainingModel", "base")
    if config["precision"] not in PRECISIONS:
        raise ValueError(f"unknown training precision: {config['precision']}")
    if config["trainingModel"] not in TRAINING_MODELS:
        raise ValueError(f"unknown training model: {config['trainingModel']}")
    if config["trainingModel"] == "turbo" and not str(config.get("assistantLoraPath") or "").strip():
        raise ValueError("Turbo training needs the Z-Image-Turbo training adapter (assistantLoraPath)")
    return config


def bnb_config_path(config: dict[str, Any]) -> Path:
    return Path(config["runDirectory"]) / "bnb-nf4.json"


def write_bnb_config(config: dict[str, Any]) -> Path:
    path = bnb_config_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    # The trainer sets bnb_4bit_compute_dtype from --mixed_precision itself.
    path.write_text(json.dumps({"load_in_4bit": True, "bnb_4bit_quant_type": "nf4"}), encoding="utf-8")
    return path


def finalize_lora(output_dir: Path, config: dict[str, Any]) -> tuple[Path, Path]:
    weights = output_dir / "pytorch_lora_weights.safetensors"
    if not weights.is_file():
        candidates = sorted(output_dir.glob("*.safetensors"))
        if not candidates:
            raise FileNotFoundError("training completed without a LoRA safetensors file")
        weights = candidates[-1]
    display_name = str(config.get("displayName") or config["name"]).strip()
    final_weights = output_dir / lora_filename(display_name)
    if weights.resolve() != final_weights.resolve():
        weights.replace(final_weights)
    weights = final_weights
    readme = output_dir / "README.md"
    if readme.is_file():
        contents = readme.read_text(encoding="utf-8")
        readme.write_text(contents.replace("pytorch_lora_weights.safetensors", weights.name), encoding="utf-8")
    manifest = {
        "name": display_name, "directoryName": config["name"],
        "baseModel": model_label(config), "baseModelPath": config["modelPath"],
        "profile": config["profile"], "trigger": config["trigger"], "className": config["className"],
        "triggerPhrase": trigger_phrase(config["profile"], config["trigger"], config["className"]),
        "resolution": config["resolution"], "steps": config["steps"], "rank": config["rank"],
        "alpha": config["alpha"], "learningRate": config["learningRate"], "seed": config["seed"],
        "loraLayers": config.get("loraLayers", LEGACY_LORA_LAYERS),
        "precision": config.get("precision", "fp8"), "timestepShift": config.get("timestepShift"),
        "images": len(config["images"]), "weights": str(weights), "weightsFile": weights.name,
        "inference": inference_hints(config),
    }
    if config.get("assistantLoraPath"):
        manifest["trainingAdapter"] = config["assistantLoraPath"]
    if config.get("initialLoraPath"):
        manifest["continuedFrom"] = config["initialLoraPath"]
    manifest_path = output_dir / "aiv-lora.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return weights, manifest_path


def checkpoint_manifest(checkpoint: Path, config: dict[str, Any], step: int) -> Path:
    """Attach AIV metadata to an Accelerate checkpoint without changing its resumable state."""
    weights = checkpoint / "pytorch_lora_weights.safetensors"
    if not weights.is_file():
        raise FileNotFoundError(f"checkpoint has no LoRA weights: {checkpoint}")
    display_name = str(config.get("displayName") or config["name"]).strip()
    manifest = {
        "name": f"{display_name} · checkpoint {step}",
        "directoryName": config["name"],
        "kind": "training-checkpoint",
        "baseModel": model_label(config),
        "baseModelPath": config["modelPath"],
        "profile": config["profile"],
        "trigger": config["trigger"],
        "className": config["className"],
        "triggerPhrase": trigger_phrase(config["profile"], config["trigger"], config["className"]),
        "resolution": config["resolution"],
        "step": step,
        "targetSteps": config["steps"],
        "rank": config["rank"],
        "alpha": config["alpha"],
        "learningRate": config["learningRate"],
        "seed": config["seed"],
        "images": len(config["images"]),
        "weights": str(weights),
        "weightsFile": weights.name,
    }
    path = checkpoint / "aiv-lora.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def available_checkpoints(output_dir: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for checkpoint in output_dir.glob("checkpoint-*"):
        try:
            step = int(checkpoint.name.split("-", 1)[1])
        except (IndexError, ValueError):
            continue
        weights = checkpoint / "pytorch_lora_weights.safetensors"
        if not weights.is_file():
            continue
        checkpoint_manifest(checkpoint, config, step)
        found.append({"step": step, "path": str(checkpoint), "weights": str(weights)})
    return sorted(found, key=lambda item: int(item["step"]))


def trainer_supports(trainer_script: str, flag: str) -> bool:
    try:
        return flag in Path(trainer_script).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False


def build_training_command(config: dict[str, Any], trainer_script: str) -> list[str]:
    command = [
        sys.executable,
        trainer_script,
        "--pretrained_model_name_or_path", config["modelPath"],
        "--dataset_name", str(Path(config["runDirectory"]) / "dataset"),
        "--image_column", "image",
        "--caption_column", "text",
        "--instance_prompt",
        trigger_phrase(config.get("profile", "subject"), config["trigger"], config["className"]),
        "--output_dir", config["outputDirectory"],
        "--mixed_precision", "bf16",
        "--resolution", str(config["resolution"]),
        "--train_batch_size", "1",
        "--gradient_accumulation_steps", "4",
        "--max_train_steps", str(config["steps"]),
        "--learning_rate", str(config["learningRate"]),
        "--rank", str(config["rank"]),
        "--lora_alpha", str(config["alpha"]),
        "--optimizer", "adamW",
        # Plain "constant" ignores --lr_warmup_steps.
        "--lr_scheduler", "constant_with_warmup",
        "--lr_warmup_steps", str(min(100, max(0, config["steps"] // 20))),
        "--seed", str(config["seed"]),
        "--checkpointing_steps", str(config.get("checkpointEvery", 100)),
        "--checkpoints_total_limit", str(config.get("checkpointLimit", 5)),
        "--aiv_pause_file", config["pauseFile"],
        "--gradient_checkpointing",
        "--cache_latents",
        "--use_8bit_adam",
        "--use_aspect_ratio_buckets",
        "--offload",
        "--lora_layers", str(config.get("loraLayers") or DEFAULT_LORA_LAYERS),
    ]
    precision = config.get("precision", DEFAULT_PRECISION)
    if precision == "fp8":
        command.append("--do_fp8_training")
    elif precision == "nf4":
        command.extend(["--bnb_quantization_config_path", str(bnb_config_path(config))])
    assistant = str(config.get("assistantLoraPath") or "").strip()
    if assistant:
        command.extend(["--aiv_assistant_lora_path", assistant])
    shift = config.get("timestepShift", DEFAULT_TIMESTEP_SHIFT)
    if shift is not None:
        command.extend(["--aiv_timestep_shift", str(float(shift))])
    resume = str(config.get("resumeFromCheckpoint") or "").strip()
    initial_lora = str(config.get("initialLoraPath") or "").strip()
    if resume:
        command.extend(["--resume_from_checkpoint", resume])
    elif initial_lora:
        command.extend(["--initial_lora_path", initial_lora])
    return command


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--trainer-script", required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    config = resolve_training_settings(json.loads(config_path.read_text(encoding="utf-8")))
    if config.get("assistantLoraPath") and not trainer_supports(args.trainer_script, "--aiv_assistant_lora_path"):
        raise RuntimeError("the installed trainer predates Turbo training; run Setup-ZImage-Trainer.bat")
    if config.get("timestepShift") is not None and not trainer_supports(args.trainer_script, "--aiv_timestep_shift"):
        # The installed trainer predates the shift patch; rerunning Setup-ZImage-Trainer.bat adds it.
        emit("warning", message="trainer lacks --aiv_timestep_shift; using the model's own shift")
        config["timestepShift"] = None
    # Persist the resolved settings so a later pause/resume reproduces this exact adapter layout.
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    if config["precision"] == "nf4":
        write_bnb_config(config)
    run_dir = Path(config["runDirectory"])
    dataset_dir = run_dir / "dataset"
    stage_dataset(dataset_dir, config["images"])
    output_dir = Path(config["outputDirectory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    emit("prepared", images=len(config["images"]), dataset=str(dataset_dir))

    command = build_training_command(config, args.trainer_script)
    environment = os.environ.copy()
    environment.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONUNBUFFERED": "1"})
    emit("training", step=0, total=config["steps"], command=command)
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=environment,
    )
    progress = re.compile(r"(?:(\d+)%[^|]*\|\s*)?(\d+)\s*/\s*(\d+)")
    saved_state = re.compile(r"Saved state to\s+(.+?)\s*$")
    tail: list[str] = []
    paused_at: Path | None = None
    announced: set[str] = set()
    assert process.stdout is not None

    def handle_line(clean: str) -> None:
        nonlocal tail, paused_at
        if clean:
            tail.append(clean)
            tail = tail[-80:]
        if clean.startswith("AIV_PAUSED:"):
            paused_at = Path(clean.split(":", 1)[1].strip())
        checkpoint_match = saved_state.search(clean)
        if checkpoint_match:
            checkpoint = Path(checkpoint_match.group(1).strip())
            key = str(checkpoint.resolve())
            if key not in announced:
                announced.add(key)
                checkpoints = available_checkpoints(output_dir, config)
                emit("checkpoint", path=str(checkpoint), checkpoints=checkpoints)
        match = progress.search(clean)
        if match and int(match.group(3)) == int(config["steps"]):
            emit("progress", step=int(match.group(2)), total=int(match.group(3)), message=clean[-240:])

    # tqdm refreshes one terminal row with carriage returns. Iterating by line would withhold those
    # updates until training finishes, so split on both CR and LF and stream each refresh to Tauri.
    buffered = ""
    while character := process.stdout.read(1):
        if character in "\r\n":
            if buffered:
                handle_line(buffered.rstrip())
                buffered = ""
        else:
            buffered += character
    if buffered:
        handle_line(buffered.rstrip())
    code = process.wait()
    if code:
        emit("failed", error="\n".join(tail[-20:]), exitCode=code)
        return code
    if paused_at is not None:
        checkpoints = available_checkpoints(output_dir, config)
        step = int(paused_at.name.split("-", 1)[1])
        emit("paused", checkpointPath=str(paused_at), step=step, total=config["steps"], checkpoints=checkpoints)
        return 0
    weights, manifest_path = finalize_lora(output_dir, config)
    emit("complete", outputPath=str(weights), manifest=str(manifest_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
