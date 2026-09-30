from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from aiv_runtime.zimage_caption_worker import confidence_value, string_list
from aiv_runtime.zimage_train_worker import (
    DEFAULT_LORA_LAYERS,
    LEGACY_LORA_LAYERS,
    available_checkpoints,
    build_training_command,
    checkpoint_manifest,
    finalize_lora,
    resolve_training_settings,
    write_bnb_config,
)
from aiv_runtime.zimage_training import (
    build_training_caption,
    caption_instruction,
    inspect_image,
    lora_filename,
    parse_model_json,
    resolve_lora_weights,
    safe_name,
    stage_dataset,
    training_caption,
    trigger_phrase,
)


def test_trigger_is_inserted_once_for_subject() -> None:
    caption = training_caption(
        "subject", "TOK", "person",
        {"description": "A medium shot of TOK wearing a red jacket beside a window."},
    )
    assert caption.startswith("TOK person,")
    assert caption.count("TOK") == 1


def test_style_profile_leaves_style_bound_to_trigger() -> None:
    caption = training_caption(
        "style", "INK42", "image",
        {"description": "A harbor at sunset", "tags": ["wide shot", "orange sky", "wide shot"]},
    )
    # Tag lists are no longer appended: Z-Image's Qwen3 encoder is trained on prose prompts.
    assert caption == "A harbor at sunset, in INK42 style"
    assert trigger_phrase("style", "INK42", "image") == "INK42 style"


def test_prose_caption_places_trigger_where_the_subject_is() -> None:
    caption, notes = build_training_caption(
        "subject", "TOK", "woman",
        {"caption": "A close-up of <subject> smiling on a park bench, soft window light", "medium": "photograph"},
    )
    assert caption == "A close-up of TOK woman smiling on a park bench, soft window light. Photograph."
    assert notes == []


def test_character_prose_with_identity_words_is_rebuilt_from_scene_fields() -> None:
    raw = {
        "medium": "3D render", "shot_type": "medium shot", "camera_angle": "frontal",
        "pose": "standing, hands holding daggers", "expression": "neutral",
        "clothing": "black armor, red visor", "held_items": "two daggers",
        "background": "purple vortex", "lighting": "red rim light",
        "caption": "A medium shot of <subject> in black armor before a purple vortex.",
    }
    caption, notes = build_training_caption("character", "capdiciosks", "man", raw)
    assert caption == (
        "A medium shot of capdiciosks man, standing, frontal view, neutral expression. "
        "Setting: purple vortex. Lighting: red rim light. 3D render."
    )
    assert notes and notes[0].startswith("caption-rebuilt")
    clean = dict(raw, caption="A medium shot of <subject> standing before a purple vortex, red rim light.")
    caption, notes = build_training_caption("character", "capdiciosks", "man", clean)
    assert caption == "A medium shot of capdiciosks man standing before a purple vortex, red rim light. 3D render."
    assert notes == []


def test_truncated_or_broken_json_never_becomes_the_caption() -> None:
    truncated = '```json\n{"medium": "photograph", "caption": "A close-up of <subject> in a park", "tags": ["pa'
    assert parse_model_json(truncated) == {"medium": "photograph", "caption": "A close-up of <subject> in a park"}
    assert parse_model_json('{"caption": ') == {}
    assert parse_model_json("A plain prose answer.") == {"caption": "A plain prose answer."}
    assert string_list("single") == ["single"]
    assert string_list([1, " ", "b"]) == ["1", "b"]


def test_character_profile_binds_signature_design_to_trigger() -> None:
    caption = training_caption(
        "character", "diciosks", "person",
        {
            "description": "A bearded person in signature black armor and a red visor holds a plasma sword.",
            "shot_type": "medium shot",
            "camera_angle": "low angle",
            "pose": "standing",
            "expression": "determined",
            "clothing": "signature black armor and red visor",
            "background": "stormy beach",
            "lighting": "blue rim light",
            "tags": ["black armor", "red visor", "plasma sword"],
        },
    )
    assert caption == (
        "A medium shot of diciosks person, standing, low angle, determined expression. "
        "Setting: stormy beach. Lighting: blue rim light."
    )
    assert "armor" not in caption
    assert "visor" not in caption


def test_json_fences_and_safe_names() -> None:
    assert parse_model_json('```json\n{"description":"A portrait"}\n```')["description"] == "A portrait"
    assert safe_name("../My Hero / V2") == "my-hero-v2"
    assert lora_filename("Captain Dicio") == "CaptainDicio-Z-LoRA.safetensors"
    assert "identity" in caption_instruction("subject", "TOK", "person")
    assert "<subject>" in caption_instruction("character", "TOK", "person")
    assert confidence_value("high") == 0.9
    assert confidence_value("unknown") == 0.75


def test_lora_folder_resolves_manifest_and_legacy_weights(tmp_path: Path) -> None:
    named = tmp_path / "CaptainDicio-Z-LoRA.safetensors"
    named.write_bytes(b"weights")
    (tmp_path / "aiv-lora.json").write_text(
        json.dumps({"weightsFile": named.name, "weights": "missing.safetensors"}), encoding="utf-8"
    )
    assert resolve_lora_weights(tmp_path) == named
    assert resolve_lora_weights(named) == named


def test_final_adapter_uses_human_readable_name_and_manifest(tmp_path: Path) -> None:
    (tmp_path / "pytorch_lora_weights.safetensors").write_bytes(b"weights")
    (tmp_path / "README.md").write_text("weight_name='pytorch_lora_weights.safetensors'", encoding="utf-8")
    config = {
        "displayName": "Captain Dicio", "name": "captain-dicio", "modelPath": "models/Z-Image",
        "profile": "character", "trigger": "captaindiciosks", "className": "person",
        "resolution": 1024, "steps": 500, "rank": 16, "alpha": 16,
        "learningRate": 0.0001, "seed": 42, "images": [{}, {}, {}],
    }
    weights, manifest_path = finalize_lora(tmp_path, config)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert weights.name == "CaptainDicio-Z-LoRA.safetensors"
    assert manifest["weightsFile"] == weights.name
    assert manifest["triggerPhrase"] == "captaindiciosks person"
    assert "CaptainDicio-Z-LoRA.safetensors" in (tmp_path / "README.md").read_text(encoding="utf-8")


def test_checkpoint_metadata_and_resume_command(tmp_path: Path) -> None:
    output = tmp_path / "output"
    checkpoint = output / "checkpoint-125"
    checkpoint.mkdir(parents=True)
    (checkpoint / "pytorch_lora_weights.safetensors").write_bytes(b"weights")
    config = {
        "displayName": "Captain Dicio", "name": "captain-dicio", "modelPath": "models/Z-Image",
        "profile": "character", "trigger": "captaindiciosks", "className": "person",
        "resolution": 1024, "steps": 500, "rank": 16, "alpha": 16,
        "learningRate": 0.0001, "seed": 42, "images": [{}, {}, {}],
        "runDirectory": str(tmp_path / "run"), "outputDirectory": str(output),
        "pauseFile": str(tmp_path / "run" / "pause.request"), "checkpointEvery": 125,
        "checkpointLimit": 5, "resumeFromCheckpoint": str(checkpoint),
        "initialLoraPath": str(tmp_path / "old.safetensors"),
    }
    manifest_path = checkpoint_manifest(checkpoint, config, 125)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["kind"] == "training-checkpoint"
    assert manifest["triggerPhrase"] == "captaindiciosks person"
    assert available_checkpoints(output, config)[0]["step"] == 125
    command = build_training_command(config, "trainer.py")
    assert command[command.index("--resume_from_checkpoint") + 1] == str(checkpoint)
    assert "--initial_lora_path" not in command


def test_resume_of_a_pre_upgrade_job_keeps_its_original_adapter_layout(tmp_path: Path) -> None:
    config = resolve_training_settings({
        "modelPath": "m", "runDirectory": str(tmp_path), "outputDirectory": str(tmp_path / "o"),
        "trigger": "TOK", "className": "person", "profile": "character", "resolution": 1024, "steps": 600,
        "rank": 16, "alpha": 16, "learningRate": 0.0001, "seed": 1, "pauseFile": "p",
        "resumeFromCheckpoint": str(tmp_path / "checkpoint-100"),
    })
    command = build_training_command(config, "trainer.py")
    assert command[command.index("--lora_layers") + 1] == LEGACY_LORA_LAYERS
    assert "--do_fp8_training" in command
    assert "--aiv_timestep_shift" not in command


def test_new_runs_use_ffn_targets_nf4_and_shift(tmp_path: Path) -> None:
    config = resolve_training_settings({
        "modelPath": "m", "runDirectory": str(tmp_path), "outputDirectory": str(tmp_path / "o"),
        "trigger": "TOK", "className": "person", "profile": "style", "resolution": 1024, "steps": 500,
        "rank": 16, "alpha": 16, "learningRate": 0.0001, "seed": 1, "pauseFile": "p",
    })
    command = build_training_command(config, "trainer.py")
    assert command[command.index("--lora_layers") + 1] == DEFAULT_LORA_LAYERS
    assert command[command.index("--aiv_timestep_shift") + 1] == "3.0"
    assert command[command.index("--lr_scheduler") + 1] == "constant_with_warmup"
    assert command[command.index("--instance_prompt") + 1] == "TOK style"
    assert "--do_fp8_training" not in command and "--guidance_scale" not in command
    bnb = Path(command[command.index("--bnb_quantization_config_path") + 1])
    assert write_bnb_config(config) == bnb
    assert json.loads(bnb.read_text(encoding="utf-8"))["bnb_4bit_quant_type"] == "nf4"


def test_new_phase_uses_initial_lora_without_resume(tmp_path: Path) -> None:
    config = {
        "modelPath": "models/Z-Image", "runDirectory": str(tmp_path / "run"),
        "outputDirectory": str(tmp_path / "output"), "trigger": "TOK", "className": "person",
        "resolution": 768, "steps": 300, "rank": 16, "alpha": 16, "learningRate": 0.0001,
        "seed": 1, "pauseFile": str(tmp_path / "pause.request"),
        "initialLoraPath": str(tmp_path / "source.safetensors"),
    }
    command = build_training_command(config, "trainer.py")
    assert command[command.index("--initial_lora_path") + 1] == config["initialLoraPath"]
    assert "--resume_from_checkpoint" not in command


def test_inspection_and_dataset_staging(tmp_path: Path) -> None:
    items = []
    for index, shade in enumerate((40, 100, 180), 1):
        image_path = tmp_path / f"source-{index}.png"
        Image.new("RGB", (640, 768), (shade, shade, shade)).save(image_path)
        inspected = inspect_image(str(image_path))
        assert inspected.width == 640
        items.append({"imagePath": str(image_path), "caption": f"TOK person, pose {index}"})
    metadata = stage_dataset(tmp_path / "dataset", items)
    rows = [json.loads(line) for line in metadata.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 3
    assert rows[0]["text"].startswith("TOK person")


def test_inspection_reports_exif_rotated_size(tmp_path: Path) -> None:
    image_path = tmp_path / "phone.jpg"
    exif = Image.Exif()
    exif[0x0112] = 6  # stored landscape, displayed portrait
    Image.new("RGB", (800, 600), (90, 120, 150)).save(image_path, exif=exif)
    inspected = inspect_image(str(image_path))
    assert (inspected.width, inspected.height) == (600, 800)


def test_bare_shot_type_reads_as_a_shot() -> None:
    caption, _ = build_training_caption(
        "character", "capdiciosks", "man",
        {"shot_type": "medium", "pose": "kneeling", "held_items": "two swords", "background": "jungle"},
    )
    assert caption == "A medium shot of capdiciosks man, kneeling. Setting: jungle."


def test_turbo_training_uses_the_frozen_adapter_and_labels_the_lora(tmp_path: Path) -> None:
    config = resolve_training_settings({
        "modelPath": "models/Z-Image-Turbo", "runDirectory": str(tmp_path), "outputDirectory": str(tmp_path),
        "trigger": "TOK", "className": "man", "profile": "character", "resolution": 1024, "steps": 500,
        "rank": 16, "alpha": 16, "learningRate": 0.0001, "seed": 1, "pauseFile": "p", "images": [{}, {}, {}],
        "trainingModel": "turbo", "assistantLoraPath": "adapter.safetensors", "name": "tok",
    })
    command = build_training_command(config, "trainer.py")
    assert command[command.index("--aiv_assistant_lora_path") + 1] == "adapter.safetensors"
    (tmp_path / "pytorch_lora_weights.safetensors").write_bytes(b"w")
    _, manifest_path = finalize_lora(tmp_path, config)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["baseModel"] == "Z-Image-Turbo"
    assert manifest["inference"] == {"turbo": {"steps": 9, "guidanceScale": 0.0, "strength": 1.0}}
    base = resolve_training_settings({"runDirectory": str(tmp_path)})
    assert "--aiv_assistant_lora_path" not in build_training_command(
        dict(base, modelPath="m", outputDirectory="o", trigger="T", className="c", resolution=1024, steps=100,
             rank=16, alpha=16, learningRate=0.0001, seed=1, pauseFile="p"), "trainer.py")
    try:
        resolve_training_settings({"trainingModel": "turbo", "runDirectory": str(tmp_path)})
    except ValueError as error:
        assert "training adapter" in str(error)
    else:
        raise AssertionError("turbo without an adapter must be rejected")
