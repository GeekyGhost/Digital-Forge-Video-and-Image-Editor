"""Dataset preparation helpers shared by the Z-Image caption and training workers."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps, ImageStat

PROFILES = {"subject", "character", "style", "product", "concept"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True, slots=True)
class ImageInspection:
    path: str
    width: int
    height: int
    sha256: str
    perceptual_hash: str
    quality_flags: tuple[str, ...]


def safe_name(value: str) -> str:
    words = re.findall(r"[a-z0-9]+", value.lower())
    return "-".join(words[:8])[:64] or "zimage-lora"


def lora_filename(value: str) -> str:
    """Return a readable, filesystem-safe final adapter filename.

    Training checkpoints retain Diffusers' conventional filename because the trainer needs it for
    resume support. The final user-facing adapter gets an identifiable name instead.
    """
    words = re.findall(r"[A-Za-z0-9]+", value)
    stem = "".join(word[:1].upper() + word[1:] for word in words)[:80] or "ZImage"
    return f"{stem}-Z-LoRA.safetensors"


def resolve_lora_weights(path: str | Path) -> Path:
    """Resolve either a legacy adapter file or an AIV LoRA folder to its final weights."""
    selected = Path(path)
    if selected.is_file():
        return selected
    if not selected.is_dir():
        raise FileNotFoundError(f"Z-Image LoRA not found: {selected}")

    manifest_path = selected / "aiv-lora.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid Z-Image LoRA manifest: {manifest_path}") from exc
        for key in ("weightsFile", "weights"):
            value = manifest.get(key)
            if not isinstance(value, str) or not value.strip():
                continue
            candidate = Path(value)
            if not candidate.is_absolute():
                candidate = selected / candidate
            if candidate.is_file():
                return candidate

    preferred = sorted(selected.glob("*-Z-LoRA.safetensors"))
    legacy = selected / "pytorch_lora_weights.safetensors"
    if preferred:
        return preferred[0]
    if legacy.is_file():
        return legacy
    candidates = sorted(selected.glob("*.safetensors"))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"Z-Image LoRA folder contains no safetensors weights: {selected}")
    raise ValueError(f"Z-Image LoRA folder is ambiguous; select a folder with an aiv-lora.json manifest: {selected}")


_EXIF_ORIENTATION = 0x0112


def inspect_image(path: str) -> ImageInspection:
    source = Path(path)
    if not source.is_file() or source.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError(f"unsupported training image: {source}")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    with Image.open(source) as opened:
        width, height = opened.size
        # Report the size the trainer will actually see. The Diffusers dataset applies
        # exif_transpose, so a portrait phone photo stored as landscape must be reported as portrait.
        if opened.getexif().get(_EXIF_ORIENTATION) in (5, 6, 7, 8):
            width, height = height, width
        opened.draft("RGB", (2048, 2048))
        image = ImageOps.exif_transpose(opened).convert("RGB")
        image.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
        sample = image.resize((9, 8), Image.Resampling.LANCZOS).convert("L")
        values = list(sample.get_flattened_data())
        differences = [
            values[row * 9 + column] >= values[row * 9 + column + 1]
            for row in range(8)
            for column in range(8)
        ]
        colors = ImageStat.Stat(image.resize((32, 32), Image.Resampling.BILINEAR)).mean
        color_bucket = "".join(f"{min(15, int(channel) // 16):x}" for channel in colors)
        phash = f"{sum((1 << index) for index, value in enumerate(differences) if value):016x}{color_bucket}"
        luminance = ImageStat.Stat(image.resize((64, 64), Image.Resampling.BILINEAR).convert("L"))
        mean = luminance.mean[0]
        deviation = luminance.stddev[0]
    flags: list[str] = []
    if min(width, height) < 512:
        flags.append("low-resolution")
    if max(width, height) / max(1, min(width, height)) > 3.0:
        flags.append("extreme-aspect-ratio")
    if mean < 25:
        flags.append("very-dark")
    if mean > 235:
        flags.append("very-bright")
    if deviation < 12:
        flags.append("low-contrast")
    return ImageInspection(str(source), width, height, digest, phash, tuple(flags))


SUBJECT_PLACEHOLDER = "<subject>"

# Words that describe a designed character's fixed appearance. If the captioner writes any of these
# into the prose caption for the character profile, ordinary words would explain those pixels and
# the trigger would learn less of the design, so the caption is rebuilt from scene-only fields.
_CHARACTER_IDENTITY_WORDS = frozenset({
    "armor", "armour", "armored", "armoured", "helmet", "visor", "goggles", "mask", "costume",
    "uniform", "suit", "beard", "bearded", "mustache", "moustache", "hair", "haired", "hairstyle",
    "tattoo", "tattoos", "cape", "gauntlet", "gauntlets", "weapon", "weapons", "sword", "swords",
    "blade", "blades", "dagger", "daggers", "gun", "guns", "rifle", "pistol", "shield",
})
_STOPWORDS = frozenset({
    "a", "an", "and", "the", "with", "of", "in", "on", "at", "to", "for", "his", "her", "their",
    "its", "wearing", "wears", "holding", "holds", "none", "visible", "some", "large", "small",
    "one", "two", "three", "four", "pair", "both", "each",
    # Colours are dropped from the field-derived block list: they appear in lighting and backgrounds
    # far more often than as identity words, and the noun they modify ("armor") is already caught.
    "black", "white", "red", "blue", "green", "yellow", "orange", "purple", "pink", "brown", "grey",
    "gray", "gold", "golden", "silver", "dark", "light", "bright", "glowing", "accents", "accent",
})


def trigger_phrase(profile: str, trigger: str, class_name: str) -> str:
    """The words a prompt must contain to invoke the LoRA, matching how captions were written."""
    token = _clean_fragment(trigger)
    if profile == "style":
        return f"{token} style"
    return f"{token} {_clean_fragment(class_name)}".strip()


def caption_instruction(profile: str, trigger: str, class_name: str) -> str:
    if profile not in PROFILES:
        raise ValueError(f"unknown caption profile: {profile}")
    common = (
        "Return only compact JSON with keys medium, shot_type, camera_angle, pose, expression, clothing, "
        "held_items, background, lighting, tags, quality_flags, confidence, caption. "
        "medium is the image medium such as photograph, 3D render, digital painting, anime illustration, "
        "or pencil sketch. Use 3-8 concrete visual tags. quality_flags must list only actual problems "
        "such as blur, artifacts, obstruction, poor exposure, or low detail; return an empty list when "
        "none exist. Describe visible facts only. Do not guess a person's identity, ethnicity, exact age, "
        "attractiveness, or private attributes. Do not include the training trigger token. "
        "caption is one or two plain natural-language sentences of 20 to 60 words, written like an image "
        f"generation prompt. In caption, refer to the main subject only with the literal placeholder "
        f"{SUBJECT_PLACEHOLDER}, exactly once, for example: 'A medium shot of {SUBJECT_PLACEHOLDER} "
        "walking through a desert canyon at midday, warm natural sunlight, 3D render.'"
    )
    specific = {
        "subject": (
            "In caption, describe changeable details such as clothing, pose, framing, background, lighting, "
            "and medium; do not describe the subject's face, body, or other permanent identity traits."
        ),
        "character": (
            "This is a designed character whose canonical appearance must bind to the trigger. "
            "The caption must NOT mention the character's face, hair, beard, body, signature clothing or armor, "
            "signature colors, accessories, held items, or weapons. In caption describe only framing, camera "
            "angle, pose, expression, background, lighting, and medium. The clothing and held_items fields "
            "may still list what you see; they are used only to check the caption."
        ),
        "style": (
            "Describe the depicted subjects, composition, and setting completely, but never name or "
            f"characterize the artistic style or medium. For style images the {SUBJECT_PLACEHOLDER} "
            "placeholder is not required; describe the scene directly."
        ),
        "product": (
            "In caption describe product view, orientation, environment, surface, lighting, camera framing, "
            "and medium."
        ),
        "concept": (
            "In caption describe the scene and surrounding content, lighting, and medium while avoiding a "
            "name for the target concept itself."
        ),
    }[profile]
    return f"{common} {specific} The class noun is {class_name!r}; the application will insert {trigger!r}."


def _clean_fragment(value: Any) -> str:
    if isinstance(value, list):
        value = ", ".join(str(item) for item in value if str(item).strip())
    return re.sub(r"\s+", " ", str(value or "")).strip(" ,.;")


def _repair_truncated_json(candidate: str) -> dict[str, Any] | None:
    """Recover the complete leading fields of JSON that hit the generation token limit."""
    for index in range(len(candidate) - 1, 0, -1):
        if candidate[index] != ",":
            continue
        head = candidate[:index]
        for closer in ("}", "]}"):
            try:
                parsed = json.loads(head + closer)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
    return None


def parse_model_json(text: str) -> dict[str, Any]:
    """Parse the captioner's JSON answer.

    Plain prose (no JSON at all) is kept as a caption. Broken JSON is repaired when only its tail was
    cut off; otherwise an empty dict is returned so the raw JSON text never becomes a training caption.
    """
    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    candidate = fenced.group(1) if fenced else cleaned
    if "{" not in candidate:
        return {"caption": cleaned} if cleaned else {}
    candidate = candidate[candidate.index("{"):]
    unterminated_fence = candidate.rfind("```")
    if unterminated_fence > 0:
        candidate = candidate[:unterminated_fence]
    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        last_brace = candidate.rfind("}")
        if last_brace > 0:
            try:
                parsed = json.loads(candidate[: last_brace + 1])
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass
        return _repair_truncated_json(candidate) or {}


def _words(value: str) -> set[str]:
    return {word for word in re.findall(r"[a-z]+", value.casefold()) if len(word) >= 3}


def _identity_leaks(caption: str, raw: dict[str, Any]) -> list[str]:
    reported = _words(_clean_fragment(raw.get("clothing"))) | _words(_clean_fragment(raw.get("held_items")))
    blocked = (reported - _STOPWORDS) | _CHARACTER_IDENTITY_WORDS
    return sorted(_words(caption) & blocked)


def _sentence_case(value: str) -> str:
    return value[:1].upper() + value[1:] if value else value


def _scene_sentence(
    subject: str, raw: dict[str, Any], *, include_clothing: bool, blocked: frozenset[str] = frozenset()
) -> str:
    """Natural-language caption assembled from structured fields (fallback path).

    Comma-separated fragments that contain a blocked word are dropped, so a pose such as
    "standing, hands holding weapons" keeps "standing" without naming the signature weapons.
    """

    def field(key: str) -> str:
        fragments = [part.strip() for part in _clean_fragment(raw.get(key)).split(",")]
        return ", ".join(part for part in fragments if part and not (_words(part) & blocked))

    shot = field("shot_type")
    angle = field("camera_angle")
    pose = field("pose")
    expression = field("expression")
    clothing = field("clothing") if include_clothing else ""
    background = field("background")
    lighting = field("lighting")
    # Captioners often answer "medium" or "wide" for shot_type; "A medium of X" is not a phrase.
    if shot and not re.search(r"\b(shot|close-?up|view|portrait)\b", shot, re.I):
        shot = f"{shot} shot"
    lead = f"A {shot} of {subject}" if shot else f"An image of {subject}"
    parts = [lead]
    if clothing:
        parts.append(f"wearing {clothing}")
    if pose:
        parts.append(pose)
    if angle:
        parts.append(angle if re.search(r"\b(angle|view|shot)\b", angle, re.I) else f"{angle} view")
    if expression:
        parts.append(f"{expression} expression")
    sentence = ", ".join(parts) + "."
    if background:
        sentence += f" Setting: {background}."
    if lighting:
        sentence += f" Lighting: {lighting}."
    return sentence


def _insert_subject(caption: str, subject: str, noun: str) -> str:
    pattern = re.compile(re.escape(SUBJECT_PLACEHOLDER), re.IGNORECASE)
    if not pattern.search(caption):
        return f"{subject}, {caption[:1].lower() + caption[1:]}" if caption else subject
    first = pattern.sub(subject, caption, count=1)
    return pattern.sub(f"the {noun}", first)


def build_training_caption(profile: str, trigger: str, class_name: str, raw: dict[str, Any]) -> tuple[str, list[str]]:
    """Return the training caption and review notes describing any automatic correction."""
    if profile not in PROFILES:
        raise ValueError(f"unknown caption profile: {profile}")
    token = _clean_fragment(trigger)
    noun = _clean_fragment(class_name)
    if not token or not noun:
        raise ValueError("trigger and class name are required")
    notes: list[str] = []
    subject = f"{token} {noun}"
    # Never trust a captioner to preserve the trigger contract. Remove accidental mentions before
    # inserting exactly one app-owned subject phrase.
    strip_token = re.compile(rf"\b{re.escape(token)}\b", re.IGNORECASE)
    prose = strip_token.sub("", _clean_fragment(raw.get("caption")))
    prose = re.sub(r"\s+", " ", prose).strip(" ,;")
    medium = _clean_fragment(raw.get("medium"))

    if profile == "style":
        base = re.sub(re.escape(SUBJECT_PLACEHOLDER), "the main subject", prose, flags=re.IGNORECASE)
        if not base:
            base = strip_token.sub("", _clean_fragment(raw.get("description"))).strip(" ,;")
        if not base:
            base = _scene_sentence(f"a {noun}", raw, include_clothing=True).rstrip(".")
        return f"{base.rstrip('.')}, in {token} style", notes

    if profile == "character":
        leaks = _identity_leaks(prose, raw) if prose else []
        if prose and not leaks:
            caption = _insert_subject(prose, subject, noun)
        else:
            if leaks:
                notes.append("caption-rebuilt: identity words " + ", ".join(leaks))
            elif raw:
                notes.append("caption-rebuilt: no prose caption")
            reported = _words(_clean_fragment(raw.get("clothing"))) | _words(_clean_fragment(raw.get("held_items")))
            blocked = frozenset((reported - _STOPWORDS) | _CHARACTER_IDENTITY_WORDS)
            caption = _scene_sentence(subject, raw, include_clothing=False, blocked=blocked)
    elif prose:
        caption = _insert_subject(prose, subject, noun)
    else:
        legacy = strip_token.sub("", _clean_fragment(raw.get("description"))).strip(" ,;")
        caption = f"{subject}, {legacy}" if legacy else _scene_sentence(subject, raw, include_clothing=True)

    if medium and medium.casefold() not in caption.casefold():
        caption = f"{caption.rstrip('.')}. {_sentence_case(medium)}."
    caption = re.sub(r"\s+", " ", caption).strip()
    if caption.casefold().count(token.casefold()) != 1:
        caption = f"{subject}, " + strip_token.sub("", caption).strip(" ,;")
    if caption and caption[-1] not in ".!?":
        caption += "."
    return caption, notes


def training_caption(profile: str, trigger: str, class_name: str, raw: dict[str, Any]) -> str:
    return build_training_caption(profile, trigger, class_name, raw)[0]


def stage_dataset(directory: Path, items: list[dict[str, Any]]) -> Path:
    if not items:
        raise ValueError("at least one captioned image is required")
    directory.mkdir(parents=True, exist_ok=True)
    metadata: list[str] = []
    seen_hashes: set[str] = set()
    for index, item in enumerate(items, 1):
        inspection = inspect_image(str(item.get("imagePath", "")))
        if inspection.sha256 in seen_hashes:
            continue
        seen_hashes.add(inspection.sha256)
        caption = _clean_fragment(item.get("caption"))
        if not caption:
            raise ValueError(f"image {inspection.path} has no approved caption")
        suffix = Path(inspection.path).suffix.lower()
        filename = f"image-{index:04d}{suffix}"
        shutil.copy2(inspection.path, directory / filename)
        metadata.append(json.dumps({"file_name": filename, "text": caption}, ensure_ascii=False))
    if len(metadata) < 3:
        raise ValueError("at least three unique captioned images are required")
    output = directory / "metadata.jsonl"
    output.write_text("\n".join(metadata) + "\n", encoding="utf-8")
    return output
