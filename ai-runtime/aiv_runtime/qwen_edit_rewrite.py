"""Rewrite a Qwen-Image-Edit-2511 instruction the way Qwen's own prompt enhancer does.

Qwen recommend rewriting edit prompts ("we strongly recommend applying prompt rewriting to improve
the stability of editing tasks"); their demo does it with Qwen-VL-Max and `EDIT_SYSTEM_PROMPT` from
QwenLM/Qwen-Image `src/examples/tools/prompt_utils.py` (Apache-2.0). This runs the same rules on the
Qwen2.5-VL-7B that 2511 already ships as its text encoder, locally, at 4-bit. Two changes from the
official demo, which only shows the rewriter the first image:

* every edit input is shown, labelled exactly as the edit pipeline labels them ("Picture 1:",
  "Picture 2:", ...), so "picture 2" in the rewrite means the same image to the editor;
* our fast path runs at CFG 1, where the negative prompt is ignored, so anything to keep or avoid
  is written into the instruction itself.

Output: JSON lines on stdout, the last one {"event": "complete", "prompt": ..., "raw": ...}.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any

# Same reason as the caption worker: transformers' threaded loader has faulted in torch_cpu.dll on
# this machine while loading large models. Must be set before transformers is imported.
os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")

from PIL import Image, ImageOps

MAX_NEW_TOKENS = 320
# Each picture is shown at up to about a megapixel (~1300 vision tokens for Qwen2.5-VL): enough
# to read small detail, small enough that three pictures stay cheap.
VIEW_PIXELS = 1024 * 1024
MAX_PICTURES = 3

EDIT_SYSTEM_PROMPT = """# Edit Prompt Enhancer
You are a professional edit prompt enhancer for the Qwen-Image-Edit image editing model. Your task is to generate a direct and specific edit prompt based on the user-provided instruction and the input images.
The images are labelled Picture 1, Picture 2, Picture 3 in the order given. Picture 1 is always the image being edited; any further pictures are references.
Please strictly follow the enhancing rules below:
## 1. General Principles
- Keep the enhanced prompt **direct and specific**, as one line.
- If the instruction is contradictory, vague, or unachievable, prioritize reasonable inference and correction, and supplement details when necessary.
- Keep the core intention of the original instruction unchanged, only enhancing its clarity, rationality, and visual feasibility.
- All added objects or modifications must align with the logic and style of the edited input image's overall scene.
- Look at the pictures: name things by what they actually are and where they are in the picture (e.g. "the wooden chair in the center", not "the object").
- The edit runs without a negative prompt, so anything that must stay the same is stated in the prompt itself, e.g. "keep the lighting, background and camera angle unchanged".
## 2. Task-Type Handling Rules
### 1. Add, Delete, Replace Tasks
- If the instruction is clear (already includes task type, target entity, position, quantity, attributes), preserve the original intent and only refine the grammar.
- If the description is vague, supplement with minimal but sufficient details (category, color, size, orientation, position, etc.). For example:
    > Original: "Add an animal"
    > Rewritten: "Add a light-gray cat in the bottom-right corner, sitting and facing the camera"
- For deletions, say what fills the space afterwards when it is not obvious (e.g. "restore the chair's seat and backrest where they were covered").
- Remove meaningless instructions: e.g., "Add 0 objects" should be ignored or flagged as invalid.
- For replacement tasks, specify "Replace Y with X" and briefly describe the key visual features of X.
### 2. Text Editing Tasks
- All text content must be enclosed in English double quotes " ". Keep the original language of the text, and keep the capitalization.
- Both adding new text and replacing existing text are text replacement tasks, For example:
    - Replace "xx" to "yy"
    - Replace the mask / bounding box to "yy"
    - Replace the visual object to "yy"
- Specify text position, color, and layout only if user has required.
- If font is specified, keep the original language of the font.
### 3. Human (ID) Editing Tasks
- Emphasize maintaining the person's core visual consistency (gender, age, hairstyle, expression, outfit, etc.).
- Make the smallest changes. If changes to background, action, expression, camera shot, or ambient lighting are required, list each modification individually.
- If modifying appearance (e.g., clothes, hairstyle), ensure the new element is consistent with the original style.
- **For expression changes / beauty / make up changes, they must be natural and subtle, never exaggerated.**
- Example:
    > Original: "Change the person's hat"
    > Rewritten: "Replace the man's hat with a dark brown beret; keep smile, short hair, and gray jacket unchanged"
### 4. Style Conversion or Enhancement Tasks
- If a style is specified, describe it concisely using key visual features. For example:
    > Original: "Disco style"
    > Rewritten: "1970s disco style: flashing lights, disco ball, mirrored walls, colorful tones"
- For style reference, analyze the reference picture and extract key characteristics (color, composition, texture, lighting, artistic style, etc.), integrating them into the instruction.
- **Colorization tasks (including old photo restoration) must use the fixed template:** "Restore and colorize the photo."
- Clearly specify the object to be modified. For example:
    > Original: Modify the subject in Picture 1 to match the style of Picture 2.
    > Rewritten: Change the girl in Picture 1 to the ink-wash style of Picture 2 — rendered in black-and-white watercolor with soft color transitions.
- If there are other changes, place the style description at the end.
### 5. Material Replacement and Logo Tasks
- Clearly specify the object and the new material, e.g. "Change the material of the apple to papercut style"; for text, "Change the material of text "xxxx" to ...".
- Material replacement and logo migration should preserve the original shape and structure as much as possible.
### 6. Content Filling Tasks
- For inpainting tasks, always use the fixed template: "Perform inpainting on this image. The original caption is: ".
- For outpainting tasks, always use the fixed template: "Extend the image beyond its boundaries using outpainting. The original caption is: ".
### 7. Multi-Image Tasks
- Rewritten prompts must clearly point out which picture's element is being used or modified, with a short description of it. For example:
    > Original: "Replace the subject of picture 1 with the subject of picture 2"
    > Rewritten: "Replace the girl of picture 1 with the boy of picture 2, keeping picture 1's background unchanged"
- For stylization tasks, describe the reference picture's style in the rewritten prompt, while preserving the visual content of the source picture.
## 3. Rationale and Logic Checks
- Resolve contradictory instructions: e.g., "Remove all trees but keep all trees" should be logically corrected.
- Add missing key information: e.g., if position is unspecified, choose a reasonable area based on composition (near subject, empty space, center/edge, etc.).
## 4. Output Format
- Write the rewritten prompt in the language of the user's instruction (English or Chinese).
- Return only JSON, with no other text: {"Rewritten": "..."}"""


def emit(event: str, **payload: Any) -> None:
    print(json.dumps({"event": event, **payload}, ensure_ascii=False), flush=True)


def fit_picture(image: Image.Image, pixels: int = VIEW_PIXELS) -> Image.Image:
    """RGB, upright, and no larger than `pixels` (aspect kept; never enlarged)."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    area = image.width * image.height
    if area > pixels:
        scale = (pixels / area) ** 0.5
        image = image.resize((max(28, round(image.width * scale)), max(28, round(image.height * scale))), Image.Resampling.LANCZOS)
    return image


def build_messages(prompt: str, pictures: int) -> list[dict[str, Any]]:
    """System rules, then "Picture n: " + image for each input, then the instruction.

    The labels are written out rather than left to the chat template's `add_vision_id`, and read
    exactly as the edit pipeline's `"Picture {}: <|vision_start|>..."` does.
    """
    content: list[dict[str, Any]] = []
    for index in range(1, pictures + 1):
        content.append({"type": "text", "text": f"Picture {index}: "})
        content.append({"type": "image"})
    content.append({"type": "text", "text": f"User Input: {prompt.strip()}\n\nRewritten Prompt:"})
    return [
        {"role": "system", "content": [{"type": "text", "text": EDIT_SYSTEM_PROMPT}]},
        {"role": "user", "content": content},
    ]


def _one_line(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\n", " ")).strip()


def parse_rewritten(text: str) -> str:
    """The rewrite from the model's answer: JSON (fenced or not), a truncated JSON, or plain text."""
    cleaned = text.strip().replace("```json", "").replace("```", "").strip()
    match = re.search(r"\{.*\}", cleaned, re.S)
    if match:
        try:
            value = json.loads(match.group(0))
            if isinstance(value, dict):
                for key in ("Rewritten", "rewritten", "prompt"):
                    if isinstance(value.get(key), str) and value[key].strip():
                        return _one_line(value[key])
        except json.JSONDecodeError:
            pass
    # A truncated or loosely quoted answer: take what follows the key.
    loose = re.search(r'"?Rewritten"?\s*:\s*"(.*)', cleaned, re.S | re.I)
    if loose:
        body = loose.group(1)
        end = re.search(r'"\s*}?\s*$', body)
        body = body[: end.start()] if end else body
        body = body.replace('\\"', '"')
        if body.strip():
            return _one_line(body)
    if cleaned.startswith("{"):
        return ""
    return _one_line(cleaned.strip('"'))


def rewrite(model, processor, prompt: str, pictures: list[Image.Image]) -> str:
    import torch

    messages = build_messages(prompt, len(pictures))
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=pictures or None, padding=True, return_tensors="pt")
    inputs = {key: value.to(model.device) if hasattr(value, "to") else value for key, value in inputs.items()}
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    prompt_length = inputs["input_ids"].shape[-1]
    return processor.batch_decode(generated[:, prompt_length:], skip_special_tokens=True)[0].strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rewrite a Qwen-Image-Edit instruction")
    parser.add_argument("--model-path", required=True, help="Qwen-Image-Edit-2511 text_encoder (Qwen2.5-VL-7B)")
    parser.add_argument("--processor-path", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--image-path", action="append", default=[], help="Picture 1 (the edited image), then references")
    args = parser.parse_args(argv)
    if not args.prompt.strip():
        raise ValueError("the edit instruction is empty")
    if len(args.image_path) > MAX_PICTURES:
        raise ValueError(f"at most {MAX_PICTURES} pictures")
    pictures = []
    for path in args.image_path:
        with Image.open(path) as opened:
            opened.draft("RGB", (2048, 2048))
            pictures.append(fit_picture(opened))

    from aiv_runtime.zimage_caption_worker import load_model

    emit("loading", pictures=len(pictures))
    model, processor = load_model("qwen", args.model_path, args.processor_path)
    emit("rewriting")
    raw = rewrite(model, processor, args.prompt, pictures)
    rewritten = parse_rewritten(raw)
    if not rewritten:
        print(f"[rewrite] could not read the model's answer: {raw!r}", file=sys.stderr, flush=True)
        raise RuntimeError("the rewriter returned no usable prompt; try again or reword the instruction")
    emit("complete", prompt=rewritten, raw=raw)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
