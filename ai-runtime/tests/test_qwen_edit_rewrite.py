"""The Qwen edit prompt rewriter: message layout, answer parsing and the worker's contract.

The model itself is not loaded here: `load_model` and `rewrite` are replaced.
"""

import json
import sys
import types

import pytest
from PIL import Image

from aiv_runtime import qwen_edit_rewrite as rw

# Qwen-Image-Edit-2511's processor/chat_template.jinja, verbatim.
CHAT_TEMPLATE = (
    "{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}"
    "{% if loop.first and message['role'] != 'system' %}<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n{% endif %}"
    "<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n"
    "{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}"
    "{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}"
    "<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}"
    "{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}"
    "<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n"
    "{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


def render(messages):
    jinja2 = pytest.importorskip("jinja2")
    return jinja2.Environment().from_string(CHAT_TEMPLATE).render(messages=messages, add_generation_prompt=True)


def test_pictures_are_labelled_as_the_edit_pipeline_labels_them():
    text = render(rw.build_messages("  put the hat from picture 2 on her ", 3))
    assert text.startswith("<|im_start|>system\n# Edit Prompt Enhancer")
    user = text.split("<|im_start|>user\n", 1)[1]
    assert user.startswith(
        "Picture 1: <|vision_start|><|image_pad|><|vision_end|>"
        "Picture 2: <|vision_start|><|image_pad|><|vision_end|>"
        "Picture 3: <|vision_start|><|image_pad|><|vision_end|>"
        "User Input: put the hat from picture 2 on her\n\nRewritten Prompt:<|im_end|>"
    )
    assert text.endswith("<|im_start|>assistant\n")


def test_no_pictures_is_a_text_only_rewrite():
    text = render(rw.build_messages("make it night", 0))
    assert "<|image_pad|>" not in text and "Picture 1" not in text.split("<|im_start|>user")[1]


def test_the_rules_carry_the_official_templates_and_our_two_changes():
    rules = rw.EDIT_SYSTEM_PROMPT
    for fixed in ("Restore and colorize the photo.", "Perform inpainting on this image. The original caption is: ",
                  "Extend the image beyond its boundaries using outpainting. The original caption is: ",
                  '{"Rewritten": "..."}', "Picture 1 is always the image being edited", "without a negative prompt"):
        assert fixed in rules


@pytest.mark.parametrize("answer, expected", [
    ('{"Rewritten": "Remove the arms and legs from the wooden chair; keep the room unchanged"}',
     "Remove the arms and legs from the wooden chair; keep the room unchanged"),
    ('```json\n{"Rewritten": "Replace \\"SALE\\" to \\"OPEN\\" on the sign"}\n```', 'Replace "SALE" to "OPEN" on the sign'),
    ('Sure! {"Rewritten": "Add a cat\\nin the corner"}', "Add a cat in the corner"),
    ('{"Rewritten": "Add a light-gray cat in the bottom-right corner, sitting', "Add a light-gray cat in the bottom-right corner, sitting"),
    ("Add a light-gray cat in the corner", "Add a light-gray cat in the corner"),
    ('{"other": 1}', ""),
])
def test_answers_are_parsed_to_one_line(answer, expected):
    assert rw.parse_rewritten(answer) == expected


def test_pictures_are_upright_rgb_and_at_most_a_megapixel():
    big = Image.new("RGBA", (4000, 2000), (10, 20, 30, 128))
    out = rw.fit_picture(big)
    assert out.mode == "RGB" and out.width * out.height <= rw.VIEW_PIXELS
    assert abs(out.width / out.height - 2) < 0.01
    small = Image.new("L", (300, 200))
    assert rw.fit_picture(small).size == (300, 200)


def test_the_worker_reports_the_rewrite(monkeypatch, tmp_path, capsys):
    paths = []
    for index, size in enumerate([(640, 480), (320, 320)]):
        path = tmp_path / f"p{index}.png"
        Image.new("RGB", size, (index * 80, 40, 40)).save(path)
        paths.append(str(path))
    seen = {}
    fake = types.ModuleType("aiv_runtime.zimage_caption_worker")
    fake.load_model = lambda backend, model, processor: seen.setdefault("load", (backend, model, processor)) and ("model", "processor")
    monkeypatch.setitem(sys.modules, "aiv_runtime.zimage_caption_worker", fake)

    def fake_rewrite(model, processor, prompt, pictures):
        seen["rewrite"] = (prompt, [picture.size for picture in pictures])
        return '{"Rewritten": "Remove the arms and legs sticking through the chair in picture 1; keep the chair shape unchanged"}'

    monkeypatch.setattr(rw, "rewrite", fake_rewrite)
    assert rw.main(["--model-path", "te", "--processor-path", "proc", "--prompt", "fix the chair",
                    "--image-path", paths[0], "--image-path", paths[1]]) == 0
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [event["event"] for event in events] == ["loading", "rewriting", "complete"]
    assert events[-1]["prompt"].startswith("Remove the arms and legs")
    assert seen["load"] == ("qwen", "te", "proc")
    assert seen["rewrite"] == ("fix the chair", [(640, 480), (320, 320)])


def test_the_worker_refuses_an_empty_answer_and_too_many_pictures(monkeypatch, tmp_path):
    fake = types.ModuleType("aiv_runtime.zimage_caption_worker")
    fake.load_model = lambda *args: ("model", "processor")
    monkeypatch.setitem(sys.modules, "aiv_runtime.zimage_caption_worker", fake)
    monkeypatch.setattr(rw, "rewrite", lambda *args: '{"nothing": true}')
    with pytest.raises(RuntimeError, match="no usable prompt"):
        rw.main(["--model-path", "a", "--processor-path", "b", "--prompt", "x"])
    with pytest.raises(ValueError, match="at most 3"):
        rw.main(["--model-path", "a", "--processor-path", "b", "--prompt", "x"] + ["--image-path", "p"] * 4)
