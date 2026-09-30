"""Cross-checks our presentation label order against the one Diffusers actually emits.

The numbering lives entirely in TypeScript (`src/features/generation/h3Ref.ts::allocateH3Labels`)
and is well covered there -- but only against itself. Nothing asserted that it agrees with
`MiniMaxH3Ref2VATextEncoderStep._build_presentation`, and getting the order wrong points every
subsequent label at the wrong reference with no error at all: the model is simply told that
picture 2 is picture 1.

This test calls the real builder with a stub tokenizer, so it needs no weights, no GPU and no
conditioner -- only the H3 environment's diffusers. It skips elsewhere, including in the
application `.venv`, which carries a diffusers without the minimax_h3 modular pipeline.

The rule, read from the builder on 2026-09-08: independent per-modality counters, and a
sound-bearing reference emits its `<Audio j>` label BEFORE its own `<Picture i>` / `<Video k>`.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest


@dataclass
class StubReference:
    kind: str
    has_audio: bool


class StubTokenizer:
    """Records the text it is asked to tokenize; one token per call is enough to keep order."""

    def __init__(self) -> None:
        self.texts: list[str] = []

    def __call__(self, value: str, add_special_tokens: bool = True) -> dict:
        self.texts.append(value)
        return {"input_ids": [len(self.texts)]}

    def convert_tokens_to_ids(self, token: str) -> int:
        return {"<|vision_start|>": -1, "<|image_pad|>": -2, "<|vision_end|>": -3, "<|video_pad|>": -4}[token]


def _build_presentation(*args, **kwargs):
    encoders = pytest.importorskip(
        "diffusers.modular_pipelines.minimax_h3.encoders",
        reason="the minimax_h3 modular pipeline is only installed in the H3 environment",
    )
    return encoders.MiniMaxH3Ref2VATextEncoderStep._build_presentation(*args, **kwargs)


def test_a_sound_bearing_video_is_labelled_audio_before_video():
    tokenizer = StubTokenizer()
    references = [
        StubReference(kind="image", has_audio=False),
        StubReference(kind="video", has_audio=True),
        StubReference(kind="audio", has_audio=True),
        StubReference(kind="image", has_audio=False),
        StubReference(kind="video", has_audio=False),
    ]

    _build_presentation(
        tokenizer,
        "a prompt",
        references,
        image_token_counts=[2, 2],
        video_block_token_counts=[2, 2],
        video_block_timestamps=[[0.0], [0.5]],
    )

    assert tokenizer.texts == [
        "<Picture 1>: ",
        # The sound-bearing video takes an audio number first, and it is audio 1 -- the standalone
        # audio reference that follows is audio 2, not audio 1.
        "<Audio 1>: ",
        "<Video 1>: ",
        "<0.0 seconds>",
        # A waveform never reaches the conditioner: a standalone audio reference is its label alone.
        "<Audio 2>: ",
        "<Picture 2>: ",
        "<Video 2>: ",
        "<0.5 seconds>",
        "a prompt",
    ]


def test_the_modality_counters_are_independent():
    tokenizer = StubTokenizer()
    references = [StubReference(kind="image", has_audio=False)] * 2 + [
        StubReference(kind="video", has_audio=False)
    ] * 2

    _build_presentation(
        tokenizer,
        "p",
        references,
        image_token_counts=[1, 1],
        video_block_token_counts=[1, 1],
        video_block_timestamps=[[0.0], [0.0]],
    )

    labels = [text for text in tokenizer.texts if text.startswith("<Picture") or text.startswith("<Video")]
    assert labels == ["<Picture 1>: ", "<Picture 2>: ", "<Video 1>: ", "<Video 2>: "]


def test_the_tags_mark_a_vision_block_as_video_and_everything_else_as_text():
    tokenizer = StubTokenizer()
    _, tags = _build_presentation(
        tokenizer,
        "p",
        [StubReference(kind="image", has_audio=False)],
        image_token_counts=[3],
        video_block_token_counts=[],
        video_block_timestamps=[],
    )

    # label(1 token) + vision_start + 3 pads + vision_end + prompt(1 token)
    assert tags == [1, 0, 0, 0, 0, 0, 1]
