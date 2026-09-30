"""Exercises the H3 profiler without loading production model weights."""

from __future__ import annotations

import json

import pytest
import torch
from torch import nn

from aiv_runtime.minimax_h3_profiler import (
    MiniMaxH3DenoiseProfiler,
    profile_requested,
    resolve_profile_path,
)


class _TinyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = nn.Linear(4, 4, bias=False)
        self.to_k = nn.Linear(4, 4, bias=False)
        self.to_v = nn.Linear(4, 4, bias=False)

    def forward(self, hidden_states):
        return self.to_q(hidden_states) + self.to_k(hidden_states) + self.to_v(hidden_states)


class _TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = _TinyAttention()
        self.ff = nn.Sequential(nn.Linear(4, 8), nn.SiLU(), nn.Linear(8, 4))

    def forward(self, hidden_states):
        return self.ff(self.attn(hidden_states))


class _TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = nn.ModuleList([_TinyBlock()])

    def forward(self, *, hidden_states, timestep):
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states)
        return hidden_states


def test_profiler_writes_step_phases_and_summary(tmp_path):
    transformer = _TinyTransformer()
    report = tmp_path / "profile.jsonl"

    with MiniMaxH3DenoiseProfiler(
        transformer,
        report,
        metadata={"workflow": "test"},
    ):
        transformer(hidden_states=torch.ones(1, 5, 4), timestep=torch.tensor([0.5]))

    records = [json.loads(line) for line in report.read_text(encoding="utf-8").splitlines()]
    assert [record["type"] for record in records] == ["metadata", "step", "summary"]
    assert records[0]["format"] == "aiv-h3-denoise-profile-v1"
    assert records[0]["workflow"] == "test"
    step = records[1]
    assert step["index"] == 0
    assert step["sequence_rows"] == 5
    assert step["timestep"] == 0.5
    assert step["phases"]["qkv"]["calls"] == 3
    assert step["phases"]["attention_total"]["calls"] == 1
    assert step["phases"]["mlp"]["calls"] == 1
    assert step["phases"]["transformer_block"]["calls"] == 1
    assert records[2]["completed_steps"] == 1
    assert records[2]["error"] is None


def test_profile_path_enables_profiler_and_defaults_beside_video(tmp_path):
    class Args:
        profile_denoise = False
        profile_path = None
        output_path = str(tmp_path / "clip.mp4")

    args = Args()
    assert not profile_requested(args)
    assert resolve_profile_path(args) == tmp_path / "clip.mp4.h3-profile.jsonl"

    args.profile_path = str(tmp_path / "diagnostic.jsonl")
    assert profile_requested(args)
    assert resolve_profile_path(args) == tmp_path / "diagnostic.jsonl"


def test_profile_report_cannot_overwrite_the_generated_video(tmp_path):
    class Args:
        output_path = str(tmp_path / "clip.mp4")
        audio_path = str(tmp_path / "clip.wav")
        profile_path = output_path

    with pytest.raises(ValueError, match="must not overwrite"):
        resolve_profile_path(Args())
