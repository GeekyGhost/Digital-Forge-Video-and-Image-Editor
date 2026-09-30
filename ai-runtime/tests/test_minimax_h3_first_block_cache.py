"""Pins the independently implemented MiniMax-H3 First Block Cache without model weights."""

from __future__ import annotations

import pytest

from aiv_runtime.minimax_h3_first_block_cache import (
    DEFAULT_FIRST_BLOCK_CACHE_THRESHOLD,
    MiniMaxH3FirstBlockCache,
    _target_audio_start,
    install_first_block_cache,
    validate_first_block_cache_threshold,
)


def _transformer(blocks: int = 50):
    from torch import nn

    class AdditiveBlock(nn.Module):
        def __init__(self, delta: float) -> None:
            super().__init__()
            self.delta = delta
            self.calls = 0

        def forward(self, hidden_states, temb, adaln_indices, rotary_emb):
            self.calls += 1
            # Block 0's residual changes with temb. The rest of the stack is deterministic, which
            # makes an accepted cache hit exactly checkable on the target rows.
            delta = temb if self.delta == 0.0 else self.delta
            return hidden_states + delta

    class FakeTransformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.transformer_blocks = nn.ModuleList(
                AdditiveBlock(0.0 if index == 0 else index / 100.0)
                for index in range(blocks)
            )

        def forward(
            self,
            hidden_states,
            audio_hidden_states,
            encoder_hidden_states,
            timestep,
            timestep_indices,
            token_tags,
            position_ids,
            video_indices,
            audio_indices,
            text_indices,
            attention_kwargs=None,
            return_dict=True,
        ):
            del audio_hidden_states, encoder_hidden_states, timestep, token_tags
            del position_ids, video_indices, audio_indices, text_indices, attention_kwargs
            rotary_emb = (None, None)
            for block in self.transformer_blocks:
                hidden_states = block(hidden_states, self._temb, timestep_indices, rotary_emb)
            return hidden_states

    return FakeTransformer()


def _run(transformer, *, signal: float = 1.0, mixed_references: bool = False):
    import torch

    transformer._temb = torch.tensor(signal)
    hidden = torch.arange(24, dtype=torch.float32).reshape(1, 6, 4)
    if mixed_references:
        # Text 0..1, reference audio 2..3 at AdaLN row 1, target audio 4..5 at row 2.
        audio_indices = torch.tensor([2, 3, 4, 5])
        timestep_indices = torch.tensor([0, 0, 1, 1, 2, 2])
    else:
        audio_indices = torch.tensor([2, 3])
        timestep_indices = torch.tensor([0, 0, 1, 1, 2, 2])
    output = transformer(
        hidden_states=hidden,
        audio_hidden_states=torch.empty(1, 0, 1),
        encoder_hidden_states=torch.empty(1, 0, 1),
        timestep=torch.tensor([0.0]),
        timestep_indices=timestep_indices,
        token_tags=torch.zeros(6, dtype=torch.long),
        position_ids=torch.zeros(6, 3),
        video_indices=torch.tensor([4, 5]),
        audio_indices=audio_indices,
        text_indices=torch.tensor([0, 1]),
    )
    return hidden, output


def test_target_audio_start_excludes_adjacent_reference_audio():
    import torch

    audio_indices = torch.tensor([2, 3, 4, 5])
    timestep_indices = torch.tensor([0, 0, 1, 1, 2, 2])
    assert _target_audio_start(audio_indices, timestep_indices) == 4


def test_first_block_signature_is_capped_without_a_full_residual_copy():
    import torch

    transformer = _transformer(2)
    cache = MiniMaxH3FirstBlockCache(
        transformer, threshold=0.08, total_evaluations=2, signature_element_cap=8
    )
    before = torch.arange(100, dtype=torch.float32)
    signature = cache._signature(before, before + 3)
    assert signature.numel() <= 8
    torch.testing.assert_close(signature, torch.full_like(signature, 3))


def test_a_cache_hit_runs_block_zero_and_adds_only_the_target_residual():
    import torch

    transformer = _transformer()
    cache = install_first_block_cache(
        transformer, enabled=True, threshold=0.08, total_evaluations=4
    )
    assert cache is not None

    _, first = _run(transformer, mixed_references=True)
    _, second = _run(transformer, mixed_references=True)

    assert transformer.transformer_blocks[0].calls == 2
    assert all(block.calls == 1 for block in transformer.transformer_blocks[1:])
    # The output heads discard rows before audio_start. The cached target tail is exactly the full
    # stack's result, while prefix rows intentionally contain block 0 only on the skipped step.
    torch.testing.assert_close(second[:, 4:, :], first[:, 4:, :])
    assert not torch.equal(second[:, :4, :], first[:, :4, :])
    assert cache.stats.full_steps == 1
    assert cache.stats.skipped_steps == 1


def test_a_changed_block_zero_signature_forces_a_full_step():
    transformer = _transformer()
    cache = install_first_block_cache(
        transformer, enabled=True, threshold=0.08, total_evaluations=4
    )
    assert cache is not None
    _run(transformer, signal=1.0)
    _run(transformer, signal=2.0)
    assert all(block.calls == 2 for block in transformer.transformer_blocks)
    assert cache.stats.full_steps == 2
    assert cache.stats.skipped_steps == 0


def test_cache_state_resets_before_a_second_generation():
    transformer = _transformer()
    cache = install_first_block_cache(
        transformer, enabled=True, threshold=0.08, total_evaluations=2
    )
    assert cache is not None
    _run(transformer)
    _run(transformer)
    _run(transformer)
    # First run: full + skipped. The first step of the next run must be full again.
    assert transformer.transformer_blocks[1].calls == 2
    assert cache.stats.full_steps == 2
    assert cache.stats.skipped_steps == 1


def test_threshold_zero_and_default_off_leave_forward_untouched():
    transformer = _transformer()
    original = transformer.transformer_blocks[0].forward
    assert install_first_block_cache(
        transformer, enabled=False, total_evaluations=4
    ) is None
    assert transformer.transformer_blocks[0].forward == original
    assert install_first_block_cache(
        transformer, enabled=True, threshold=0.0, total_evaluations=4
    ) is None
    assert transformer.transformer_blocks[0].forward == original


def test_cache_threshold_contract():
    assert validate_first_block_cache_threshold(DEFAULT_FIRST_BLOCK_CACHE_THRESHOLD) == 0.08
    assert validate_first_block_cache_threshold(0.0) == 0.0
    assert validate_first_block_cache_threshold(0.14) == 0.14
    for invalid in (-0.01, 0.1401, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="threshold"):
            validate_first_block_cache_threshold(invalid)
