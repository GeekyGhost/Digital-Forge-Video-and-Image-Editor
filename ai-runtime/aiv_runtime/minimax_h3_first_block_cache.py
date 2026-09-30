"""Default-off First Block Cache for MiniMax-H3 denoising.

The transformer packs ``[text | conditions/references | target audio | target video]`` into one
sequence.  On similar adjacent denoising steps we still evaluate block 0, then reuse the residual
produced by the rest of the stack for target rows only.  Reference and text rows are deliberately
not cached because the output heads discard them.

This module owns no model code and imports no worker-only package at module import time.  It wraps
the already-instantiated Diffusers blocks, after any group-offload hooks have been installed, so a
skipped block also avoids its onload transfer.
"""

from __future__ import annotations

import math
import types
from dataclasses import dataclass
from typing import Any

DEFAULT_FIRST_BLOCK_CACHE_THRESHOLD = 0.08
MAX_FIRST_BLOCK_CACHE_THRESHOLD = 0.14
SIGNATURE_ELEMENT_CAP = 1 << 20


def validate_first_block_cache_threshold(value: float) -> float:
    """Validate the measured-useful cache threshold range, including zero for a parity probe."""

    value = float(value)
    if not math.isfinite(value) or value < 0.0 or value > MAX_FIRST_BLOCK_CACHE_THRESHOLD:
        raise ValueError(
            "first-block cache threshold must be between 0.0 and "
            f"{MAX_FIRST_BLOCK_CACHE_THRESHOLD:.2f}, got {value!r}"
        )
    return value


def _target_audio_start(audio_indices: Any, timestep_indices: Any) -> int:
    """Find the first target-audio row in an H3 packed sequence.

    ``audio_indices`` contains reference audio before target audio on Ref2VA.  The final target run
    is contiguous and shares one AdaLN timestep; walking backward across both properties separates
    it even when a standalone audio reference happens to be adjacent in the packed layout.
    """

    audio_rows = audio_indices.detach().to("cpu").reshape(-1).tolist()
    if not audio_rows:
        raise ValueError("MiniMax-H3 First Block Cache requires target audio rows")
    row_timesteps = timestep_indices.detach().to("cpu").reshape(-1)
    start = len(audio_rows) - 1
    target_timestep = int(row_timesteps[audio_rows[-1]].item())
    while start > 0:
        previous = int(audio_rows[start - 1])
        current = int(audio_rows[start])
        if previous + 1 != current or int(row_timesteps[previous].item()) != target_timestep:
            break
        start -= 1
    return int(audio_rows[start])


@dataclass
class FirstBlockCacheStats:
    full_steps: int = 0
    skipped_steps: int = 0
    minimum_change: float | None = None
    maximum_change: float | None = None

    def observe_change(self, value: float) -> None:
        self.minimum_change = value if self.minimum_change is None else min(self.minimum_change, value)
        self.maximum_change = value if self.maximum_change is None else max(self.maximum_change, value)


class MiniMaxH3FirstBlockCache:
    """State and wrappers for one transformer's repeated denoising forwards."""

    def __init__(
        self,
        transformer: Any,
        *,
        threshold: float,
        total_evaluations: int,
        signature_element_cap: int = SIGNATURE_ELEMENT_CAP,
    ) -> None:
        if total_evaluations < 1:
            raise ValueError("First Block Cache requires at least one transformer evaluation")
        if signature_element_cap < 1:
            raise ValueError("signature_element_cap must be positive")
        blocks = list(transformer.transformer_blocks)
        if len(blocks) < 2:
            raise ValueError("First Block Cache requires a transformer with at least two blocks")

        self.transformer = transformer
        self.blocks = blocks
        self.threshold = validate_first_block_cache_threshold(threshold)
        self.total_evaluations = int(total_evaluations)
        self.warmup_evaluations = max(1, math.ceil(self.total_evaluations * 0.25))
        self.signature_element_cap = int(signature_element_cap)
        self.stats = FirstBlockCacheStats()
        self._step_index = 0
        self._audio_start: int | None = None
        self._previous_signature: Any | None = None
        self._cached_tail_residual: Any | None = None
        self._tail_input: Any | None = None
        self._skip_step = False
        self._pre_hook = None
        self._original_forwards: list[Any] = []

    def reset(self, *, clear_stats: bool = False) -> None:
        """Drop every tensor that belongs to a completed or abandoned generation."""

        self._step_index = 0
        self._audio_start = None
        self._previous_signature = None
        self._cached_tail_residual = None
        self._tail_input = None
        self._skip_step = False
        if clear_stats:
            self.stats = FirstBlockCacheStats()

    def _begin_forward(self, _module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        if self._step_index >= self.total_evaluations:
            self.reset()

        def argument(name: str, position: int) -> Any:
            if name in kwargs:
                return kwargs[name]
            if position < len(args):
                return args[position]
            raise ValueError(f"MiniMax-H3 transformer forward did not receive {name}")

        # Bound Module.forward positional order: hidden, audio, text, timestep, timestep_indices,
        # token_tags, position_ids, video_indices, audio_indices, text_indices.
        timestep_indices = argument("timestep_indices", 4)
        audio_indices = argument("audio_indices", 8)
        audio_start = _target_audio_start(audio_indices, timestep_indices)
        if self._audio_start is None:
            self._audio_start = audio_start
        elif self._audio_start != audio_start:
            # Geometry/reference changes mean a new run even if the previous one ended early.
            self.reset()
            self._audio_start = audio_start

    def _signature(self, before: Any, after: Any) -> Any:
        """Sample block 0's summed gated branches without materialising a full residual copy."""

        before_flat = before.reshape(-1)
        after_flat = after.reshape(-1)
        stride = max(1, math.ceil(before_flat.numel() / self.signature_element_cap))
        return (
            after_flat[::stride][: self.signature_element_cap]
            - before_flat[::stride][: self.signature_element_cap]
        ).detach()

    def _observe_block_zero(self, before: Any, after: Any) -> None:
        signature = self._signature(before, after)
        should_skip = False
        if self._previous_signature is not None:
            numerator = (signature.float() - self._previous_signature.float()).abs().mean()
            denominator = self._previous_signature.float().abs().mean().clamp_min(1e-8)
            change = float((numerator / denominator).item())
            self.stats.observe_change(change)
            should_skip = (
                self._step_index >= self.warmup_evaluations
                and self._cached_tail_residual is not None
                and change <= self.threshold
            )
        self._previous_signature = signature
        self._skip_step = should_skip

    def _run_block(self, index: int, original: Any, hidden_states: Any, *args: Any, **kwargs: Any) -> Any:
        if index == 0:
            output = original(hidden_states, *args, **kwargs)
            self._observe_block_zero(hidden_states, output)
            return output

        if self._skip_step:
            if index == len(self.blocks) - 1:
                if self._audio_start is None or self._cached_tail_residual is None:
                    raise RuntimeError("First Block Cache attempted a skip without a target residual")
                # Inference-only path: block 0 produced this tensor for the current step and nothing
                # retains it after the stack.  Updating its target tail avoids a full-sequence copy.
                hidden_states[:, self._audio_start :, :].add_(self._cached_tail_residual)
                self.stats.skipped_steps += 1
                self._step_index += 1
            return hidden_states

        if index == 1:
            if self._audio_start is None:
                raise RuntimeError("First Block Cache did not receive the packed target boundary")
            self._tail_input = hidden_states[:, self._audio_start :, :].detach().clone()

        output = original(hidden_states, *args, **kwargs)
        if index == len(self.blocks) - 1:
            if self._audio_start is None or self._tail_input is None:
                raise RuntimeError("First Block Cache lost the full-step target input")
            self._cached_tail_residual = (
                output[:, self._audio_start :, :] - self._tail_input
            ).detach()
            self._tail_input = None
            self.stats.full_steps += 1
            self._step_index += 1
        return output

    def install(self) -> MiniMaxH3FirstBlockCache:
        """Wrap the current block forwards and add a transformer pre-hook."""

        if self._original_forwards:
            return self
        self._pre_hook = self.transformer.register_forward_pre_hook(
            self._begin_forward, with_kwargs=True
        )
        for index, block in enumerate(self.blocks):
            original = block.forward
            self._original_forwards.append(original)

            def cached_forward(
                _block: Any,
                hidden_states: Any,
                *args: Any,
                _index: int = index,
                _original: Any = original,
                **kwargs: Any,
            ) -> Any:
                return self._run_block(_index, _original, hidden_states, *args, **kwargs)

            block.forward = types.MethodType(cached_forward, block)
        return self

    def uninstall(self) -> None:
        for block, original in zip(self.blocks, self._original_forwards, strict=True):
            block.forward = original
        self._original_forwards.clear()
        if self._pre_hook is not None:
            self._pre_hook.remove()
            self._pre_hook = None
        self.reset()

    def summary(self) -> str:
        changes = "no comparable steps"
        if self.stats.minimum_change is not None and self.stats.maximum_change is not None:
            changes = (
                f"change={self.stats.minimum_change:.5f}..{self.stats.maximum_change:.5f}"
            )
        return (
            f"first block cache · threshold={self.threshold:.3f} · "
            f"full={self.stats.full_steps} · skipped={self.stats.skipped_steps} · {changes}"
        )


def install_first_block_cache(
    transformer: Any,
    *,
    enabled: bool,
    threshold: float = DEFAULT_FIRST_BLOCK_CACHE_THRESHOLD,
    total_evaluations: int,
) -> MiniMaxH3FirstBlockCache | None:
    """Install the cache only when explicitly enabled; disabled is an untouched parity path."""

    threshold = validate_first_block_cache_threshold(threshold)
    if not enabled or threshold == 0.0:
        return None
    return MiniMaxH3FirstBlockCache(
        transformer,
        threshold=threshold,
        total_evaluations=total_evaluations,
    ).install()
