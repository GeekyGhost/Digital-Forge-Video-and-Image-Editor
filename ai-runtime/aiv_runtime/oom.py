"""Automatic recovery from CUDA out-of-memory, shared by the model adapters.

Running 22B-class models on a 16 GB card means OOM is a normal operating condition, not an
exceptional one. Every adapter should degrade and retry rather than hand the user a stack trace,
so the recovery ladder lives here instead of being reinvented per worker.

Degradations are ordered least-destructive first: change how memory is *managed* before changing
anything that alters the result, and say so loudly when the result does change.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Callable, NamedTuple


class Degradation(NamedTuple):
    """One rung of the recovery ladder."""

    description: str
    apply: Callable[[Any], bool]
    """Mutates the argument namespace. Returns False when it is already in effect, so the ladder
    skips rungs that would not change anything."""
    changes_output: bool = False


def is_out_of_memory(error: BaseException) -> bool:
    """Catch OOM regardless of which layer raised it.

    torch.cuda.OutOfMemoryError is the common case, but allocator failures also surface as a plain
    RuntimeError from cuBLAS/cuDNN, and safetensors/accelerate can wrap them again.
    """
    try:
        import torch

        if isinstance(error, torch.cuda.OutOfMemoryError):
            return True
    except (ImportError, AttributeError):
        pass
    text = str(error).lower()
    return any(
        marker in text
        for marker in (
            "out of memory",
            "cuda error: out of memory",
            "cublas_status_alloc_failed",
            "cudnn_status_alloc_failed",
        )
    )


def free_cuda_memory() -> None:
    try:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except ImportError:
        pass


def run_with_recovery(
    run: Callable[[], Any],
    args: Any,
    degradations: list[Degradation],
    *,
    label: str = "worker",
) -> Any:
    """Call `run()`, walking down the ladder on each OOM until something completes.

    Non-OOM errors propagate immediately — degrading in response to an unrelated bug would only
    obscure it.
    """
    attempted: list[str] = []
    for index in range(len(degradations) + 1):
        try:
            return run()
        except Exception as error:  # noqa: BLE001 - re-raised below unless it is an OOM
            if not is_out_of_memory(error) or index >= len(degradations):
                if attempted:
                    print(
                        f"[{label}] still out of memory after: {', '.join(attempted)}",
                        file=sys.stderr,
                    )
                raise
            free_cuda_memory()
            # Skip rungs that are already in effect so one OOM does not burn a useful option.
            step = index
            while step < len(degradations):
                degradation = degradations[step]
                if degradation.apply(args):
                    note = " (this changes the result)" if degradation.changes_output else ""
                    print(
                        f"[{label}] out of memory — retrying with {degradation.description}{note}",
                        file=sys.stderr,
                    )
                    attempted.append(degradation.description)
                    break
                step += 1
            else:
                print(f"[{label}] out of memory and no recovery options remain", file=sys.stderr)
                raise
    raise RuntimeError("unreachable")


DISK_OFFLOAD_RECOVERY_ENV = "AIV_ALLOW_DISK_OFFLOAD_RECOVERY"


def disk_offload_recovery_enabled() -> bool:
    """Whether the recovery ladder is allowed to fall back to disk weight streaming.

    Off unless explicitly asked for, because on this machine the disk path does not fail -- it
    crashes. Six `0xC0000005` access violations on 2026-08-11/12, every one inside `torch_cpu.dll`
    at the same offset, every one on the disk path, with the identical clip completing cleanly
    under `cpu`. Both LTX adapters document it at their `--offload` argument.

    An access violation is not a Python exception. `is_out_of_memory` never sees it, the `except`
    in `run_with_recovery` never runs, and the process dies where it stands -- so a ladder that
    reaches for disk first converts a recoverable OOM into an unrecoverable crash. `disk` stays
    available as a deliberate `--offload disk` choice for machines short of RAM; what this gates
    is the ladder choosing it on the user's behalf.
    """
    return os.environ.get(DISK_OFFLOAD_RECOVERY_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def ltx_degradations() -> list[Degradation]:
    """The ladder shared by the LTX adapters.

    Dropping CFG comes first. It genuinely changes the output -- and it says so -- but it halves
    the per-step work by removing the unconditional pass, and it returns. Disk streaming costs no
    quality on paper and is the more appealing first rung for exactly that reason, which is how it
    came to be one; it is gated behind `AIV_ALLOW_DISK_OFFLOAD_RECOVERY` instead, for the reason
    `disk_offload_recovery_enabled` gives. A rung that takes the process down with an access
    violation is not a recovery option, however cheap it looks.
    """

    def drop_cfg(args: Any) -> bool:
        video = getattr(args, "cfg_scale", 1.0)
        audio = getattr(args, "audio_cfg_scale", 1.0)
        if video <= 1.0 and audio <= 1.0:
            return False
        args.cfg_scale = 1.0
        args.audio_cfg_scale = 1.0
        return True

    def to_disk_offload(args: Any) -> bool:
        # "cpu", not "disk". The old default said that a namespace carrying no `offload` at all
        # was already streaming from disk, so this rung reported itself unavailable and the ladder
        # silently skipped the only option it had left.
        if getattr(args, "offload", "cpu") == "disk":
            return False
        args.offload = "disk"
        return True

    ladder = [Degradation("guidance disabled (CFG 1.0)", drop_cfg, changes_output=True)]
    if disk_offload_recovery_enabled():
        ladder.append(Degradation("disk weight streaming", to_disk_offload))
    return ladder
