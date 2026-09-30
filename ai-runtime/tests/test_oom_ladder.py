"""The recovery ladder's ordering, which used to make an OOM strictly worse."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from aiv_runtime.oom import (
    DISK_OFFLOAD_RECOVERY_ENV,
    disk_offload_recovery_enabled,
    ltx_degradations,
    run_with_recovery,
)


def test_the_first_rung_is_not_the_one_that_crashes(monkeypatch):
    """Disk offload was rung one, and on this machine it takes the process down natively.

    Six 0xC0000005 access violations, every one inside torch_cpu.dll at the same offset, every one
    on the disk path. An access violation is not a Python exception, so `is_out_of_memory` never
    sees it and `run_with_recovery` never returns -- the ladder turned a recoverable OOM into an
    unrecoverable crash.
    """
    monkeypatch.delenv(DISK_OFFLOAD_RECOVERY_ENV, raising=False)
    ladder = ltx_degradations()

    assert ladder[0].description == "guidance disabled (CFG 1.0)"
    assert ladder[0].changes_output, "dropping guidance must announce that it changes the result"
    assert not any("disk" in rung.description for rung in ladder)


@pytest.mark.parametrize("value", ["1", "true", "YES", "on"])
def test_the_disk_rung_is_reachable_when_it_is_asked_for(monkeypatch, value):
    monkeypatch.setenv(DISK_OFFLOAD_RECOVERY_ENV, value)
    assert disk_offload_recovery_enabled()
    assert [rung.description for rung in ltx_degradations()] == [
        "guidance disabled (CFG 1.0)",
        "disk weight streaming",
    ]


@pytest.mark.parametrize("value", ["", "0", "false", "no", " "])
def test_anything_else_leaves_it_off(monkeypatch, value):
    monkeypatch.setenv(DISK_OFFLOAD_RECOVERY_ENV, value)
    assert not disk_offload_recovery_enabled()


def test_a_namespace_without_an_offload_is_not_assumed_to_be_on_disk(monkeypatch):
    """`getattr(args, "offload", "disk")` reported the rung already in effect and skipped it."""
    monkeypatch.setenv(DISK_OFFLOAD_RECOVERY_ENV, "1")
    disk = next(rung for rung in ltx_degradations() if "disk" in rung.description)

    args = SimpleNamespace()
    assert disk.apply(args) is True
    assert args.offload == "disk"
    # Second time it really is in effect, so the ladder moves on rather than burning a rung.
    assert disk.apply(args) is False


def test_the_ladder_actually_degrades_and_returns(monkeypatch):
    monkeypatch.delenv(DISK_OFFLOAD_RECOVERY_ENV, raising=False)
    args = SimpleNamespace(cfg_scale=2.5, audio_cfg_scale=2.5, offload="cpu")
    calls = {"n": 0}

    def run():
        calls["n"] += 1
        if args.cfg_scale > 1.0:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        return "done"

    assert run_with_recovery(run, args, ltx_degradations(), label="test") == "done"
    assert calls["n"] == 2
    assert (args.cfg_scale, args.audio_cfg_scale) == (1.0, 1.0)
