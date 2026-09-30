"""Named offload profiles for the LTX block streamer.

Pure data and two lookups, like `ltx_geometry`, so the three adapters, the tests and the Rust
mirror can agree on the same names without importing torch.

`cpu_slots_count` is a first-class argument of the streaming builder and we were using only its
two extremes. Measured from the FP8 checkpoint's own header on 2026-09-08 -- a header read, no
mapping -- the denoiser has **48 transformer blocks of 0.398 GiB each**: 19.11 GiB of block
weights, plus 4.55 GiB that is not block weights. So the choice was between pinning all 19.11 GiB
of blocks and pinning 0.80 GiB, with nothing in between:

    cpu           None   all 48 blocks pinned                      19.11 GiB pinned
    disk-cached     16   disk-backed source, a third cached         6.37 GiB pinned
    disk             2   DISK_CPU_SLOTS                             0.80 GiB pinned

`disk` is also the configuration this machine is documented as crashing on -- six 0xC0000005
access violations, every one inside `torch_cpu.dll` at the same offset, every one on the disk
path. `disk-cached` is the honest answer to "disk crashes, so cpu is the only option": same
disk-backed source, but with enough pinned buffers that the machine is not living at the bottom of
its slot count.

**16 is a starting point, not a measurement.** It is a third of the blocks and roughly a third of
the RAM. Peak host commitment and wall-clock against `cpu` on a standard retake are what should
set it; the adapters print peak VRAM and the geometry guard prints token counts, so one run at
each profile answers it.
"""

from __future__ import annotations

#: A third of the 48 blocks. See the module docstring for why this number is provisional.
DISK_CACHED_CPU_SLOTS = 16

#: The pinned-slot count each profile asks the builder for. `None` means "pin every block".
#: Ordered most-RAM-first, which is also fastest-first.
CPU_SLOTS_BY_PROFILE: dict[str, int | None] = {
    "cpu": None,
    "disk-cached": DISK_CACHED_CPU_SLOTS,
    #: Kept literal rather than imported: this module must not need torch. `scaled_fp8_lora`
    #: asserts it against the pinned tree's own `DISK_CPU_SLOTS` where that import is free.
    "disk": 2,
}

# Qualification is evidence, not availability. CPU completed the 9,609,216 pixel-frame run in the
# local worker log. Disk has six repeatable native faults on this host. Disk-cached remains usable
# for deliberate probing, but must not be represented as validated until it completes the same
# geometry/mode matrix.
QUALIFICATION_BY_PROFILE = {
    "cpu": "qualified-9609216",
    "disk-cached": "experimental-unqualified",
    "disk": "failed-native-access-violation",
}

#: The `--offload` choices, in the order the panel should offer them.
OFFLOAD_PROFILES = tuple(CPU_SLOTS_BY_PROFILE)

#: What the profile costs in pinned host RAM, for the log line and the panel's hint.
BLOCK_COUNT = 48
BLOCK_GIB = 0.398


def cpu_slots_for(profile: str) -> int | None:
    """The pinned-slot count a named profile asks for."""
    if profile not in CPU_SLOTS_BY_PROFILE:
        raise ValueError(
            f"unknown offload profile {profile!r}; expected one of {', '.join(OFFLOAD_PROFILES)}"
        )
    return CPU_SLOTS_BY_PROFILE[profile]


def pinned_gib_for(profile: str) -> float:
    """Roughly what a profile pins, from the measured per-block size."""
    slots = cpu_slots_for(profile)
    return (BLOCK_COUNT if slots is None else slots) * BLOCK_GIB


def describe_profile(profile: str) -> str:
    """One line for the run log: what was asked for, and what it costs."""
    slots = cpu_slots_for(profile)
    where = "all blocks pinned in RAM" if slots is None else f"disk-backed, {slots} blocks cached"
    return (
        f"offload {profile} · {where} · ~{pinned_gib_for(profile):.1f} GiB pinned · "
        f"{QUALIFICATION_BY_PROFILE[profile]}"
    )


def offload_mode_for(profile: str):
    """The vendored `OffloadMode` a named profile maps onto.

    `disk-cached` is ours, not theirs: their enum has two members, and what separates it from
    `disk` is the slot count rather than the mode. Both are disk-backed sources.
    """
    from ltx_pipelines.utils.types import OffloadMode

    return OffloadMode("cpu" if profile == "cpu" else "disk")
