"""The named offload profiles, and the intermediate one that did not exist.

`cpu_slots_count` is a first-class argument of the streaming builder and only its two extremes
were reachable: pin all 48 blocks, or pin 2 -- and 2 is the configuration this machine is
documented as crashing on, with six 0xC0000005 access violations at one `torch_cpu.dll` offset.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aiv_runtime import ltx_offload
from aiv_runtime.ltx_offload import (
    BLOCK_COUNT,
    CPU_SLOTS_BY_PROFILE,
    DISK_CACHED_CPU_SLOTS,
    OFFLOAD_PROFILES,
    QUALIFICATION_BY_PROFILE,
    cpu_slots_for,
    describe_profile,
    pinned_gib_for,
)

RUNTIME = Path(ltx_offload.__file__).parent


def test_there_is_something_between_all_and_almost_nothing():
    assert OFFLOAD_PROFILES == ("cpu", "disk-cached", "disk")
    assert cpu_slots_for("cpu") is None, "None means pin every block"
    assert cpu_slots_for("disk-cached") == DISK_CACHED_CPU_SLOTS == 16
    assert cpu_slots_for("disk") == 2

    # Ordered most-RAM-first, which is also fastest-first.
    costs = [pinned_gib_for(profile) for profile in OFFLOAD_PROFILES]
    assert costs == sorted(costs, reverse=True)


def test_the_intermediate_profile_sits_where_it_claims_to():
    """Measured from the FP8 checkpoint's header: 48 blocks of 0.398 GiB."""
    assert BLOCK_COUNT == 48
    assert pinned_gib_for("cpu") == pytest.approx(19.1, abs=0.1)
    assert pinned_gib_for("disk-cached") == pytest.approx(6.4, abs=0.1)
    assert pinned_gib_for("disk") == pytest.approx(0.8, abs=0.1)
    # A third of the blocks, so a third of the RAM.
    assert DISK_CACHED_CPU_SLOTS == BLOCK_COUNT // 3


def test_an_unknown_profile_is_named_rather_than_silently_defaulted():
    with pytest.raises(ValueError, match="unknown offload profile"):
        cpu_slots_for("balanced")
    with pytest.raises(ValueError, match="cpu, disk-cached, disk"):
        cpu_slots_for("")


def test_the_description_says_what_it_costs():
    assert "all blocks pinned" in describe_profile("cpu")
    assert "16 blocks cached" in describe_profile("disk-cached")
    assert "6.4 GiB pinned" in describe_profile("disk-cached")


def test_qualification_is_not_inferred_from_an_available_menu_option():
    assert QUALIFICATION_BY_PROFILE == {
        "cpu": "qualified-9609216",
        "disk-cached": "experimental-unqualified",
        "disk": "failed-native-access-violation",
    }
    for profile, status in QUALIFICATION_BY_PROFILE.items():
        assert status in describe_profile(profile)


def test_the_module_needs_no_torch():
    """It is imported by the adapters, the tests and -- in mirror -- by Rust and the panel."""
    import ast

    tree = ast.parse((RUNTIME / "ltx_offload.py").read_text(encoding="utf-8"))
    imported = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported <= {"__future__"}, f"module-level imports: {sorted(imported)}"


def test_every_adapter_offers_the_same_three_and_reports_which_it_got():
    for name in ("ltx_single_stage.py", "ltx_retake_16gb.py", "ltx_foley_v2a.py"):
        source = (RUNTIME / name).read_text(encoding="utf-8")
        assert 'choices=OFFLOAD_PROFILES' in source, name
        assert "offload_mode_for(args.offload)" in source, name
        assert "cpu_slots_for(args.offload)" in source, name
        assert "describe_profile(args.offload)" in source, name


def test_the_rust_mirror_lists_the_same_profiles():
    rust = RUNTIME.parents[1] / "src-tauri" / "src" / "ai.rs"
    if not rust.is_file():
        pytest.skip("the Rust source is not present")
    line = next(
        text for text in rust.read_text(encoding="utf-8").splitlines()
        if text.startswith("const LTX_OFFLOAD_PROFILES")
    )
    for profile in OFFLOAD_PROFILES:
        assert f'"{profile}"' in line, f"{profile} is missing from the Rust mirror"
