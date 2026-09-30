"""The header reader, and the rule that metadata questions never map a checkpoint.

`safe_open` on Windows maps the file copy-on-write, and Windows charges a private mapping's whole
range against the commit limit at map time. Six probes in this package asked one key-suffix or
metadata question apiece and paid 20-27.5 GiB of commit each to get the answer.
"""

from __future__ import annotations

import json
import os
import struct
from pathlib import Path

import pytest

from aiv_runtime.safetensors_header import (
    MAX_HEADER_BYTES,
    dtypes,
    has_key_suffix,
    metadata,
    read_header,
    tensor_keys,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNTIME = REPO_ROOT / "ai-runtime" / "aiv_runtime"

#: Small enough that mapping them in a test costs nothing, real enough to be worth comparing.
SMALL_CHECKPOINTS = (
    "models/loras/ltx/ltx-2.3-22b-lora-foley-v2a-1.0.safetensors",
    "models/kiwi-edit-5b/source_embedder/diffusion_pytorch_model.safetensors",
    "models/LTX-2.3/ltx-2.3-temporal-upscaler-x2-1.0.safetensors",
)

#: The plan's acceptance criterion is stated on the real checkpoints, and running it costs the
#: exact commit charge this work removes -- so it is opt-in rather than part of every gate.
PRODUCTION_CHECKPOINTS = (
    "models/LTX-2.3/ltx-2.3-22b-distilled-fp8.safetensors",
    "models/MiniMax-H3/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
    "models/MiniMax-H3/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors",
)

FULL_CHECK_ENV = "AIV_VERIFY_HEADER_READS_FULL"


def _write_safetensors_header(path: Path, header: dict, payload: bytes = b"") -> None:
    raw = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)


def _compare_with_safe_open(relative: str) -> None:
    checkpoint = REPO_ROOT / relative
    if not checkpoint.is_file():
        pytest.skip(f"{relative} is not installed on this machine")
    safetensors = pytest.importorskip("safetensors")

    with safetensors.safe_open(str(checkpoint), framework="pt") as mapped:
        mapped_keys = set(mapped.keys())
        mapped_metadata = mapped.metadata() or {}

    assert tensor_keys(checkpoint) == mapped_keys
    assert metadata(checkpoint) == mapped_metadata


@pytest.mark.parametrize("relative", SMALL_CHECKPOINTS)
def test_the_header_reader_returns_what_safe_open_returns(relative):
    """The acceptance criterion, on files small enough to check every run."""
    _compare_with_safe_open(relative)


@pytest.mark.parametrize("relative", PRODUCTION_CHECKPOINTS)
@pytest.mark.skipif(
    os.environ.get(FULL_CHECK_ENV, "").strip().lower() not in {"1", "true", "yes", "on"},
    reason=f"set {FULL_CHECK_ENV}=1 to compare against the 20-27.5 GiB production checkpoints",
)
def test_the_header_reader_agrees_on_the_production_checkpoints(relative):
    """The same comparison on the real files. Opt-in: running it charges the commit this removes."""
    _compare_with_safe_open(relative)


def test_the_probes_answer_what_the_call_sites_ask(tmp_path):
    blob = tmp_path / "fake.safetensors"
    _write_safetensors_header(blob, {
        "__metadata__": {"alpha": "8", "is_turbo_baked": "True"},
        "blocks.0.weight": {"dtype": "F8_E4M3", "shape": [2, 2], "data_offsets": [0, 4]},
        "blocks.0.weight_scale": {"dtype": "F32", "shape": [1], "data_offsets": [4, 8]},
        "blocks.0.comfy_quant": {"dtype": "I64", "shape": [1], "data_offsets": [8, 16]},
    }, payload=b"\0" * 16)

    assert tensor_keys(blob) == {"blocks.0.weight", "blocks.0.weight_scale", "blocks.0.comfy_quant"}
    assert "__metadata__" not in tensor_keys(blob)
    assert metadata(blob) == {"alpha": "8", "is_turbo_baked": "True"}
    assert has_key_suffix(blob, ".weight_scale") is True
    assert has_key_suffix(blob, ".comfy_quant") is True
    assert has_key_suffix(blob, ".aiv_fp8.qdata") is False
    assert dtypes(blob)["blocks.0.weight"] == "F8_E4M3"


def test_a_file_without_metadata_reads_as_an_empty_dict(tmp_path):
    blob = tmp_path / "bare.safetensors"
    _write_safetensors_header(blob, {"w": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}})
    assert metadata(blob) == {}
    assert tensor_keys(blob) == {"w"}


@pytest.mark.parametrize(
    ("name", "content", "message"),
    [
        ("short", b"\x01\x02", "too short"),
        ("zero", struct.pack("<Q", 0), "not plausible"),
        ("absurd", struct.pack("<Q", MAX_HEADER_BYTES + 1), "not plausible"),
        ("truncated", struct.pack("<Q", 4096) + b"{}", "ends inside its own header"),
        ("garbage", struct.pack("<Q", 4) + b"not!", "unreadable"),
        ("not_an_object", struct.pack("<Q", 2) + b"[]", "not an object"),
    ],
)
def test_a_file_that_is_not_safetensors_says_so(tmp_path, name, content, message):
    """A `.pt` renamed, a truncated download, or a file still being written."""
    blob = tmp_path / f"{name}.safetensors"
    blob.write_bytes(content)
    with pytest.raises(ValueError, match=message):
        read_header(blob)


def test_no_metadata_probe_maps_a_checkpoint_again():
    """`safe_open` belongs in exactly one module, which chooses how the bytes are read.

    Checking the source rather than behaviour, for the same reason the worker-import check does:
    an import-based test passes on whichever machine happens to have the file, and a probe that
    maps 27.5 GiB is not something you want to discover from a full disk.
    """
    import ast

    def calls_safe_open(source: Path) -> bool:
        # AST, not text: the header module names `safe_open` all over its own docstrings, and a
        # comment mentioning it is not a call.
        for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            name = function.attr if isinstance(function, ast.Attribute) else getattr(function, "id", None)
            if name == "safe_open":
                return True
        return False

    users = {source.name for source in sorted(RUNTIME.glob("*.py")) if calls_safe_open(source)}
    assert users == {"safetensors_reader.py"}, (
        "every safetensors read goes through `safetensors_reader`, which chooses a backend that "
        "does not map the file, and every metadata question goes through `safetensors_header`, "
        f"which reads the header alone. Unexpected callers: {sorted(users - {'safetensors_reader.py'})}"
    )


def test_the_header_module_stays_importable_from_every_venv():
    """It is imported from the LTX venv, the H3 venv and the application venv."""
    import ast

    tree = ast.parse((RUNTIME / "safetensors_header.py").read_text(encoding="utf-8"))
    imported = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported <= {"__future__", "json", "struct", "pathlib"}, (
        f"the header reader must stay stdlib-only; it imports {sorted(imported)}"
    )
