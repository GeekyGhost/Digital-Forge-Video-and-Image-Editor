"""The read backend, and the loader patch that reaches the vendored LTX tree.

safetensors 0.8.0 added a keyword-only `backend`: `"mmap"` (the default, and what Windows charges
in full against the commit limit) or `"pread"`, which maps nothing. The LTX venv is still on
0.7.0, which has no such argument, so every call here has to degrade to the mapping and say so
rather than silently pretending the fix is in force.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from aiv_runtime import safetensors_reader
from aiv_runtime.safetensors_reader import (
    PREAD_BACKEND,
    _PreadSafetensors,
    describe_backend,
    install_pread_state_dict_loader,
    load_state_dict,
    open_tensors,
)

RUNTIME = Path(safetensors_reader.__file__).parent


@pytest.fixture(autouse=True)
def _forget_the_probe(monkeypatch):
    """Each test settles the cache itself; a leaked answer would make the next one meaningless."""
    monkeypatch.setattr(safetensors_reader, "_PREAD_SUPPORTED", None)


def _fake_safetensors(monkeypatch, *, accepts_backend: bool):
    calls = []

    def safe_open(filename, framework, device="cpu", **kwargs):
        if "backend" in kwargs and not accepts_backend:
            raise TypeError("safe_open() got an unexpected keyword argument 'backend'")
        calls.append(dict(filename=filename, framework=framework, device=device, **kwargs))
        return f"handle:{filename}"

    def load_file(filename, device="cpu", **kwargs):
        if "backend" in kwargs and not accepts_backend:
            raise TypeError("load_file() got an unexpected keyword argument 'backend'")
        calls.append(dict(filename=filename, device=device, **kwargs))
        return {"w": filename}

    module = types.SimpleNamespace(safe_open=safe_open, __version__="0.8.0" if accepts_backend else "0.7.0")
    torch_module = types.SimpleNamespace(load_file=load_file)
    monkeypatch.setitem(sys.modules, "safetensors", module)
    monkeypatch.setitem(sys.modules, "safetensors.torch", torch_module)
    module.torch = torch_module
    return calls


def test_a_modern_safetensors_reads_without_mapping(monkeypatch):
    calls = _fake_safetensors(monkeypatch, accepts_backend=True)

    assert open_tensors("model.safetensors") == "handle:model.safetensors"
    assert calls[-1]["backend"] == PREAD_BACKEND
    assert load_state_dict("model.safetensors") == {"w": "model.safetensors"}
    assert calls[-1]["backend"] == PREAD_BACKEND
    assert "pread" in describe_backend()
    assert "0.8.0" in describe_backend()


def test_an_older_safetensors_falls_back_and_admits_it(monkeypatch):
    """The LTX venv is on 0.7.0 today. A silent fallback would be worse than no fix at all."""
    calls = _fake_safetensors(monkeypatch, accepts_backend=False)

    assert open_tensors("model.safetensors") == "handle:model.safetensors"
    assert "backend" not in calls[-1]
    assert load_state_dict("model.safetensors") == {"w": "model.safetensors"}
    assert "backend" not in calls[-1]

    reported = describe_backend()
    assert "memory-mapped" in reported
    assert "commit limit" in reported
    assert ">= 0.8.0" in reported


def test_the_answer_is_settled_once_not_probed_every_call(monkeypatch):
    calls = _fake_safetensors(monkeypatch, accepts_backend=False)
    for _ in range(5):
        open_tensors("model.safetensors")
    # Five reads, five calls: the failed probe happens once, not once per read.
    assert len(calls) == 5
    assert all("backend" not in call for call in calls)


def test_the_loader_patch_redirects_that_loader_and_nothing_else(monkeypatch):
    """Patching `safetensors.safe_open` itself would reach every library in the process."""
    _fake_safetensors(monkeypatch, accepts_backend=True)
    original = sys.modules["safetensors"]
    sft_loader = types.SimpleNamespace(safetensors=original)
    monkeypatch.setitem(
        sys.modules, "ltx_core.loader", types.SimpleNamespace(sft_loader=sft_loader)
    )

    assert install_pread_state_dict_loader() is True
    assert isinstance(sft_loader.safetensors, _PreadSafetensors)
    assert sft_loader.safetensors.safe_open("x", "pt") == "handle:x"
    # The real module is untouched, and anything else on it still resolves through.
    assert sys.modules["safetensors"] is original
    assert sft_loader.safetensors.__version__ == "0.8.0"

    # Idempotent: a second install must not wrap the wrapper.
    wrapped = sft_loader.safetensors
    install_pread_state_dict_loader()
    assert sft_loader.safetensors is wrapped


def test_the_patch_reports_honestly_on_an_old_safetensors(monkeypatch):
    _fake_safetensors(monkeypatch, accepts_backend=False)
    sft_loader = types.SimpleNamespace(safetensors=sys.modules["safetensors"])
    monkeypatch.setitem(
        sys.modules, "ltx_core.loader", types.SimpleNamespace(sft_loader=sft_loader)
    )
    assert install_pread_state_dict_loader() is False
    assert "memory-mapped" in describe_backend()


def test_every_ltx_adapter_installs_the_patch_and_says_what_it_got():
    """Three adapters, three venvs' worth of doubt; none of them should leave it implicit."""
    for name in ("ltx_single_stage.py", "ltx_retake_16gb.py", "ltx_foley_v2a.py"):
        source = (RUNTIME / name).read_text(encoding="utf-8")
        assert "install_pread_state_dict_loader()" in source, name
        assert "describe_backend()" in source, name
