"""Read tensors without a copy-on-write mapping, where the installed safetensors can.

`safetensors.safe_open` and `safetensors.torch.load_file` memory-map the file. On Windows a
private (copy-on-write) mapping is charged against the commit limit for its **entire range** at
map time -- traced through safetensors' `MmapOptions::map_copy_read_only` to memmap2's
`PAGE_WRITECOPY` + `FILE_MAP_COPY`, with the `make_read_only()` downgrade not firing because the
file is opened read-only. So opening the 27.5 GiB LTX checkpoint charges 27.5 GiB, and
`--offload cpu` then allocates roughly 26 GiB of pinned buffers on top, on a 63.7 GiB machine.

**safetensors 0.8.0 added the way out**: a keyword-only `backend` argument. `"mmap"` is the
default and is the behaviour above; `"pread"` reads tensor bytes with `pread(2)` and maps nothing.
Nothing in this module changes what is read -- only how the bytes get there.

The version matters and differs per venv on this machine, so `describe_backend` reports what is
actually in force rather than what was intended:

    application .venv        safetensors 0.8.0   -> pread available
    vendor/MiniMax-H3/.venv  safetensors 0.8.0   -> pread available
    vendor/LTX-2/.venv       safetensors 0.7.0   -> no `backend` argument; falls back to mmap

`ltx-core` requires a bare `safetensors` and `ltx-trainer` `>=0.5.0`, so the LTX venv is free to
move to 0.8.0; it simply resolved 0.7.0 when it was created. Until it does, every call here
degrades silently to the mapping and says so in the log.

The fallback is on `TypeError` rather than a version check: the argument's presence is the thing
that matters, and a version string is a proxy for it.
"""

from __future__ import annotations

from pathlib import Path

#: The backend that does not map. See the module docstring.
PREAD_BACKEND = "pread"

#: `None` until the first call has settled it. Cached because the answer cannot change within a
#: process, and because the probe is a failed call.
_PREAD_SUPPORTED: bool | None = None


def _remember(supported: bool) -> bool:
    global _PREAD_SUPPORTED
    _PREAD_SUPPORTED = supported
    return supported


def open_tensors(path: str | Path, framework: str = "pt", device: str = "cpu"):
    """`safe_open`, without the mapping where the installed safetensors allows it.

    Returns the handle itself, so callers keep using it as a context manager unchanged.
    """
    from safetensors import safe_open

    location = str(path)
    if _PREAD_SUPPORTED is not False:
        try:
            handle = safe_open(location, framework=framework, device=device, backend=PREAD_BACKEND)
        except TypeError:
            # The argument does not exist on this safetensors. Anything else means it was
            # accepted and the *file* is the problem, so record support and let the real error
            # through rather than retrying and reporting it twice.
            _remember(False)
        except Exception:
            _remember(True)
            raise
        else:
            _remember(True)
            return handle
    return safe_open(location, framework=framework, device=device)


def load_state_dict(path: str | Path, device: str = "cpu") -> dict:
    """`safetensors.torch.load_file`, without the mapping where it is supported."""
    import safetensors.torch

    location = str(path)
    if _PREAD_SUPPORTED is not False:
        try:
            state = safetensors.torch.load_file(location, device=device, backend=PREAD_BACKEND)
        except TypeError:
            _remember(False)
        except Exception:
            _remember(True)
            raise
        else:
            _remember(True)
            return state
    return safetensors.torch.load_file(location, device=device)


def describe_backend() -> str:
    """One line for the run log, so it is never a guess whether the fix is in force."""
    try:
        import safetensors

        version = getattr(safetensors, "__version__", "unknown")
    except ImportError:  # pragma: no cover - safetensors is a hard dependency of every worker
        return "safetensors is not installed"
    if _PREAD_SUPPORTED is None:
        return f"safetensors {version} · read backend not yet exercised"
    if _PREAD_SUPPORTED:
        return f"safetensors {version} · reading with pread, no file mapping"
    return (
        f"safetensors {version} · no `backend` argument, so reads are memory-mapped and Windows "
        "charges the whole file against the commit limit; safetensors >= 0.8.0 fixes this"
    )


class _PreadSafetensors:
    """A stand-in for the `safetensors` module whose `safe_open` avoids the mapping.

    The vendored `sft_loader` holds `safetensors` as a module global and calls
    `safetensors.safe_open(...)` through it, so replacing that one global redirects exactly that
    loader and nothing else. Patching `safetensors.safe_open` itself would reach every library in
    the process, which is a much larger promise than this needs to make.
    """

    def __init__(self, original):
        self._original = original

    def __getattr__(self, name):
        return getattr(self._original, name)

    def safe_open(self, filename, framework, device="cpu"):
        return open_tensors(filename, framework, device)


def install_pread_state_dict_loader() -> bool:
    """Point the vendored LTX state-dict loader at the non-mapping read. Idempotent.

    Returns whether the loader now reads without mapping, which is False on safetensors < 0.8.0.
    The vendored tree is left untouched so `uv sync` stays reproducible -- the same reason
    `scaled_fp8_lora` patches `WeightsProvider` in-process rather than editing it.
    """
    from ltx_core.loader import sft_loader

    if not isinstance(sft_loader.safetensors, _PreadSafetensors):
        sft_loader.safetensors = _PreadSafetensors(sft_loader.safetensors)

    # Settle whether pread is actually available, so the caller can report what is in force
    # rather than what was intended. `open_tensors` records the answer either way: a TypeError
    # means the argument does not exist, and any other failure -- this file is Python, not
    # safetensors -- means it was accepted. One throwaway call, once per process.
    if _PREAD_SUPPORTED is None:
        try:
            open_tensors(__file__)
        except Exception:  # noqa: BLE001 - the probe is the exception, not its message
            pass
    return bool(_PREAD_SUPPORTED)
