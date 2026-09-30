"""Read a safetensors header without mapping the file.

Every question these functions answer -- which keys exist, what the metadata says -- lives in a
few kilobytes at the front of the file. `safetensors.safe_open` answers them too, and on Windows
it costs the whole file.

The chain, traced 2026-09-08: `safe_open` -> safetensors 0.7.0 `MmapOptions::map_copy_read_only`
-> memmap2 `PAGE_WRITECOPY` + `FILE_MAP_COPY`. A private, copy-on-write file mapping is charged
against the Windows commit limit for its **entire range** at map time, and the `make_read_only()`
downgrade does not fire because the file is opened read-only. So probing a 27.5 GiB checkpoint for
a key suffix charges 27.5 GiB of commit, and probing the 20 GiB H3 denoiser charges 20 GiB -- each
time, on a 63.7 GiB machine that is also holding ~26 GiB of pinned CPU buffers.

The format is documented and trivial: an 8-byte little-endian header length, then that many bytes
of JSON. The pinned LTX tree already reads it this way in
`ltx_core/quantization/fp8_scaled_mm.py::_read_safetensors_dtypes`; this module is that idea with
the guards a shared helper needs.

Stdlib only, deliberately: this is imported from the LTX venv, the MiniMax-H3 venv and the
application venv, and it must not drag safetensors or torch into any of them.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

#: Sanity bound on the header length read out of the file. Real headers run to a few hundred
#: kilobytes; anything past this means the first eight bytes were not a header length -- a
#: truncated download, a `.pt` renamed to `.safetensors`, or a file still being written.
MAX_HEADER_BYTES = 64 * 1024 * 1024


def read_header(path: str | Path) -> dict:
    """The file's JSON header, `__metadata__` included. Never maps the file."""
    location = Path(path)
    with location.open("rb") as blob:
        prefix = blob.read(8)
        if len(prefix) < 8:
            raise ValueError(f"{location.name} is too short to be a safetensors file")
        length = struct.unpack("<Q", prefix)[0]
        if length == 0 or length > MAX_HEADER_BYTES:
            raise ValueError(
                f"{location.name} declares a {length:,}-byte safetensors header, which is not "
                "plausible; the file is probably truncated or is not safetensors at all"
            )
        raw = blob.read(length)
    if len(raw) < length:
        raise ValueError(f"{location.name} ends inside its own header; the file is truncated")
    try:
        header = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"{location.name} has an unreadable safetensors header: {error}") from error
    if not isinstance(header, dict):
        raise ValueError(f"{location.name} has a safetensors header that is not an object")
    return header


def tensor_keys(path: str | Path) -> set[str]:
    """Every tensor name in the file -- the same set `safe_open(...).keys()` returns."""
    return {key for key in read_header(path) if key != "__metadata__"}


def metadata(path: str | Path) -> dict:
    """The file's `__metadata__`, or an empty dict. Mirrors `safe_open(...).metadata() or {}`."""
    value = read_header(path).get("__metadata__")
    return value if isinstance(value, dict) else {}


def has_key_suffix(path: str | Path, suffix: str) -> bool:
    """Whether any tensor name ends with `suffix`. The question all three probes were asking."""
    return any(key.endswith(suffix) for key in tensor_keys(path))


def dtypes(path: str | Path) -> dict[str, str]:
    """`{tensor_name: dtype_string}`, as the pinned LTX tree's own header reader returns."""
    return {key: value["dtype"] for key, value in read_header(path).items() if key != "__metadata__"}
