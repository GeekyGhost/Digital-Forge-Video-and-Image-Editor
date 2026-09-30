"""Small dependency-free contracts shared by the imported image pipelines."""

from __future__ import annotations

import json
import mmap
import struct
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, TypeAlias

import torch
from PIL.Image import Image as PILImage

PILImageType: TypeAlias = PILImage


class ImagePipelineOutputLike(Protocol):
    images: Sequence[PILImageType]


@dataclass(slots=True, frozen=True)
class BgRemovalResult:
    rgba: PILImageType
    mask: PILImageType


def get_device_type(device: str | torch.device | object | None) -> str:
    if device is None:
        return "cpu"
    device_type = getattr(device, "type", None)
    if isinstance(device_type, str):
        return device_type
    if isinstance(device, str):
        try:
            return str(torch.device(device).type)
        except Exception:
            return device
    return "cpu"


_DTYPES = {
    "F64": torch.float64, "F32": torch.float32, "F16": torch.float16,
    "BF16": torch.bfloat16, "I64": torch.int64, "I32": torch.int32,
    "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8,
    "BOOL": torch.bool, "F8_E4M3": torch.float8_e4m3fn, "F8_E5M2": torch.float8_e5m2,
}


def load_safetensors_direct(path: str) -> dict[str, torch.Tensor]:
    """Load through Python mmap and retain mappings on the returned tensors."""
    handle = open(path, "rb")
    mapping = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
    view = memoryview(mapping)
    header_size = struct.unpack("<Q", view[:8])[0]
    header = json.loads(bytes(view[8:8 + header_size]).decode("utf-8"))
    data = view[8 + header_size:]
    refs = (mapping, view, data, handle)
    tensors: dict[str, torch.Tensor] = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        start, end = info["data_offsets"]
        if start == end:
            tensor = torch.empty(info["shape"], dtype=_DTYPES[info["dtype"]])
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="The given buffer is not writable")
                tensor = torch.frombuffer(data[start:end], dtype=_DTYPES[info["dtype"]]).reshape(info["shape"])
            setattr(tensor, "_aiv_mmap_refs", refs)
            setattr(tensor.untyped_storage(), "_aiv_mmap_refs", refs)
        tensors[name] = tensor
    return tensors
