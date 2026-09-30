"""A CPU-resident, quantised reference K/V cache for Wan-Animate-2 (VFX Phase 5).

Animate-2's generation pass attends over every layer's reference K/V, which Diffusers 0.40 keeps on
the GPU in `WanAnimate2KVCache` -- tens of GiB at the upstream 800x640 / 81-frame preset. This cache
has the same interface (`get(layer).store / get / clear`, `clear()`) and keeps each inactive layer in
pinned CPU memory as INT8 (per head vector) or INT4 (groups of 64), restoring only the layer being
computed and prefetching the next one on a side stream.

The pattern is copied from the Digital Forge Spritesheet Maker's runtime (its own copy, nothing
shared); the worker installs it by replacing the name `WanAnimate2KVCache` in Diffusers' modular
denoise module, so the pipeline still allocates a fresh cache per segment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


def _dynamo_disable(function):
    """Keep cache transfers outside compiled graphs."""
    compiler = getattr(torch, "compiler", None)
    disable = getattr(compiler, "disable", None)
    if disable is None:
        disable = getattr(getattr(torch, "_dynamo", None), "disable", None)
    return disable(function) if disable is not None else function


def _maybe_pin(tensor: torch.Tensor, enabled: bool) -> torch.Tensor:
    if not enabled or not torch.cuda.is_available():
        return tensor
    try:
        return tensor.pin_memory()
    except RuntimeError:
        return tensor


CODECS = ("int8", "int4", "bf16")


@dataclass
class StoredTensor:
    data: torch.Tensor
    scale: Optional[torch.Tensor]
    shape: tuple
    dtype: torch.dtype
    device: torch.device
    codec: str
    group_size: int = 0

    @property
    def byte_count(self) -> int:
        total = self.data.numel() * self.data.element_size()
        if self.scale is not None:
            total += self.scale.numel() * self.scale.element_size()
        return total

    @classmethod
    def capture(cls, tensor: torch.Tensor, codec: str, pin_memory: bool) -> "StoredTensor":
        source = tensor.detach()
        shape, dtype, device = tuple(source.shape), source.dtype, source.device
        if codec == "bf16":
            return cls(_maybe_pin(source.to(device="cpu", dtype=torch.bfloat16), pin_memory), None, shape, dtype, device, codec)
        if codec == "int8":
            # One scale per head vector: bounds an outlier's damage without a scale per element.
            scale = source.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 127.0
            data = torch.round(source / scale).clamp(-127, 127).to(torch.int8)
            return cls(_maybe_pin(data.to("cpu"), pin_memory), _maybe_pin(scale.to(device="cpu", dtype=torch.float16), pin_memory),
                       shape, dtype, device, codec)
        if codec != "int4":
            raise ValueError(f"unsupported cache codec {codec!r}; use one of {CODECS}")
        group_size = min(64, shape[-1])
        group_size += group_size % 2
        groups = (shape[-1] + group_size - 1) // group_size
        padded = groups * group_size
        if padded != shape[-1]:
            source = torch.nn.functional.pad(source, (0, padded - shape[-1]))
        grouped = source.reshape(*shape[:-1], groups, group_size)
        scale = grouped.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 7.0
        encoded = (torch.round(grouped / scale).clamp(-7, 7).to(torch.int8) + 8).to(torch.uint8)
        packed = encoded[..., 0::2] | (encoded[..., 1::2] << 4)
        return cls(_maybe_pin(packed.to("cpu"), pin_memory), _maybe_pin(scale.to(device="cpu", dtype=torch.float16), pin_memory),
                   shape, dtype, device, codec, group_size)

    def restore(self, non_blocking: bool = True) -> torch.Tensor:
        if self.codec == "bf16":
            return self.data.to(self.device, dtype=self.dtype, non_blocking=non_blocking)
        data = self.data.to(self.device, non_blocking=non_blocking)
        scale = self.scale.to(self.device, non_blocking=non_blocking)
        if self.codec == "int8":
            return (data.to(self.dtype) * scale.to(self.dtype)).reshape(self.shape)
        unpacked = torch.stack((data & 0x0F, data >> 4), dim=-1).flatten(-2)
        restored = (unpacked.to(torch.int8) - 8).to(self.dtype) * scale.to(self.dtype)
        return restored.flatten(-2)[..., : self.shape[-1]].reshape(self.shape)


class StreamingKVLayerCache:
    def __init__(self, owner: "StreamingKVCache", index: int):
        self.owner = owner
        self.index = index
        self.key: Optional[StoredTensor] = None
        self.value: Optional[StoredTensor] = None
        self.prefetched: Optional[tuple] = None

    @_dynamo_disable
    def store(self, key: torch.Tensor, value: torch.Tensor):
        self.key = StoredTensor.capture(key, self.owner.codec, self.owner.pin_memory)
        self.value = StoredTensor.capture(value, self.owner.codec, self.owner.pin_memory)
        self.owner.stored_bytes += self.key.byte_count + self.value.byte_count

    @_dynamo_disable
    def get(self):
        return self.owner.take(self.index)

    @_dynamo_disable
    def clear(self):
        self.key = self.value = self.prefetched = None


class StreamingKVCache:
    """Drop-in for Diffusers' `WanAnimate2KVCache(num_layers)`."""

    #: The largest cache any instance held, for the worker's stats.
    peak_bytes = 0

    def __init__(self, num_layers: int, codec: str = "int8", pin_memory: bool = True, prefetch: bool = True):
        codec = codec.lower()
        if codec not in CODECS:
            raise ValueError(f"unsupported cache codec {codec!r}; use one of {CODECS}")
        self.codec = codec
        self.pin_memory = bool(pin_memory)
        self.prefetch = bool(prefetch)
        self.layer_caches = [StreamingKVLayerCache(self, index) for index in range(num_layers)]
        self.stored_bytes = 0
        self._stream = None

    def get(self, layer_idx: int) -> StreamingKVLayerCache:
        return self.layer_caches[layer_idx]

    def _restore(self, index: int):
        layer = self.layer_caches[index]
        if layer.key is None or layer.value is None:
            raise RuntimeError("the reference K/V cache is empty: the reference pass did not run")
        return layer.key.restore(), layer.value.restore()

    def _prefetch(self, index: int):
        layer = self.layer_caches[index]
        if layer.prefetched is not None or layer.key is None or layer.key.device.type != "cuda":
            return
        if self._stream is None:
            with torch.cuda.device(layer.key.device):
                self._stream = torch.cuda.Stream()
        with torch.cuda.stream(self._stream):
            layer.prefetched = self._restore(index)

    @_dynamo_disable
    def take(self, index: int):
        StreamingKVCache.peak_bytes = max(StreamingKVCache.peak_bytes, self.stored_bytes)
        layer = self.layer_caches[index]
        if layer.prefetched is not None:
            current = torch.cuda.current_stream(layer.key.device)
            current.wait_stream(self._stream)
            key, value = layer.prefetched
            layer.prefetched = None
            key.record_stream(current)
            value.record_stream(current)
        else:
            key, value = self._restore(index)
        if self.prefetch and len(self.layer_caches) > 1:
            self._prefetch((index + 1) % len(self.layer_caches))
        return key, value

    @_dynamo_disable
    def clear(self):
        if self._stream is not None:
            self._stream.synchronize()
        for layer in self.layer_caches:
            layer.clear()
        self.stored_bytes = 0
        self._stream = None
