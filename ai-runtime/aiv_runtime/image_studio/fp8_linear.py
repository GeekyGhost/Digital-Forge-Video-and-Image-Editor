"""FP8-native linear layer for Qwen-Image-Edit-2511.

This module is the diffusers analogue of ComfyUI's
``comfy.ops.Linear`` for FP8 checkpoints. The standard diffusers
loader casts FP8 weights to bf16 the moment they enter an
``nn.Linear``, doubling the memory footprint (~32 GB) and forcing
``enable_sequential_cpu_offload`` on a 15 GB consumer GPU. Replacing
the linears in the Qwen transformer with :class:`FP8Linear` keeps
weights in ``torch.float8_e4m3fn`` (~14.5 GB resident) and runs the
matmul through ``torch._scaled_mm`` on Ada Lovelace + Hopper GPUs,
which have native FP8 GEMM tensor cores.

Forward strategy
----------------

* ``_forward_scaled_mm`` (Ada+ on CUDA): per-tensor quantize the bf16
  input to FP8 e4m3fn, run ``torch._scaled_mm(input_fp8,
  weight_fp8.t(), scale_a, scale_b)``. ~2x throughput vs bf16 matmul.
* ``_forward_dequant`` (older GPUs / CPU): dequantize the weight to
  the input dtype (bf16) per-forward, run standard ``F.linear``. No
  speedup, but the model still fits in memory at FP8.

Compatibility
-------------

:class:`FP8Linear` subclasses :class:`torch.nn.Linear` so existing
diffusers LoRA hooks (which read ``in_features`` / ``out_features``
and patch ``forward``) keep working.
"""

from __future__ import annotations

import logging
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_SCALED_MM_AVAILABLE = hasattr(torch, "_scaled_mm")
_FP8_E4M3_MAX = 448.0  # Max representable value in float8_e4m3fn


class FP8Linear(nn.Linear):
    """A Linear layer that stores weights in FP8 and computes in FP8
    if the hardware supports it, falling back to BF16 dequantization
    on the fly if not.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        # Default weight scale is 1.0 (TensorWise)
        self.register_buffer(
            "weight_scale",
            torch.tensor(1.0, dtype=torch.float32, device=device),
            persistent=True,
        )

    def _forward_dequant(self, input: torch.Tensor) -> torch.Tensor:
        """Fallback path: dequantize weight to input dtype and use F.linear."""
        # Move weight scale to input dtype
        ws = cast(torch.Tensor, self.weight_scale).to(input.dtype)
        
        # Cast weight up to input dtype (e.g. bfloat16) and apply scale
        if ws.numel() == 1:
            weight = self.weight.to(input.dtype) * ws
        else:
            weight = self.weight.to(input.dtype) * ws.view(-1, 1)

        return F.linear(input, weight, self.bias)

    def _forward_scaled_mm(self, input: torch.Tensor) -> torch.Tensor:
        """Hardware-accelerated FP8 path using torch._scaled_mm."""
        original_shape = input.shape
        input_2d = input.reshape(-1, self.in_features).contiguous()

        # 1. TensorWise scaling for input (Scalar) - Fully supported on RTX 4090
        amax = input_2d.abs().amax().clamp(min=1e-12)
        input_scale = (amax / _FP8_E4M3_MAX).float().reshape(())
        input_fp8 = (input_2d / input_scale).to(torch.float8_e4m3fn)

        # 2. Bypass Ada hardware limitation on RowWise scales
        # We pass a scalar 1.0 to keep the hardware kernel happy.
        dummy_scale_b = torch.tensor(1.0, device=input.device, dtype=torch.float32)

        # 3. Hardware-accelerated TensorWise FP8 Matmul
        # CRITICAL ARTIFACT FIX: Force output to float32 to preserve pristine precision
        try:
            out_fp32 = torch._scaled_mm(  # type: ignore[attr-defined]
                input_fp8,
                self.weight.t(),
                scale_a=input_scale,
                scale_b=dummy_scale_b,
                bias=None,  # MUST be None. We apply bias manually AFTER scaling.
                out_dtype=torch.float32,  # No premature rounding!
            )
        except Exception as exc:
            # If _scaled_mm fails for any other hardware reason, fallback safely
            logger.warning("FP8Linear: _scaled_mm failed (%s); falling back to dequant for this layer", exc)
            return self._forward_dequant(input)

        # 4. Apply the RowWise weight scales mathematically in full float32 precision
        ws = cast(torch.Tensor, self.weight_scale).to(torch.float32)
        if ws.numel() == 1:
            out_fp32 = out_fp32 * ws
        else:
            out_fp32 = out_fp32 * ws.view(1, -1)

        # 5. Apply the bias manually at the very end
        if self.bias is not None:
            out_fp32 = out_fp32 + self.bias.to(torch.float32).view(1, -1)

        # Cast back to the required precision at the very end
        out = out_fp32.to(input.dtype)
        return out.view(*original_shape[:-1], self.out_features)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not _SCALED_MM_AVAILABLE or not input.is_cuda or self.weight.dtype != torch.float8_e4m3fn:
            return self._forward_dequant(input)
        
        return self._forward_scaled_mm(input)


def replace_linears_with_fp8(module: nn.Module, fp8_state: Any, prefix: str = "") -> dict[str, int]:
    """Recursively replace nn.Linear modules with FP8Linear where weights exist.
    
    Supports the LazySafetensors stream loader from QwenImageEditPipeline.
    """
    counts = {"replaced": 0, "skipped_no_fp8_weight": 0, "skipped_not_fp8": 0}

    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name

        if isinstance(child, nn.Linear):
            weight_key = f"{full_name}.weight"
            
            if weight_key not in fp8_state:
                counts["skipped_no_fp8_weight"] += 1
                continue
            
            fp8_weight = fp8_state[weight_key]  # triggers streaming read
            
            if fp8_weight.dtype != torch.float8_e4m3fn:
                counts["skipped_not_fp8"] += 1
                continue

            scale_key = f"{full_name}.weight_scale"
            scale_raw = fp8_state.get(scale_key)

            if scale_raw is None:
                scale_value = torch.tensor(1.0, dtype=torch.float32)
            else:
                # Normalize scale shape for safe broadcasting during ``forward``.
                scale_flat = scale_raw.float().contiguous()
                if scale_flat.numel() == 1:
                    scale_value = scale_flat.reshape(())
                else:
                    scale_value = scale_flat.reshape(-1)

            bias_key = f"{full_name}.bias"
            bias_tensor = fp8_state.get(bias_key)

            new_layer = FP8Linear(
                in_features=child.in_features,
                out_features=child.out_features,
                bias=bias_tensor is not None,
                device="meta",
                dtype=torch.bfloat16,
            )
            
            # Copy weights via parameter assignment (no in-place copy
            # that would error on FP8 vs FP8 dtype mismatches).
            new_layer.weight = nn.Parameter(
                fp8_weight.contiguous(),
                requires_grad=False,
            )
            
            # Replace the buffer.
            del new_layer.weight_scale
            new_layer.register_buffer("weight_scale", scale_value, persistent=True)

            if bias_tensor is not None and new_layer.bias is not None:
                new_layer.bias = nn.Parameter(
                    bias_tensor.to(torch.bfloat16).contiguous(),
                    requires_grad=False,
                )

            setattr(module, name, new_layer)
            counts["replaced"] += 1
        else:
            sub_counts = replace_linears_with_fp8(child, fp8_state, full_name)
            for key, val in sub_counts.items():
                counts[key] = counts.get(key, 0) + val

    return counts
