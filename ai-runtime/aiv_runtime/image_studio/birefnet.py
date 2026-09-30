"""BiRefNet bilateral-reference background-removal pipeline.

BiRefNet (bilateral reference network, `ZhengPeng7/BiRefNet`_) is the
current state-of-the-art dichotomous image segmentation model. It
produces a soft alpha mask which we use as the canonical still-image
matting primitive in this project (and which will later be consumed by
the video-matting service in M5 as a per-clip mask seed).

The implementation wraps the model published on HuggingFace and is
loaded via ``transformers.AutoModelForImageSegmentation`` with
``trust_remote_code=True`` — BiRefNet ships its architecture code in its
repository rather than upstream ``transformers``. This follows the
author's officially documented inference recipe.

Canonical preprocessing from the model card:

* resize to 1024×1024 (the BiRefNet training resolution)
* ``ToTensor`` (uint8 → float32 in [0, 1])
* normalize with ImageNet mean/std

Canonical postprocessing: take the final decoder output, sigmoid it, and
resize back to the original image size. The result is a soft alpha mask
in [0, 1] where 1 = foreground.

.. _ZhengPeng7/BiRefNet: https://huggingface.co/ZhengPeng7/BiRefNet
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image as PILModule

from aiv_runtime.image_studio.common import BgRemovalResult, PILImageType, get_device_type

# BiRefNet's training resolution. Changing this degrades mask quality.
_BIREFNET_INPUT_SIZE = 1024
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
# PIL BILINEAR resample constant. Pillow 10+ exposes this via
# ``Image.Resampling.BILINEAR``; older releases used ``Image.BILINEAR``.
# Both resolve to the same numeric value (2).
_PIL_BILINEAR: int = 2


class BiRefNetPipeline:
    """Thin wrapper around ``ZhengPeng7/BiRefNet`` loaded via
    :func:`transformers.AutoModelForImageSegmentation.from_pretrained`.

    Matches the :class:`BgRemovalPipeline` Protocol so it can be parked
    and evicted by :class:`PipelinesHandler` the same way the image
    generation and image edit pipelines are.
    """

    @staticmethod
    def create(
        model_path: str,
        device: str | None = None,
    ) -> "BiRefNetPipeline":
        return BiRefNetPipeline(model_path=model_path, device=device)

    def __init__(self, model_path: str, device: str | None = None) -> None:
        # BiRefNet's remote code registers the custom segmentation head with
        # transformers' auto factory. trust_remote_code is required because
        # the architecture lives in the model repo, not upstream transformers.
        from transformers import AutoModelForImageSegmentation

        model_factory: Any = AutoModelForImageSegmentation
        model: Any = model_factory.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        model.eval()

        # Recommended by BiRefNet's author — tensor-core friendly matmul
        # precision; safe to set globally on torch.
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

        self._model: Any = model
        self._device: str = "cpu"
        self._dtype: torch.dtype = torch.float32

        if device is not None:
            self.to(device)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def to(self, device: str) -> None:
        runtime_device = get_device_type(device)
        if runtime_device == "cuda":
            # fp16 on CUDA matches the author's reference recipe and halves
            # weight memory with no visible quality regression at 1024².
            self._model.to(runtime_device)
            self._model.half()
            self._dtype = torch.float16
        elif runtime_device == "mps":
            self._model.to(runtime_device)
            # MPS has historically had spotty fp16 support for custom ops;
            # stay in fp32 on MPS to maximise compatibility.
            self._dtype = torch.float32
        else:
            self._model.to(runtime_device)
            self._model.float()
            self._dtype = torch.float32

        self._device = runtime_device

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def remove_background(self, image: PILImageType) -> BgRemovalResult:
        rgb_image = image.convert("RGB")
        tensor = self._preprocess(rgb_image)

        outputs: Any = self._model(tensor)
        logits = self._select_final_output(outputs)
        mask_tensor = torch.sigmoid(logits).float().clamp(0.0, 1.0)

        # Downstream: squeeze to (H, W) and resize back to the original image
        # size so the mask aligns pixel-for-pixel with the input.
        mask_squeezed = mask_tensor[0].squeeze(0).detach().to("cpu")
        mask_pil = self._tensor_to_pil_mask(mask_squeezed, rgb_image.size)

        rgba = self._compose_rgba(rgb_image, mask_pil)
        return BgRemovalResult(rgba=rgba, mask=mask_pil)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _preprocess(self, image: PILImageType) -> torch.Tensor:
        """Resize → tensor → ImageNet-normalize → move to device/dtype.

        We do the preprocessing manually (rather than relying on
        torchvision.transforms) so we don't pull torchvision into the
        service's critical path.
        """
        resized = image.resize(
            (_BIREFNET_INPUT_SIZE, _BIREFNET_INPUT_SIZE),
            _PIL_BILINEAR,
        )
        arr: Any = np.asarray(resized, dtype=np.float32) / 255.0  # (H, W, 3)
        mean: Any = np.array(_IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 3)
        std: Any = np.array(_IMAGENET_STD, dtype=np.float32).reshape(1, 1, 3)
        normalized: Any = (arr - mean) / std

        # ``torch.from_numpy`` is typed as returning ``Tensor`` in torch's stubs
        # but pyright tracks numpy dtypes through the input and occasionally
        # leaks ``Unknown``; use an ``Any`` hop and re-anchor the final tensor.
        raw_tensor: Any = torch.from_numpy(normalized)  # type: ignore[reportUnknownMemberType]
        reshaped: Any = raw_tensor.permute(2, 0, 1).unsqueeze(0).contiguous()
        tensor: torch.Tensor = reshaped
        return tensor.to(device=self._device, dtype=self._dtype)

    @staticmethod
    def _select_final_output(outputs: object) -> torch.Tensor:
        """Extract the final segmentation logits from a BiRefNet forward pass.

        BiRefNet's forward returns a list of decoder outputs — the final
        element is the full-resolution prediction. Some transformers
        wrappers wrap that list inside a ``logits`` attribute, so handle
        both shapes defensively.
        """
        logits_attr: Any = getattr(outputs, "logits", None)
        candidate: Any = logits_attr if logits_attr is not None else outputs

        # Unwrap BiRefNet's list-of-decoder-outputs. We avoid an
        # ``isinstance(candidate, (list, tuple))`` check because pyright
        # narrows the True branch to ``list[Unknown] | tuple[Unknown, ...]``
        # which then leaks ``Unknown`` into the surrounding scope. Duck-type
        # via ``__getitem__`` + ``__len__`` instead, keeping the binding ``Any``.
        if hasattr(candidate, "__len__") and hasattr(candidate, "__getitem__") and not isinstance(candidate, torch.Tensor):
            length: int = int(len(candidate))  # type: ignore[reportUnknownArgumentType]
            if length == 0:
                raise RuntimeError("BiRefNet returned an empty output sequence")
            candidate = candidate[-1]  # type: ignore[reportUnknownVariableType]

        if not isinstance(candidate, torch.Tensor):
            raise RuntimeError(
                f"Unexpected BiRefNet output type: {type(candidate).__name__}"
            )
        return candidate

    @staticmethod
    def _tensor_to_pil_mask(
        tensor: torch.Tensor, target_size: tuple[int, int]
    ) -> PILImageType:
        numpy_array: Any = tensor.numpy()
        array: Any = (numpy_array * 255.0).clip(0.0, 255.0).astype(np.uint8)
        mask: PILImageType = PILModule.fromarray(array, mode="L")
        if mask.size != target_size:
            mask = mask.resize(target_size, _PIL_BILINEAR)
        return mask

    @staticmethod
    def _compose_rgba(rgb_image: PILImageType, mask: PILImageType) -> PILImageType:
        rgba = rgb_image.copy().convert("RGBA")
        rgba.putalpha(mask)
        return rgba
