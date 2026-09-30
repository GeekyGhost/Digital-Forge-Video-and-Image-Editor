"""Z-Image Base/Turbo generation with optional LoRA adapters."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import torch
from diffusers.pipelines.auto_pipeline import (
    ZImagePipeline,  # type: ignore[reportUnknownVariableType]
)
from PIL.Image import Image as PILImage

from aiv_runtime.image_studio.common import ImagePipelineOutputLike, PILImageType, get_device_type
from aiv_runtime.zimage_training import resolve_lora_weights


@dataclass(slots=True)
class _ZImageOutput:
    images: Sequence[PILImageType]


class ZitImageGenerationPipeline:
    @staticmethod
    def create(
        model_path: str,
        device: str | None = None,
        *,
        turbo: bool = True,
        lora_path: str | None = None,
        lora_scale: float = 1.0,
    ) -> ZitImageGenerationPipeline:
        return ZitImageGenerationPipeline(
            model_path=model_path,
            device=device,
            turbo=turbo,
            lora_path=lora_path,
            lora_scale=lora_scale,
        )

    def __init__(
        self,
        model_path: str,
        device: str | None = None,
        *,
        turbo: bool = True,
        lora_path: str | None = None,
        lora_scale: float = 1.0,
    ) -> None:
        self._device: str | None = None
        self._cpu_offload_active = False
        self._turbo = turbo
        self.lora_path: str | None = None
        self.active_adapters: list[str] = []
        self.pipeline = ZImagePipeline.from_pretrained(  # type: ignore[reportUnknownMemberType]
            model_path,
            torch_dtype=torch.bfloat16,
        )
        if lora_path:
            adapter = resolve_lora_weights(lora_path)
            self.pipeline.load_lora_weights(  # type: ignore[reportUnknownMemberType]
                str(adapter.parent), weight_name=adapter.name, adapter_name="creative"
            )
            if hasattr(self.pipeline, "set_adapters"):
                self.pipeline.set_adapters(  # type: ignore[reportUnknownMemberType]
                    ["creative"], adapter_weights=[float(lora_scale)]
                )
            if not hasattr(self.pipeline, "get_active_adapters"):
                raise RuntimeError("this Image Studio runtime cannot verify the active Z-Image LoRA")
            active = list(self.pipeline.get_active_adapters())  # type: ignore[reportUnknownMemberType]
            if "creative" not in active:
                raise RuntimeError("Z-Image LoRA loaded without an active adapter; generation was stopped")
            peft_config = getattr(self.pipeline.transformer, "peft_config", {})
            if "creative" not in peft_config:
                raise RuntimeError("Z-Image LoRA is reported active but no transformer adapter was installed")
            self.lora_path = str(adapter.resolve())
            self.active_adapters = active
        if hasattr(self.pipeline, "vae") and self.pipeline.vae is not None:
            self.pipeline.vae = self.pipeline.vae.to(dtype=torch.bfloat16)
        if device is not None:
            self.to(device)

    def _resolve_generator_device(self) -> str:
        if self._cpu_offload_active:
            return "cuda"
        if self._device is not None:
            return self._device

        execution_device = getattr(self.pipeline, "_execution_device", None)
        return get_device_type(execution_device)

    @staticmethod
    def _normalize_output(output: object) -> ImagePipelineOutputLike:
        images = getattr(output, "images", None)
        if not isinstance(images, Sequence):
            raise TypeError("Unexpected ZIT pipeline output format: missing images sequence")

        images_list = cast(Sequence[object], images)
        validated_images: list[PILImageType] = []
        for image in images_list:
            if not isinstance(image, PILImage):
                raise TypeError("Unexpected ZIT pipeline output format: images must be PIL.Image instances")
            validated_images.append(image)

        return _ZImageOutput(images=validated_images)

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        height: int,
        width: int,
        guidance_scale: float,
        num_inference_steps: int,
        seed: int,
        negative_prompt: str = "",
    ) -> ImagePipelineOutputLike:
        generator = torch.Generator(device=self._resolve_generator_device()).manual_seed(seed)
        pipeline = cast(Any, self.pipeline)
        request: dict[str, Any] = {
            "prompt": prompt,
            "height": height,
            "width": width,
            "guidance_scale": 0.0 if self._turbo else guidance_scale,
            "num_inference_steps": num_inference_steps,
            "generator": generator,
            "output_type": "pil",
            "return_dict": True,
        }
        if not self._turbo:
            request["negative_prompt"] = negative_prompt
        output = pipeline(**request)
        return self._normalize_output(output)

    def release(self) -> None:
        try:
            if hasattr(self.pipeline, "unload_lora_weights"):
                self.pipeline.unload_lora_weights()  # type: ignore[reportUnknownMemberType]
        finally:
            self.to("cpu")

    def to(self, device: str) -> None:
        runtime_device = get_device_type(device)
        if runtime_device in ("cuda", "mps"):
            self.pipeline.enable_sequential_cpu_offload()  # type: ignore[reportUnknownMemberType]
            self._cpu_offload_active = True
        else:
            if self._cpu_offload_active:
                try:
                    if hasattr(self.pipeline, "remove_all_hooks"):
                        self.pipeline.remove_all_hooks()
                except Exception:  # noqa: BLE001, S110 - teardown is deliberately best effort
                    pass
                try:
                    from accelerate.hooks import remove_hook_from_module
                    for module in self.pipeline.components.values():
                        if isinstance(module, torch.nn.Module):
                            remove_hook_from_module(module, recurse=True)
                except Exception:  # noqa: BLE001, S110 - teardown is deliberately best effort
                    pass
                self._cpu_offload_active = False
            self.pipeline.to(runtime_device)  # type: ignore[reportUnknownMemberType]
        self._device = runtime_device

