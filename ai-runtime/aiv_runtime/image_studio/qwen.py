"""Qwen-Image-Edit-2511 pipeline wrapper"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn
from PIL.Image import Image as PILImage

from aiv_runtime.image_studio.common import ImagePipelineOutputLike, PILImageType, get_device_type

logger = logging.getLogger(__name__)
logging.getLogger("aiv_runtime.image_studio.fp8_linear").setLevel(logging.ERROR)


# Diffusers' VAE_IMAGE_SIZE: every input is resized to this area before the VAE,
# and it is also the area the output size is derived from.
QWEN_VAE_AREA = 1024 * 1024


def canvas_output_size(source: PILImage) -> tuple[int, int]:
    """Output width/height for an edit, taken from the image being edited.

    QwenImageEditPlusPipeline defaults the output size to ``image[-1].size`` --
    the LAST reference in the list -- when height/width are not supplied. For an
    edit that is the wrong end: a portrait reference would silently reshape a
    landscape canvas. We reproduce the pipeline's own ``calculate_dimensions``
    against image[0] instead and pass the result in explicitly.
    """
    width, height = source.size
    ratio = (width / height) if height else 1.0
    exact_width = math.sqrt(QWEN_VAE_AREA * ratio)
    exact_height = exact_width / ratio
    return (
        max(32, round(exact_width / 32) * 32),
        max(32, round(exact_height / 32) * 32),
    )


# ── attention helpers ───────────────────────────────────────────────
_SAGE_WARNING_EMITTED = False


def is_sage_active() -> bool:
    import torch.nn.functional as F  # type: ignore[reportShadowedImports]
    current = getattr(F, "scaled_dot_product_attention", None)
    return current is not None and hasattr(current, "_original_sdpa")


@contextmanager
def vanilla_sdpa() -> Iterator[None]:
    global _SAGE_WARNING_EMITTED
    import torch.nn.functional as F  # type: ignore[reportShadowedImports]
    current = getattr(F, "scaled_dot_product_attention", None)
    original = getattr(current, "_original_sdpa", None)
    if original is None:
        yield
        return

    if not _SAGE_WARNING_EMITTED:
        logger.warning(
            "Sage attention detected on F.scaled_dot_product_attention; "
            "bypassing for Qwen-Image-Edit (model produces artifacts under "
            "sage/flash attention)."
        )
        _SAGE_WARNING_EMITTED = True

    F.scaled_dot_product_attention = original
    try:
        yield
    finally:
        F.scaled_dot_product_attention = current


@contextmanager
def disable_flash_attention() -> Iterator[None]:
    if not torch.cuda.is_available():
        yield
        return
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel  # type: ignore[attr-defined]
    except ImportError:
        yield
        return
    # EFFICIENT first with an explicit priority order: the memory-efficient
    # fused kernel is numerically exact (unlike sage/flash on this model) and
    # never materializes the full attention matrix. If MATH is ever picked at
    # ~5k joint tokens it allocates multi-GB score tensors per layer — on a
    # card already holding the 12 GB resident transformer that tips VRAM into
    # WDDM sysmem paging and denoise slows ~40x (2026-07-03: 362s/step).
    try:
        ctx = sdpa_kernel(
            [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH],
            set_priority_order=True,
        )
    except TypeError:  # older torch without set_priority_order
        ctx = sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH])
    with ctx:
        yield


# ── single-file detection ───────────────────────────────────────────
def _looks_like_single_file(model_path: str) -> bool:
    p = Path(model_path)
    return p.is_file() and p.suffix.lower() == ".safetensors"


def _valid_qwen_diffusers_folder(path: Path) -> bool:
    """Return True only for a complete local Diffusers pipeline folder.

    Some installs have a sibling ``Qwen-Image-Edit-2511`` directory that
    contains only raw safetensor shards. Passing that folder to Diffusers
    makes it fail with "no file named config.json" before it can fall back
    to the canonical repo configs.
    """
    return (
        path.is_dir()
        and (path / "model_index.json").is_file()
        and (path / "transformer" / "config.json").is_file()
    )


# ── diffusers import (lazy, optional) ───────────────────────────────
try:
    from diffusers import QwenImageEditPlusPipeline as _QwenImageEditPlusPipelineImported  # type: ignore[attr-defined]
except Exception as _import_exc:  # pragma: no cover
    QwenImageEditPlusPipeline: Any = None
    _QWEN_EDIT_IMPORT_ERROR: BaseException | None = _import_exc
else:
    QwenImageEditPlusPipeline = _QwenImageEditPlusPipelineImported
    _QWEN_EDIT_IMPORT_ERROR = None


# ── zero-copy pointer helper ────────────────────────────────────────
def _zero_copy_swap_weights(target: torch.nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    """Zero-copy swap module weights in-place via tensor.set_() pointers."""
    with torch.no_grad():
        for name, param in target.named_parameters():
            if name in state_dict:
                src = state_dict[name]
                try:
                    src = src.to(device=param.device, dtype=param.dtype, non_blocking=True)
                    param.data = src
                    mmap_refs = getattr(src, "_ltx_tensor_mmap_refs", None)
                    if mmap_refs is not None:
                        setattr(param, "_ltx_tensor_mmap_refs", mmap_refs)
                        setattr(param.data, "_ltx_tensor_mmap_refs", mmap_refs)
                except Exception as exc:
                    logger.warning("Zero-copy parameter swap failed for %s, falling back to copy: %s", name, exc)
                    param.copy_(src)

        for name, buf in target.named_buffers():
            if name in state_dict:
                src = state_dict[name]
                try:
                    src = src.to(device=buf.device, dtype=buf.dtype, non_blocking=True)
                    buf.data = src
                    mmap_refs = getattr(src, "_ltx_tensor_mmap_refs", None)
                    if mmap_refs is not None:
                        setattr(buf, "_ltx_tensor_mmap_refs", mmap_refs)
                except Exception as exc:
                    logger.warning("Zero-copy buffer swap failed for %s, falling back to copy: %s", name, exc)
                    buf.copy_(src)


def _assign_tensor_to_module(root: nn.Module, name: str, tensor: torch.Tensor) -> bool:
    """Replace a parameter or buffer by name without materializing meta storage."""
    parent_name, _, leaf_name = name.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root

    if leaf_name in parent._parameters:
        parent._parameters[leaf_name] = nn.Parameter(tensor.contiguous(), requires_grad=False)
        return True

    if leaf_name in parent._buffers:
        parent._buffers[leaf_name] = tensor.contiguous()
        return True

    return False


def _assert_no_meta_tensors(module: nn.Module, *, context: str) -> None:
    meta_parameters = [name for name, param in module.named_parameters() if param.is_meta]
    meta_buffers = [name for name, buf in module.named_buffers() if buf.is_meta]
    meta_attrs: list[str] = []
    for module_name, child in module.named_modules():
        for attr_name, value in vars(child).items():
            if isinstance(value, torch.Tensor) and value.is_meta:
                meta_attrs.append(f"{module_name}.{attr_name}" if module_name else attr_name)

    if not meta_parameters and not meta_buffers and not meta_attrs:
        return

    preview = ", ".join([*meta_parameters[:5], *meta_buffers[:5], *meta_attrs[:5]])
    raise RuntimeError(
        f"{context} left {len(meta_parameters)} parameters, {len(meta_buffers)} buffers, "
        f"and {len(meta_attrs)} tensor attributes on meta device"
        + (f": {preview}" if preview else "")
    )


# ── FP8 streaming loader ────────────────────────────────────────────
def _materialize_qwen_rope_tensors(transformer: nn.Module) -> None:
    """Rebuild Qwen RoPE tensor attributes created under meta init."""
    for module in transformer.modules():
        pos_freqs = getattr(module, "pos_freqs", None)
        neg_freqs = getattr(module, "neg_freqs", None)
        if not (
            isinstance(pos_freqs, torch.Tensor)
            and isinstance(neg_freqs, torch.Tensor)
            and (pos_freqs.is_meta or neg_freqs.is_meta)
            and hasattr(module, "rope_params")
            and hasattr(module, "axes_dim")
            and hasattr(module, "theta")
        ):
            continue

        axes_dim = list(getattr(module, "axes_dim"))
        theta = int(getattr(module, "theta"))
        rope_params = cast(Callable[[torch.Tensor, int, int], torch.Tensor], getattr(module, "rope_params"))
        pos_index = torch.arange(4096)
        neg_index = torch.arange(4096).flip(0) * -1 - 1
        module.pos_freqs = torch.cat(
            [
                rope_params(pos_index, axes_dim[0], theta),
                rope_params(pos_index, axes_dim[1], theta),
                rope_params(pos_index, axes_dim[2], theta),
            ],
            dim=1,
        )
        module.neg_freqs = torch.cat(
            [
                rope_params(neg_index, axes_dim[0], theta),
                rope_params(neg_index, axes_dim[1], theta),
                rope_params(neg_index, axes_dim[2], theta),
            ],
            dim=1,
        )


def _load_fp8_native_transformer(*, fp8_path: str, config_locator: str) -> tuple[Any, int]:
    """Load FP8 transformer via streaming safetensors read.

    Returns (transformer, fp8_replacement_count) so the caller can
    decide whether the FP8 fast path is genuinely active.
    """
    from diffusers import QwenImageTransformer2DModel  # type: ignore[attr-defined]
    from accelerate import init_empty_weights  # type: ignore[reportMissingImports]
    from aiv_runtime.image_studio.common import load_safetensors_direct
    from aiv_runtime.image_studio.fp8_linear import _SCALED_MM_AVAILABLE, replace_linears_with_fp8

    logger.info("QwenImageEditPipeline: streaming FP8 transformer load…")

    config = QwenImageTransformer2DModel.load_config(config_locator, subfolder="transformer")  # type: ignore
    with init_empty_weights(include_buffers=True):
        transformer: Any = QwenImageTransformer2DModel.from_config(config)  # type: ignore
    _materialize_qwen_rope_tensors(transformer)

    fp8_prefixes: set[str] = set()
    fp8_count = 0

    tensors = load_safetensors_direct(fp8_path)
    all_keys = set(tensors.keys())
    has_scales = any(k.endswith(".weight_scale") for k in all_keys)

    for k in all_keys:
        if k.endswith(".weight_scale") or k.endswith(".bias"):
            continue
        tensor = tensors[k]
        if tensor.dtype == torch.float8_e4m3fn:
            fp8_count += 1
            fp8_prefixes.add(k.rsplit(".", 1)[0])

    logger.info(
        "QwenImageEditPipeline: FP8-native load (scaled=%s, %d FP8 / %d total, "
        "scaled_mm_available=%s)",
        has_scales, fp8_count, len(all_keys), _SCALED_MM_AVAILABLE,
    )

    non_fp8_state: dict[str, torch.Tensor] = {}
    for k in all_keys:
        if k.endswith(".weight_scale"):
            continue
        prefix = k.rsplit(".", 1)[0]
        if prefix in fp8_prefixes:
            continue
        tensor = tensors[k]
        non_fp8_state[k] = tensor.to(torch.bfloat16) if tensor.is_floating_point() else tensor

    missing_non_fp8: list[str] = []
    for name, tensor in non_fp8_state.items():
        if not _assign_tensor_to_module(transformer, name, tensor):
            missing_non_fp8.append(name)
    if missing_non_fp8:
        logger.warning(
            "QwenImageEditPipeline: %d non-FP8 tensors were not present in the transformer module; first=%s",
            len(missing_non_fp8),
            missing_non_fp8[:5],
        )

    class LazySafetensors:
        def __contains__(self, key: str) -> bool: return key in all_keys
        def __getitem__(self, key: str) -> torch.Tensor: return tensors[key]
        def get(self, key: str, default: Any = None) -> Any:
            return tensors.get(key, default)

    counts = replace_linears_with_fp8(transformer, LazySafetensors())  # type: ignore
    replaced = int(counts.get("replaced", 0))
    _assert_no_meta_tensors(transformer, context="Qwen FP8 transformer load")

    logger.info("QwenImageEditPipeline: FP8 replaced %d nn.Linear modules", replaced)
    return transformer, replaced


# ── output dataclass ────────────────────────────────────────────────
@dataclass(slots=True)
class _QwenImageEditOutput:
    images: Sequence[PILImageType]


# ── VRAM wrapper for encoders ───────────────────────────────────────
class VRAMWrapper:
    """Move auxiliary module to GPU for one call, then evict to CPU.

    Intentionally avoids torch.cuda.empty_cache() except under heavy pressure.
    """

    def __init__(self, original_method: Any, module: Any, device: str) -> None:
        self.original_method = original_method
        self.module = module
        self.device = device
        self._is_on_gpu = False

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        import torch
        if not self._is_on_gpu:
            self.module.to(self.device, non_blocking=True)
            self._is_on_gpu = True

        def move_to(obj: Any) -> Any:
            if isinstance(obj, torch.Tensor):
                return obj.to(self.device, non_blocking=True)
            if isinstance(obj, dict):
                return {k: move_to(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return type(obj)(move_to(v) for v in obj)
            return obj

        args = move_to(args)
        kwargs = move_to(kwargs)

        try:
            return self.original_method(*args, **kwargs)
        finally:
            import gc
            if torch.cuda.is_available():
                self.module.to("cpu")
                self._is_on_gpu = False
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()


# ── public pipeline class ───────────────────────────────────────────
class QwenImageEditPipeline:
    # public status flags consumed by the handler / status route
    lightning_lora_loaded: bool
    fp8_native_active: bool

    @staticmethod
    def create(
        model_path: str,
        device: str | None = None,
        lora_path: str | None = None,
        lora_scale: float = 1.0,
        vae_path: str | None = None,
        text_encoder_path: str | None = None,
    ) -> "QwenImageEditPipeline":
        return QwenImageEditPipeline(
            model_path=model_path,
            device=device,
            lora_path=lora_path,
            lora_scale=lora_scale,
            vae_path=vae_path,
            text_encoder_path=text_encoder_path,
        )

    def __init__(
        self,
        model_path: str,
        device: str | None = None,
        lora_path: str | None = None,
        lora_scale: float = 1.0,
        vae_path: str | None = None,
        text_encoder_path: str | None = None,
    ) -> None:
        if QwenImageEditPlusPipeline is None:
            raise RuntimeError(
                f"diffusers.QwenImageEditPlusPipeline unavailable: {_QWEN_EDIT_IMPORT_ERROR}"
            )

        self._device: str | None = None
        self._cpu_offload_active = False
        self._group_offload_applied = False
        self.lightning_lora_loaded = False
        self.fp8_native_active = False

        single_file = _looks_like_single_file(model_path)

        if single_file:
            import os
            configured_base = os.environ.get("AIV_QWEN_BASE_MODEL", "").strip()
            base_path = Path(configured_base) if configured_base else Path(model_path).parent / "Qwen-Image-Edit-2511"
            if _valid_qwen_diffusers_folder(base_path):
                base_locator = str(base_path)
            else:
                if base_path.is_dir():
                    logger.warning(
                        "QwenImageEditPipeline: ignoring incomplete local config folder %s; "
                        "falling back to Qwen/Qwen-Image-Edit-2511 for Diffusers configs.",
                        base_path,
                    )
                base_locator = "Qwen/Qwen-Image-Edit-2511"
            transformer, replaced = _load_fp8_native_transformer(
                fp8_path=model_path, config_locator=base_locator,
            )
            self.fp8_native_active = replaced > 0
            self.pipeline = QwenImageEditPlusPipeline.from_pretrained(  # type: ignore
                base_locator, transformer=transformer, torch_dtype=torch.bfloat16,
            )
        else:
            self.pipeline = QwenImageEditPlusPipeline.from_pretrained(  # type: ignore
                model_path, torch_dtype=torch.bfloat16,
            )

        # Keep the per-step denoise progress bar visible: with leaf_level group
        # offloading each step is slow, and hiding the bar makes a slow run look
        # like a hang (nothing prints between the offload log and completion).
        self.pipeline.set_progress_bar_config(disable=False)

        if vae_path:
            self._swap_component_weights("vae", vae_path)
        if text_encoder_path:
            self._swap_component_weights("text_encoder", text_encoder_path)

        if lora_path:
            self._load_lightning_lora(lora_path, lora_scale)

        if device is not None:
            self.to(device)

    # ── internal helpers ────────────────────────────────────────────
    def _load_lightning_lora(self, lora_path: str, lora_scale: float) -> None:
        lora_file = Path(lora_path)
        if not (lora_file.exists() and lora_file.is_file()):
            logger.error(
                "Lightning LoRA path does not exist: %s — defaults of 4 steps "
                "will produce garbage. User should download it from Model Status "
                "or raise Steps to 40+.",
                lora_path,
            )
            return

        pipeline: Any = self.pipeline
        adapter_name = "lightning"
        try:
            pipeline.load_lora_weights(  # type: ignore
                str(lora_file.parent),
                weight_name=lora_file.name,
                adapter_name=adapter_name,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
        except Exception as exc:
            logger.error("Lightning LoRA load failed: %s", exc, exc_info=True)
            return

        # Cast adapter modules to bf16
        has_lora_modules = False
        for name, module in pipeline.transformer.named_modules():
            if "lora" in name.lower():
                has_lora_modules = True
                module.to(dtype=torch.bfloat16)

        if hasattr(pipeline, "set_adapters"):
            try:
                pipeline.set_adapters([adapter_name], [lora_scale])  # type: ignore
            except Exception as exc:
                logger.warning("set_adapters(%s, scale=%s) failed: %s", adapter_name, lora_scale, exc)

        active: list[str] = []
        if hasattr(pipeline, "get_active_adapters"):
            try:
                active = list(pipeline.get_active_adapters())  # type: ignore
            except Exception as exc:
                logger.error("get_active_adapters failed: %s", exc)

        if not active and has_lora_modules:
            active = [adapter_name]

        if not active:
            logger.error(
                "Lightning LoRA load returned without raising but no active "
                "adapters are reported. Treating as failed."
            )
            return

        self.lightning_lora_loaded = True
        logger.info("Lightning LoRA loaded; adapters=%s", active)

    def _swap_component_weights(self, component: str, weights_path: str) -> None:
        """Swap a sub-component's weights. Re-raises on failure."""
        try:
            from aiv_runtime.image_studio.common import load_safetensors_direct
            target = getattr(self.pipeline, component, None)
            if target is None:
                raise RuntimeError(f"Pipeline has no attribute '{component}'")
            raw = load_safetensors_direct(weights_path)
            prefix = f"{component}."
            normalized = {(k[len(prefix):] if k.startswith(prefix) else k): v for k, v in raw.items()}
            _zero_copy_swap_weights(target, normalized)
        except Exception as exc:
            logger.error("Failed to swap %s weights from %s: %s", component, weights_path, exc, exc_info=True)
            raise RuntimeError(f"Failed to swap {component} weights: {exc}") from exc

    def _resolve_generator_device(self) -> str:
        """Generator must sit on the *execution* device. Under group
        offloading the modules rest on CPU but diffusers reports the onload
        device via ``_execution_device``; the configured runtime device is
        the authoritative source here.
        """
        if self._device and self._device not in ("cpu", "mps"):
            return self._device
        try:
            return str(self.pipeline._execution_device)
        except Exception:
            return self._device or "cpu"

    @staticmethod
    def _normalize_output(output: object) -> ImagePipelineOutputLike:
        images = getattr(output, "images", None)
        images_list = cast(Sequence[object], images)
        validated_images = [img for img in images_list if isinstance(img, PILImage)]
        return _QwenImageEditOutput(images=validated_images)

    @staticmethod
    def _coerce_reference_images(images: Sequence[PILImage]) -> list[PILImage] | PILImage:
        return images[0] if len(images) == 1 else list(images)

    # ── public API ──────────────────────────────────────────────────
    @torch.inference_mode()
    def edit(
        self,
        prompt: str,
        images: Sequence[PILImage],
        negative_prompt: str,
        num_inference_steps: int,
        true_cfg_scale: float,
        seed: int,
        step_callback: Callable[[int, int], None] | None = None,
        output_size: tuple[int, int] | None = None,
    ) -> ImagePipelineOutputLike:
        import gc
        import faulthandler
        import sys
        import threading
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        if not images:
            raise ValueError("Qwen Image Edit needs at least one input image")
        reference = self._coerce_reference_images(images)
        # Shape follows the image being edited, never the last reference -- unless an exact size
        # was asked for (already snapped to 16 by the worker's `edit_output_size`).
        out_width, out_height = output_size or canvas_output_size(images[0])
        generator = torch.Generator(device=self._resolve_generator_device()).manual_seed(seed)
        pipeline: Any = self.pipeline

        # Diagnostic: record what we're feeding the denoiser. Large reference
        # images mean long token sequences, which is the prime suspect if the
        # attention backend is the stall point.
        try:
            _ref_list = reference if isinstance(reference, (list, tuple)) else [reference]
            _ref_sizes = [getattr(im, "size", None) for im in _ref_list]
        except Exception:
            _ref_sizes = None
        logger.info(
            "QwenImageEditPipeline.generate: steps=%s true_cfg=%.2f out=%dx%d ref_sizes=%s",
            num_inference_steps, true_cfg_scale, out_width, out_height, _ref_sizes,
        )

        # Device placement is handled entirely by group offloading (see .to());
        # every component streams between CPU and the onload device on demand,
        # so no manual VRAM-wrapping or CPU latent-bridging is needed here. The
        # only patch we still apply is the attention bypass: Qwen-Image-Edit
        # produces artifacts under sage/flash attention.
        orig_tf_forward = pipeline.transformer.forward
        def patched_tf_forward(*args: Any, **kwargs: Any) -> Any:
            with vanilla_sdpa(), disable_flash_attention():
                return orig_tf_forward(*args, **kwargs)
        pipeline.transformer.forward = patched_tf_forward

        # diffusers per-step callback adapter
        def _diffusers_callback(_pipe: Any, step_idx: int, _t: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
            if step_callback is not None:
                try:
                    step_callback(step_idx + 1, num_inference_steps)
                except Exception as exc:
                    logger.debug("step_callback raised: %s", exc)
            return kwargs

        # Stall watchdog: if the run doesn't finish, dump every thread's stack
        # each minute so we can see the exact line it's wedged on (offload
        # transfer vs FP8 matmul vs attention). Diagnostic only; cancelled on
        # completion. Remove once the hang is understood.
        _gen_done = threading.Event()
        def _stall_watchdog() -> None:
            secs = 0
            while not _gen_done.wait(60.0):
                secs += 60
                sys.stderr.write(
                    f"\n[qwen-watchdog] generation still running after {secs}s — "
                    f"dumping all thread stacks to locate the stall:\n"
                )
                sys.stderr.flush()
                faulthandler.dump_traceback(all_threads=True)
        _wd = threading.Thread(target=_stall_watchdog, name="qwen-stall-watchdog", daemon=True)
        _wd.start()
        try:
            output = pipeline(
                image=reference,
                height=out_height,
                width=out_width,
                prompt=prompt,
                negative_prompt=negative_prompt if true_cfg_scale > 1.0 else None,
                num_inference_steps=num_inference_steps,
                true_cfg_scale=true_cfg_scale,
                generator=generator,
                output_type="pil",
                return_dict=True,
                callback_on_step_end=_diffusers_callback if step_callback is not None else None,
            )
        finally:
            _gen_done.set()
            pipeline.transformer.forward = orig_tf_forward

        return self._normalize_output(output)

    # ── DF block streamer ────────────────────────────────────────────
    def _leaf_offload_non_transformer(self, runtime_device: str) -> None:
        """Leaf-level offload for text encoder / VAE / everything except the
        transformer — the configuration that has been stable throughout."""
        from diffusers.hooks import apply_group_offloading
        for name, comp in self.pipeline.components.items():
            if not isinstance(comp, nn.Module) or name == "transformer":
                continue
            apply_group_offloading(
                comp,
                onload_device=torch.device(runtime_device),
                offload_device=torch.device("cpu"),
                offload_type="leaf_level",
                use_stream=False,
                low_cpu_mem_usage=True,
            )

    def _apply_df_block_streaming(self, runtime_device: str) -> None:
        """Kijai/ComfyUI-style block swap for the transformer.

        Root modules (embeddings, norms, projections — small) live on the GPU
        permanently. As many transformer blocks as fit under
        (free VRAM − QWEN_STREAM_RESERVE_GB) are also made resident. Each
        remaining block gets a pinned CPU master copy plus forward hooks:
        pre-forward copies its tensors to the GPU (stream-ordered, so compute
        correctness needs no explicit sync), post-forward repoints them back at
        the CPU master. The master is never mutated and never receives a
        GPU→CPU copy, so the two Windows failure modes (offload access
        violation, prefetch NaN) are structurally impossible.

        Env knobs:
          QWEN_STREAM_RESERVE_GB  headroom kept free for activations/VAE (4.0)
          QWEN_STREAM_PIN         pin CPU masters for fast async H2D (1)
        """
        import os

        tf = self.pipeline.transformer
        dev = torch.device(runtime_device)
        reserve_gb = float(os.environ.get("QWEN_STREAM_RESERVE_GB", "4.0"))
        want_pin = os.environ.get("QWEN_STREAM_PIN", "1").strip() == "1"

        blocks = list(tf.transformer_blocks)

        # CPU masters are kept for EVERY transformer tensor — including the
        # resident window and root modules. They're the original (mostly
        # mmap-backed) CPU tensors, so this costs ~no extra RAM, and it lets
        # release()/parking free ALL CUDA weight copies by repointing —
        # without which a lingering reference to the pipeline leaves ~11 GB
        # allocated after a VRAM swap and the next pipeline (ZiT) loads into
        # ~3 GB and access-violates (2026-07-03).
        masters: dict[int, "torch.Tensor"] = {}
        handles = []
        pinned = want_pin
        streamed_bytes = 0

        def _snapshot(t: "torch.Tensor") -> None:
            if id(t) not in masters and t.data.device.type == "cpu":
                masters[id(t)] = t.data

        # 1) Root modules → resident. Everything except the block list is a
        #    few hundred MB at most.
        for name, child in tf.named_children():
            if child is tf.transformer_blocks:
                continue
            for t in list(child.parameters()) + list(child.buffers()):
                _snapshot(t)
            child.to(dev)
        for t in list(tf.parameters(recurse=False)) + list(tf.buffers(recurse=False)):
            _snapshot(t)
            t.data = t.data.to(dev)

        def _block_bytes(block: "nn.Module") -> int:
            return sum(
                t.numel() * t.element_size()
                for t in list(block.parameters()) + list(block.buffers())
            )

        # 2) Resident window: fill VRAM up to (free − reserve).
        torch.cuda.empty_cache()
        free_bytes, _total = torch.cuda.mem_get_info(dev.index or 0)
        budget = free_bytes - int(reserve_gb * (1024 ** 3))
        resident = 0
        used = 0
        for block in blocks:
            b = _block_bytes(block)
            if used + b > budget:
                break
            for t in list(block.parameters()) + list(block.buffers()):
                _snapshot(t)
            block.to(dev)
            used += b
            resident += 1

        # 3) Streamed blocks: pinned CPU masters + copy-in/repoint-out hooks.

        def _tensors_of(block: "nn.Module"):
            yield from block.parameters()
            yield from block.buffers()

        for block in blocks[resident:]:
            for t in _tensors_of(block):
                master = t.data
                if master.device.type != "cpu":
                    master = master.cpu()
                if pinned and not master.is_pinned():
                    try:
                        master = master.pin_memory()
                    except RuntimeError as exc:
                        logger.warning(
                            "df_stream: pin_memory failed (%s) — continuing "
                            "with unpinned masters (slower H2D).", exc,
                        )
                        pinned = False
                masters[id(t)] = master
                t.data = master
                streamed_bytes += master.numel() * master.element_size()

            def _onload(mod, _args, _masters=masters, _dev=dev, _pin=pinned):
                for t in _tensors_of(mod):
                    m = _masters.get(id(t))
                    if m is not None and t.data.device != _dev:
                        # Pinned + non_blocking copies are stream-ordered with
                        # the compute that consumes them — safe and overlapped.
                        t.data = m.to(_dev, non_blocking=_pin)
                return None

            def _release(mod, _args, _output, _masters=masters):
                for t in _tensors_of(mod):
                    m = _masters.get(id(t))
                    if m is not None:
                        t.data = m  # repoint only — never copy back
                return None

            handles.append(block.register_forward_pre_hook(_onload))
            handles.append(block.register_forward_hook(_release))

        self._df_stream = {"masters": masters, "handles": handles}
        logger.info(
            "QwenImageEditPipeline.to(%s): DF block streaming — %d/%d blocks "
            "resident (%.1f GB), %d streamed (%.1f GB, pinned=%s, reserve=%.1f GB)",
            runtime_device, resident, len(blocks), used / (1024 ** 3),
            len(blocks) - resident, streamed_bytes / (1024 ** 3), pinned, reserve_gb,
        )

    def _teardown_df_block_streaming(self) -> None:
        state = getattr(self, "_df_stream", None)
        if not state:
            return
        for handle in state.get("handles", []):
            try:
                handle.remove()
            except Exception:
                pass
        try:
            tf = self.pipeline.transformer
            masters = state.get("masters", {})
            # Repoint EVERY transformer tensor (root + resident + streamed)
            # back to its CPU master — this releases all CUDA weight copies
            # immediately, independent of whether the pipeline object itself
            # gets garbage-collected.
            for t in list(tf.parameters()) + list(tf.buffers()):
                m = masters.get(id(t))
                if m is not None:
                    t.data = m
        except Exception:
            pass
        self._df_stream = None

    def to(self, device: str) -> None:
        import os

        runtime_device = get_device_type(device)

        if runtime_device in ("cpu", "mps"):
            # Parking: tear down the DF block streamer so every CUDA weight
            # copy is repointed to its CPU master and VRAM actually returns
            # to the pool (a plain flag flip left ~11 GB resident).
            self._teardown_df_block_streaming()
            self._cpu_offload_active = False
            self._device = runtime_device
            return

        import torch
        from accelerate.hooks import remove_hook_from_module  # type: ignore[reportMissingImports]

        # Group-offload hooks are installed once and then stream weights on
        # demand for the lifetime of the pipeline. A second .to(cuda) — e.g.
        # reviving from the CPU slot — must not re-apply them (diffusers raises
        # if offloading is enabled twice). The DF block streamer, however, IS
        # torn down on park/release, so re-apply just that piece on revival.
        if self._group_offload_applied:
            mode_now = os.environ.get("QWEN_OFFLOAD_MODE", "df_stream").strip().lower()
            if mode_now == "df_stream" and not getattr(self, "_df_stream", None):
                self._apply_df_block_streaming(runtime_device)
            self._cpu_offload_active = True
            self._device = runtime_device
            return

        import os
        # QWEN_OFFLOAD_MODE selects the VRAM strategy:
        #   "group_block"       — diffusers block-level group offloading (DEFAULT). Moves whole
        #                         transformer blocks (~60 groups) instead of ~2773 leaves per
        #                         step. One Qwen block at FP8 is ~0.3 GB, so VRAM stays low
        #                         while pinned-memory alloc/free churn drops ~50×. This is the
        #                         fix for the 2026-07-03 Windows access-violation crash: with
        #                         leaf_level + low_cpu_mem_usage the per-leaf pin/unpin cycle
        #                         (2773 leaves × steps, with PEFT LoRA modules as separate
        #                         leaves) stalled 60s+ per transfer and eventually died with
        #                         "Windows fatal exception: access violation" inside
        #                         group_offloading._offload_to_memory → Module.to().
        #   "group_leaf"        — previous default; leaf-level offloading (lowest VRAM, most
        #                         transfer churn). Keep available for comparison/regression.
        #   "model_cpu_offload" — component-level offload; whole modules move on/off the
        #                         GPU. Needs the full ~12 GB FP8 transformer GPU-resident
        #                         during denoise (tight on a 15 GB card).
        # DEFAULT = "df_stream": in-house block streamer (see branch below).
        # 2026-07-03 findings that led here — every stock strategy fails on
        # Windows/torch 2.10 for this FP8 transformer:
        #   • diffusers non-stream offload re-allocates host RAM every cycle and
        #     calls .to("cpu") on the mmap'd FP8 weights → access violation.
        #   • diffusers stream prefetch completes but NaNs the latents (lazy
        #     prefetch tracer race) → black image.
        #   • full "resident" avoids both but oversubscribes the 15 GB card →
        #     WDDM sysmem fallback silently pages GPU memory → 362s/step.
        # df_stream keeps a resident window + synchronously streams the rest
        # from pinned CPU masters (copy-in / repoint-out, never copy back).
        offload_mode = os.environ.get("QWEN_OFFLOAD_MODE", "df_stream").strip().lower()
        torch.cuda.empty_cache()

        # Purge any stale accelerate hooks — offloading refuses to coexist with
        # leftover model-/sequential-cpu-offload hooks on the same module.
        for module in [self.pipeline.vae, self.pipeline.text_encoder, self.pipeline.transformer]:
            if hasattr(module, "_hf_hook"):
                remove_hook_from_module(module, recurse=True)

        if offload_mode == "df_stream":
            # DF block streamer (DEFAULT) — the in-house replacement for diffusers
            # group offloading, built from the operations proven safe on this box:
            #   • CPU master weights for streamed blocks, pinned ONCE at setup.
            #     The master never moves and is never written back — the
            #     .to("cpu") path that access-violates simply doesn't exist.
            #   • As many transformer blocks as fit under (free VRAM − reserve)
            #     stay permanently GPU-resident; only the remainder streams.
            #     The explicit reserve keeps activations out of WDDM sysmem
            #     paging (the 362s/step failure of full-resident mode).
            #   • Streaming is synchronous hook-driven copy-in / repoint-out on
            #     the default stream — no prefetch tracer, no NaN.
            self._apply_df_block_streaming(runtime_device)
            self._leaf_offload_non_transformer(runtime_device)
        elif offload_mode == "resident":
            logger.info(
                "QwenImageEditPipeline.to(%s): transformer GPU-resident, "
                "others leaf-level offloaded",
                runtime_device,
            )
            self.pipeline.transformer.to(torch.device(runtime_device))
            self._leaf_offload_non_transformer(runtime_device)
        elif offload_mode == "model_cpu_offload":
            logger.info("QwenImageEditPipeline.to(%s): enable_model_cpu_offload", runtime_device)
            self.pipeline.enable_model_cpu_offload(device=runtime_device)
        elif offload_mode == "group_block":
            # Hybrid offloading (default — see mode comment above):
            #   transformer   → block_level (~60 groups instead of ~2773 leaves;
            #                   kills the per-leaf pin churn that caused the
            #                   Windows access-violation crash in denoise)
            #   VAE + text encoder + everything else → leaf_level
            # block_level must NOT be applied pipeline-wide: it attaches the
            # "unmatched params" onload hook (conv_in, norms, …) to the module's
            # forward(), but the pipeline calls vae.encode() directly — forward()
            # never runs, the VAE convs stay on CPU, and generation dies with
            # "Input type (CUDABFloat16Type) and weight type (CPUBFloat16Type)".
            # leaf_level hooks every layer, so encode() paths work — same scoping
            # principle as the stream branch below (special-case transformer only).
            blocks_per_group = max(1, int(os.environ.get("QWEN_OFFLOAD_BLOCKS_PER_GROUP", "1")))
            logger.info(
                "QwenImageEditPipeline.to(%s): hybrid group offloading "
                "(transformer=block_level ×%d, others=leaf_level)",
                runtime_device, blocks_per_group,
            )
            from diffusers.hooks import apply_group_offloading
            for name, comp in self.pipeline.components.items():
                if not isinstance(comp, nn.Module):
                    continue
                if name == "transformer":
                    apply_group_offloading(
                        comp,
                        onload_device=torch.device(runtime_device),
                        offload_device=torch.device("cpu"),
                        offload_type="block_level",
                        num_blocks_per_group=blocks_per_group,
                        use_stream=False,
                        low_cpu_mem_usage=True,
                    )
                else:
                    apply_group_offloading(
                        comp,
                        onload_device=torch.device(runtime_device),
                        offload_device=torch.device("cpu"),
                        offload_type="leaf_level",
                        use_stream=False,
                        low_cpu_mem_usage=True,
                    )
        else:
            # Leaf-level group offloading: only the leaf currently executing stays on
            # the GPU. The ~12 GB FP8 transformer, the ~15 GB Qwen2.5-VL text encoder
            # and the VAE cannot be co-resident on a 15 GB card, so we stream.
            #
            # QWEN_OFFLOAD_USE_STREAM=1 turns on CUDA-stream prefetch (each leaf's
            # CPU->GPU copy overlaps with compute — much faster). IMPORTANT: the stream
            # path uses diffusers' lazy-prefetch tracer, and the Qwen2.5-VL text encoder
            # has per-layer rotary_emb submodules the tracer marks "unexecuted"; letting
            # it stream the text encoder mis-schedules transfers and corrupts the result
            # (NaNs -> black image). So prefetch is scoped to the transformer ONLY (the
            # per-step hot path); the text encoder and VAE always use the plain,
            # correct, non-streamed path. record_stream stays off to avoid the
            # allocator-corruption issue seen on this stack.
            # Default ON: the stream path is the only one that reuses the CPU
            # weight copies on offload (no per-step host re-allocation). Scoped
            # to the transformer only — streaming the text encoder corrupts
            # output (lazy-prefetch tracer bug → black images).
            use_stream = os.environ.get("QWEN_OFFLOAD_USE_STREAM", "1").strip() == "1"
            if not use_stream:
                logger.info(
                    "QwenImageEditPipeline.to(%s): group offloading (leaf_level, use_stream=False)",
                    runtime_device,
                )
                self.pipeline.enable_group_offload(
                    onload_device=torch.device(runtime_device),
                    offload_device=torch.device("cpu"),
                    offload_type="leaf_level",
                    use_stream=False,
                    low_cpu_mem_usage=True,
                )
            else:
                logger.info(
                    "QwenImageEditPipeline.to(%s): group offloading (leaf_level, transformer stream prefetch)",
                    runtime_device,
                )
                from diffusers.hooks import apply_group_offloading
                for name, comp in self.pipeline.components.items():
                    if not isinstance(comp, nn.Module):
                        continue
                    apply_group_offloading(
                        comp,
                        onload_device=torch.device(runtime_device),
                        offload_device=torch.device("cpu"),
                        offload_type="leaf_level",
                        use_stream=(name == "transformer"),
                        low_cpu_mem_usage=True,
                    )

        # image_processor tweak
        try:
            ip = getattr(self.pipeline, "image_processor", None)
            if ip is not None and hasattr(ip, "use_fast"):
                ip.use_fast = True  # type: ignore
        except Exception:
            pass

        self._group_offload_applied = True
        self._cpu_offload_active = True
        self._device = runtime_device

    def release(self) -> None:
        """Fully release this pipeline's GPU state for eviction/unload.

        Group offloading installs diffusers hooks that keep CUDA-side bookkeeping
        alive until the object is garbage-collected. When the pipeline manager
        stops parking and drops the reference, removing those hooks here ensures
        the VRAM is reclaimed promptly by the cleanup pass rather than lingering.
        All steps are best-effort: if diffusers internals differ, we fall back to
        plain GC (the manager drops the reference and runs empty_cache anyway).
        """
        import torch

        # DF block streamer: drop hooks and repoint every streamed tensor back
        # to its CPU master so the CUDA copies free immediately.
        self._teardown_df_block_streaming()

        pipeline = getattr(self, "pipeline", None)
        if pipeline is not None:
            try:
                from diffusers.hooks.group_offloading import (
                    _GROUP_OFFLOADING,
                    _LAYER_EXECUTION_TRACKER,
                    _LAZY_PREFETCH_GROUP_OFFLOADING,
                )
                for name in ("transformer", "text_encoder", "vae"):
                    comp = getattr(pipeline, name, None)
                    if comp is not None and hasattr(comp, "_diffusers_hook"):
                        registry = comp._diffusers_hook
                        for hook_name in (
                            _GROUP_OFFLOADING,
                            _LAYER_EXECUTION_TRACKER,
                            _LAZY_PREFETCH_GROUP_OFFLOADING,
                        ):
                            try:
                                registry.remove_hook(hook_name, recurse=True)
                            except Exception:
                                pass
            except Exception as exc:
                logger.warning("QwenImageEditPipeline.release: hook removal skipped (%s)", exc)
            # NOTE: deliberately NOT calling pipeline.to("cpu") here. Group
            # offloading already keeps the weights on CPU, and calling .to("cpu")
            # on the FP8 weights mmap'd from safetensors can trigger a fatal
            # C-level access violation on Windows / torch 2.10 (see
            # services/patches/record_stream_fix.py). Dropping the reference plus
            # the manager's gc + empty_cache reclaims the GPU memory safely.

        self._group_offload_applied = False
        self._cpu_offload_active = False
        self._device = "cpu"
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
