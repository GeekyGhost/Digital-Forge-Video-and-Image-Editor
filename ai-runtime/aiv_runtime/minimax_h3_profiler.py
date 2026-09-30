"""Opt-in, output-neutral profiling for MiniMax-H3 denoising.

The profiler deliberately uses PyTorch observation hooks instead of modifying the
Diffusers denoising loop.  It synchronizes CUDA at transformer boundaries, so it
must remain opt-in: the resulting timings are attributable, but a profiled run is
not itself a clean throughput benchmark.
"""

from __future__ import annotations

import functools
import json
import os
import shutil
import subprocess
import time
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

PROFILE_ENV = "AIV_H3_PROFILE_DENOISE"
PROFILE_PATH_ENV = "AIV_H3_PROFILE_PATH"


def profile_requested(args) -> bool:
    """Return whether the caller explicitly requested denoise instrumentation."""
    value = str(os.environ.get(PROFILE_ENV, "")).strip().lower()
    return bool(
        getattr(args, "profile_denoise", False)
        or getattr(args, "profile_path", None)
        or value in {"1", "true", "yes", "on"}
    )


def resolve_profile_path(args) -> Path:
    """Resolve the JSONL report beside the requested video unless overridden."""
    explicit = getattr(args, "profile_path", None) or os.environ.get(PROFILE_PATH_ENV)
    output_path = Path(args.output_path).resolve()
    profile_path = (
        Path(explicit).expanduser().resolve()
        if explicit
        else Path(str(output_path) + ".h3-profile.jsonl")
    )
    protected = {output_path}
    audio_path = getattr(args, "audio_path", None)
    if audio_path:
        protected.add(Path(audio_path).resolve())
    if profile_path in protected:
        raise ValueError("H3 profile path must not overwrite the generated video or audio")
    return profile_path


class MiniMaxH3DenoiseProfiler:
    """Collect per-transformer-evaluation timing, memory, and offload metrics."""

    def __init__(
        self,
        transformer,
        output_path: str | Path,
        *,
        metadata: dict[str, Any] | None = None,
        logger: Callable[[str], None] | None = None,
    ) -> None:
        import psutil
        import torch

        self.transformer = transformer
        self.output_path = Path(output_path)
        self.metadata = dict(metadata or {})
        self.log = logger or (lambda _message: None)
        self.torch = torch
        self.process = psutil.Process()
        self._psutil = psutil
        self._handles = []
        self._restore_methods: list[tuple[type, str, Any]] = []
        self._active_step: dict[str, Any] | None = None
        self._phase_calls: list[dict[str, Any]] = []
        self._step_records: list[dict[str, Any]] = []
        self._transfer_totals = defaultdict(float)
        self._group_size_cache: dict[int, int] = {}
        self._started_at = time.perf_counter()
        self._nvml = None
        self._nvml_handle = None
        self._nvml_error = None
        self._nvidia_smi = None
        self._initialize_nvml()

    def _initialize_nvml(self) -> None:
        try:
            import pynvml

            pynvml.nvmlInit()
            index = self.torch.cuda.current_device() if self.torch.cuda.is_available() else 0
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            self._nvml = pynvml
        except Exception as error:  # noqa: BLE001 - NVML exceptions vary by package/driver.
            self._nvml = None
            self._nvml_handle = None
            self._nvml_error = f"{type(error).__name__}: {error}"
            self._nvidia_smi = shutil.which("nvidia-smi")

    def _nvidia_smi_snapshot(self) -> dict[str, Any]:
        """Sample the NVIDIA CLI when the optional Python NVML binding is absent."""
        if self._nvidia_smi is None:
            return {}
        fields = (
            "utilization.gpu",
            "utilization.memory",
            "power.draw",
            "temperature.gpu",
            "clocks.current.graphics",
            "clocks.current.memory",
            "pcie.link.gen.current",
            "pcie.link.width.current",
        )
        try:
            completed = subprocess.run(
                [
                    self._nvidia_smi,
                    f"--query-gpu={','.join(fields)}",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=3,
            )
            values = [part.strip() for part in completed.stdout.strip().split(",")]
            if completed.returncode or len(values) != len(fields):
                detail = completed.stderr.strip() or completed.stdout.strip()
                return {"nvidia_smi_sample_error": detail}
            return {
                "utilization_percent": int(values[0]),
                "memory_utilization_percent": int(values[1]),
                "power_watts": float(values[2]),
                "temperature_c": int(values[3]),
                "graphics_clock_mhz": int(values[4]),
                "memory_clock_mhz": int(values[5]),
                "pcie_generation": int(values[6]),
                "pcie_width": int(values[7]),
            }
        except (OSError, subprocess.SubprocessError, TypeError, ValueError) as error:
            return {"nvidia_smi_sample_error": f"{type(error).__name__}: {error}"}

    def _host_snapshot(self) -> dict[str, Any]:
        memory = self.process.memory_info()
        virtual = self._psutil.virtual_memory()
        swap = self._psutil.swap_memory()
        result = {
            "process_rss_bytes": int(memory.rss),
            "process_vms_bytes": int(memory.vms),
            "system_available_bytes": int(virtual.available),
            "system_memory_percent": float(virtual.percent),
            "pagefile_used_bytes": int(swap.used),
            "pagefile_percent": float(swap.percent),
            "pagefile_read_bytes": int(getattr(swap, "sin", 0)),
            "pagefile_write_bytes": int(getattr(swap, "sout", 0)),
        }
        # psutil exposes these on Windows. Keeping them conditional makes the
        # profiler testable on other development hosts without weakening the
        # actual Windows report.
        optional = {
            "process_pagefile_bytes": "pagefile",
            "process_peak_pagefile_bytes": "peak_pagefile",
            "process_page_faults": "num_page_faults",
            "process_peak_working_set_bytes": "peak_wset",
        }
        for key, field in optional.items():
            value = getattr(memory, field, None)
            if value is not None:
                result[key] = int(value)
        return result

    def _cuda_snapshot(self) -> dict[str, Any]:
        if not self.torch.cuda.is_available():
            return {"available": False}
        free, total = self.torch.cuda.mem_get_info()
        result = {
            "available": True,
            "allocated_bytes": int(self.torch.cuda.memory_allocated()),
            "reserved_bytes": int(self.torch.cuda.memory_reserved()),
            "peak_allocated_bytes": int(self.torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(self.torch.cuda.max_memory_reserved()),
            "free_bytes": int(free),
            "total_bytes": int(total),
        }
        if self._nvml is not None and self._nvml_handle is not None:
            try:
                utilization = self._nvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
                result.update(
                    {
                        "utilization_percent": int(utilization.gpu),
                        "memory_utilization_percent": int(utilization.memory),
                        "power_watts": self._nvml.nvmlDeviceGetPowerUsage(self._nvml_handle)
                        / 1000.0,
                        "temperature_c": int(
                            self._nvml.nvmlDeviceGetTemperature(
                                self._nvml_handle, self._nvml.NVML_TEMPERATURE_GPU
                            )
                        ),
                    }
                )
                for key, counter in (
                    ("pcie_rx_kib_per_s", self._nvml.NVML_PCIE_UTIL_RX_BYTES),
                    ("pcie_tx_kib_per_s", self._nvml.NVML_PCIE_UTIL_TX_BYTES),
                ):
                    try:
                        result[key] = int(
                            self._nvml.nvmlDeviceGetPcieThroughput(self._nvml_handle, counter)
                        )
                    except Exception as error:  # noqa: BLE001 - optional across NVML versions.
                        result[f"{key}_error"] = f"{type(error).__name__}: {error}"
            except Exception as error:  # noqa: BLE001 - NVML exceptions vary by package/driver.
                result["nvml_sample_error"] = f"{type(error).__name__}: {error}"
        elif self._nvidia_smi is not None:
            result.update(self._nvidia_smi_snapshot())
        return result

    def _write_record(self, record: dict[str, Any]) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")

    def _phase_pre(self, category: str, name: str):
        def hook(_module, _args, _kwargs):
            if self._active_step is None:
                return
            call = {
                "category": category,
                "name": name,
                "wall_started": time.perf_counter(),
            }
            if self.torch.cuda.is_available():
                call["cuda_started"] = self.torch.cuda.Event(enable_timing=True)
                call["cuda_finished"] = self.torch.cuda.Event(enable_timing=True)
                call["cuda_started"].record()
            self._phase_calls.append(call)

        return hook

    def _phase_post(self, category: str, name: str):
        def hook(_module, _args, _kwargs, _output):
            if self._active_step is None:
                return
            for call in reversed(self._phase_calls):
                if call["category"] == category and call["name"] == name and "wall_s" not in call:
                    call["wall_s"] = time.perf_counter() - call["wall_started"]
                    if "cuda_finished" in call:
                        call["cuda_finished"].record()
                    break

        return hook

    def _attach_phase(self, module, category: str, name: str) -> None:
        self._handles.append(
            module.register_forward_pre_hook(self._phase_pre(category, name), with_kwargs=True)
        )
        self._handles.append(
            module.register_forward_hook(self._phase_post(category, name), with_kwargs=True)
        )

    def _attach_phase_hooks(self) -> None:
        for name, module in self.transformer.named_modules():
            parts = name.split(".")
            category = None
            if (
                len(parts) >= 2
                and parts[-2] == "attn"
                and parts[-1]
                in {
                    "to_q",
                    "to_k",
                    "to_v",
                    "to_qkv",
                }
            ):
                category = "qkv"
            elif parts and parts[-1] == "attn":
                category = "attention_total"
            elif parts and parts[-1] == "ff":
                category = "mlp"
            elif len(parts) == 2 and parts[0] == "transformer_blocks" and parts[1].isdigit():
                category = "transformer_block"
            if category is not None:
                self._attach_phase(module, category, name)

    def _group_bytes(self, group) -> int:
        cache_key = id(group)
        if cache_key in self._group_size_cache:
            return self._group_size_cache[cache_key]
        tensors = []
        for module in getattr(group, "modules", []):
            tensors.extend(module.parameters())
            tensors.extend(module.buffers())
        tensors.extend(getattr(group, "parameters", []))
        tensors.extend(getattr(group, "buffers", []))
        seen = set()
        total = 0
        for tensor in tensors:
            if id(tensor) in seen:
                continue
            seen.add(id(tensor))
            try:
                total += int(tensor.numel() * tensor.element_size())
            except (RuntimeError, TypeError, ValueError, AttributeError):
                continue
        self._group_size_cache[cache_key] = total
        return total

    def _record_transfer_call(self, category: str, group, elapsed: float) -> None:
        if self._active_step is None:
            return
        self._transfer_totals[f"{category}_calls"] += 1
        self._transfer_totals[f"{category}_wall_s"] += elapsed
        if category == "onload":
            self._transfer_totals["onload_scheduled_bytes"] += self._group_bytes(group)

    def _patch_offload_methods(self) -> None:
        try:
            from diffusers.hooks.group_offloading import GroupOffloadingHook, ModuleGroup
        except (ImportError, AttributeError) as error:
            self.log(f"profile · offload hook timing unavailable ({type(error).__name__}: {error})")
            return

        profiler = self

        for method_name, category in (("onload_", "onload"), ("offload_", "offload")):
            original = getattr(ModuleGroup, method_name)

            @functools.wraps(original)
            def wrapped(group, *args, __original=original, __category=category, **kwargs):
                started = time.perf_counter()
                try:
                    return __original(group, *args, **kwargs)
                finally:
                    profiler._record_transfer_call(__category, group, time.perf_counter() - started)

            setattr(ModuleGroup, method_name, wrapped)
            self._restore_methods.append((ModuleGroup, method_name, original))

        for method_name, category in (
            ("pre_forward", "group_pre_forward"),
            ("post_forward", "group_post_forward"),
        ):
            original = getattr(GroupOffloadingHook, method_name)

            @functools.wraps(original)
            def wrapped(hook, *args, __original=original, __category=category, **kwargs):
                started = time.perf_counter()
                try:
                    return __original(hook, *args, **kwargs)
                finally:
                    profiler._record_transfer_call(
                        __category, hook.group, time.perf_counter() - started
                    )

            setattr(GroupOffloadingHook, method_name, wrapped)
            self._restore_methods.append((GroupOffloadingHook, method_name, original))

    @staticmethod
    def _timestep_value(kwargs: dict[str, Any]):
        value = kwargs.get("timestep")
        if value is None:
            return None
        try:
            values = value.detach().float().reshape(-1).cpu().tolist()
            return values[0] if len(values) == 1 else values
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return str(value)

    def _transformer_pre(self, _module, args, kwargs):
        if self.torch.cuda.is_available():
            self.torch.cuda.synchronize()
            self.torch.cuda.reset_peak_memory_stats()
        self._phase_calls = []
        self._transfer_totals = defaultdict(float)
        hidden_states = kwargs.get("hidden_states")
        self._active_step = {
            "index": len(self._step_records),
            "wall_started": time.perf_counter(),
            "timestep": self._timestep_value(kwargs),
            "sequence_rows": int(hidden_states.shape[-2]) if hidden_states is not None else None,
            "host_before": self._host_snapshot(),
            "cuda_before": self._cuda_snapshot(),
        }

    def _aggregate_phases(self) -> dict[str, Any]:
        aggregate: dict[str, dict[str, Any]] = {}
        for call in self._phase_calls:
            if "wall_s" not in call:
                continue
            entry = aggregate.setdefault(call["category"], {"calls": 0, "wall_s": 0.0})
            entry["calls"] += 1
            entry["wall_s"] += float(call["wall_s"])
            if "cuda_started" in call:
                try:
                    elapsed = call["cuda_started"].elapsed_time(call["cuda_finished"]) / 1000.0
                    entry["cuda_s"] = entry.get("cuda_s", 0.0) + elapsed
                except RuntimeError:
                    entry["cuda_timing_failed"] = True
        attention = aggregate.get("attention_total")
        qkv = aggregate.get("qkv")
        if attention is not None and qkv is not None:
            aggregate["attention_non_qkv"] = {
                "note": "attention total minus Q/K/V projections; includes QK norm, RoPE, and output projection",
                "wall_s": max(float(attention["wall_s"]) - float(qkv["wall_s"]), 0.0),
            }
            if "cuda_s" in attention and "cuda_s" in qkv:
                aggregate["attention_non_qkv"]["cuda_s"] = max(
                    float(attention["cuda_s"]) - float(qkv["cuda_s"]), 0.0
                )
        return aggregate

    def _transformer_post(self, _module, _args, _kwargs, _output):
        if self._active_step is None:
            return
        if self.torch.cuda.is_available():
            self.torch.cuda.synchronize()
        active = self._active_step
        record = {
            "type": "step",
            "index": active["index"],
            "timestep": active["timestep"],
            "sequence_rows": active["sequence_rows"],
            "wall_s": time.perf_counter() - active["wall_started"],
            "phases": self._aggregate_phases(),
            "offload": {
                key: int(value) if key.endswith(("_calls", "_bytes")) else value
                for key, value in self._transfer_totals.items()
            },
            "host_before": active["host_before"],
            "host_after": self._host_snapshot(),
            "cuda_before": active["cuda_before"],
            "cuda_after": self._cuda_snapshot(),
        }
        record["host_delta"] = {
            key: record["host_after"].get(key, 0) - value
            for key, value in record["host_before"].items()
            if isinstance(value, (int, float))
            and isinstance(record["host_after"].get(key), (int, float))
        }
        self._step_records.append(record)
        self._write_record(record)
        self.log(
            "profile step "
            f"{record['index'] + 1:02d} · {record['wall_s']:.2f}s · "
            f"RSS={record['host_after']['process_rss_bytes'] / 1024**3:.2f} GiB · "
            f"pagefile={record['host_after']['pagefile_used_bytes'] / 1024**3:.2f} GiB · "
            f"VRAM peak={record['cuda_after'].get('peak_allocated_bytes', 0) / 1024**3:.2f} GiB · "
            f"reserved={record['cuda_after'].get('reserved_bytes', 0) / 1024**3:.2f} GiB · "
            f"GPU={record['cuda_after'].get('utilization_percent', 0)}% · "
            f"onload/wait={record['offload'].get('onload_wall_s', 0.0):.2f}s"
        )
        self._active_step = None

    def __enter__(self):
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.unlink(missing_ok=True)
        self._write_record(
            {
                "type": "metadata",
                "format": "aiv-h3-denoise-profile-v1",
                "profiling_note": "CUDA is synchronized at transformer boundaries; use unprofiled runs for final throughput.",
                "nvml_available": self._nvml is not None,
                "nvml_error": self._nvml_error,
                "gpu_sampler": (
                    "nvml" if self._nvml is not None else "nvidia-smi" if self._nvidia_smi else None
                ),
                **self.metadata,
            }
        )
        self._patch_offload_methods()
        self._attach_phase_hooks()
        self._handles.append(
            self.transformer.register_forward_pre_hook(self._transformer_pre, with_kwargs=True)
        )
        self._handles.append(
            self.transformer.register_forward_hook(self._transformer_post, with_kwargs=True)
        )
        self.log(f"profile enabled · per-step JSONL -> {self.output_path}")
        return self

    def __exit__(self, error_type, error, _traceback):
        for handle in reversed(self._handles):
            handle.remove()
        for owner, name, original in reversed(self._restore_methods):
            setattr(owner, name, original)
        elapsed = time.perf_counter() - self._started_at
        wall_values = [record["wall_s"] for record in self._step_records]
        summary = {
            "type": "summary",
            "completed_steps": len(self._step_records),
            "profile_wall_s": elapsed,
            "step_min_s": min(wall_values) if wall_values else None,
            "step_max_s": max(wall_values) if wall_values else None,
            "step_mean_s": sum(wall_values) / len(wall_values) if wall_values else None,
            "error": f"{error_type.__name__}: {error}" if error_type is not None else None,
        }
        self._write_record(summary)
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception as shutdown_error:  # noqa: BLE001 - NVML exceptions vary by package/driver.
                self.log(
                    "profile · NVML shutdown warning "
                    f"({type(shutdown_error).__name__}: {shutdown_error})"
                )
        self.log(
            f"profile complete · {summary['completed_steps']} steps · report {self.output_path}"
        )
        return False
