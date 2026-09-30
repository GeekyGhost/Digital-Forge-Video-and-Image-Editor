"""Wan-Alpha v2.0 text-to-RGBA-video worker (VFX Phase 0, task 0.6).

Runs in its own venv (`vendor/Wan-Alpha/.venv`) against WeChatCV/Wan-Alpha pinned at the commit in
`scripts/runtime-matrix.psd1`, cloned to `vendor/Wan-Alpha/repo`. Weights live under
`models/Wan-Alpha/`. Nothing is shared with the spritesheet maker; only its 16 GB *pattern* is
reused (FP8 storage + BF16 compute via Diffusers layerwise casting, one-block group offloading).

Why upstream's script is not used as-is:
- it keeps the fp32 14B DiT resident on the GPU (57 GB); here the DoRA + LightX2V merge is baked
  once into FP8 shards (`wan_alpha_bake`) and blocks are streamed;
- it calls FlashAttention unconditionally; here PyTorch SDPA stands in (batch 1, no padding
  mask is ever used for T2V, so the result is the same attention);
- its Gaussian mask has 21 latent frames (81 video frames); other lengths resample it in time.

Three stages, each its own process, because process exit is the only reliable way to hand commit
back on Windows (see the 2026-08-10 LTX fault and the MiniMax-H3 two-process design):

    encode   UMT5-XXL (11.4 GB bf16) -> prompt context         (stage-encode.json)
    denoise  baked FP8 DiT, streamed; 4 UniPC steps            (stage-denoise.json)
    decode   Wan VAE with Wan-Alpha's colour + alpha decoders  (stage-decode.json)

`generate` runs all three and writes `stats.json`, RGBA PNG frames and a checkerboard preview.
Each stage records wall time, peak VRAM (torch), peak process commit (PeakPagefileUsage) and the
lowest free system commit seen, which are the Phase 0 measurements.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import subprocess
import sys
import threading
import time
import types
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPO = PROJECT_ROOT / "vendor" / "Wan-Alpha" / "repo"
DEFAULT_MODELS = PROJECT_ROOT / "models" / "Wan-Alpha"

# Upstream constants (wan/configs/shared_config.py, wan_t2v_14B.py, generate_dora_lightx2v_mask.py).
TEXT_LEN = 512
VAE_STRIDE = (4, 8, 8)
PATCH_SIZE = (1, 2, 2)
NUM_TRAIN_TIMESTEPS = 1000
DEFAULT_SHIFT = 5.0  # generate script: sample_shift defaults to 5.0 when not given
DEFAULT_ALPHA_SHIFT_MEAN = 0.05  # README: "a fixed parameter"
LIGHTX2V_FILE = "lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank64_bf16.safetensors"


def paths(models: Path) -> dict[str, Path]:
    base = models / "Wan2.1-T2V-14B"
    alpha = models / "Wan-Alpha-v2.0"
    return {
        "base": base,
        "t5": base / "models_t5_umt5-xxl-enc-bf16.pth",
        "tokenizer": base / "google" / "umt5-xxl",
        "vae": base / "Wan2.1_VAE.pth",
        "config": base / "config.json",
        "dora": alpha / "t2v.safetensors",
        "decoder": alpha / "decoder.bin",
        "gauss_mask": alpha / "gauss_mask",
        "lightx2v": models / "Lightx2v" / LIGHTX2V_FILE,
        "baked": models / "baked",
    }


def emit(event: str, **fields) -> None:
    """One JSON object per line on stdout; the Rust side parses these."""
    print(json.dumps({"event": event, **fields}), flush=True)


def log(message: str) -> None:
    print(f"[wan-alpha] {message}", file=sys.stderr, flush=True)


# ------------------------------------------------------------------------------------------------
# Memory accounting. Windows: process peak commit and system free commit via kernel32.
# ------------------------------------------------------------------------------------------------

GIB = 1024 ** 3


class MemoryWatch:
    """Samples system free commit every 250 ms; reads this process's peak commit at the end."""

    def __init__(self) -> None:
        self.min_free_commit: int | None = None
        self.commit_limit: int | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=2)

    def _run(self) -> None:
        while not self._stop.is_set():
            status = system_commit()
            if status:
                limit, free = status
                self.commit_limit = limit
                self.min_free_commit = free if self.min_free_commit is None else min(self.min_free_commit, free)
            self._stop.wait(0.25)

    def report(self) -> dict:
        return {
            "processPeakCommitGb": _gb(process_peak_commit()),
            "systemMinFreeCommitGb": _gb(self.min_free_commit),
            "commitLimitGb": _gb(self.commit_limit),
        }


def _gb(value: int | None) -> float | None:
    return None if value is None else round(value / GIB, 2)


def system_commit() -> tuple[int, int] | None:
    if os.name != "nt":
        return None
    import ctypes

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

    status = MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return status.ullTotalPageFile, status.ullAvailPageFile


def process_peak_commit() -> int | None:
    if os.name != "nt":
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024  # Linux: peak RSS, not commit
    import ctypes
    from ctypes import wintypes

    class PMC(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

    # A private WinDLL so the prototypes below do not leak into ctypes.windll. Without restype the
    # pseudo-handle (HANDLE)-1 came back as a 32-bit int and the call failed (Phase 0.6 run: null).
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.GetCurrentProcess.argtypes = []
    kernel32.K32GetProcessMemoryInfo.restype = wintypes.BOOL
    kernel32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
    counters = PMC()
    counters.cb = ctypes.sizeof(PMC)
    if not kernel32.K32GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        return None
    return counters.PeakPagefileUsage


def cuda_report() -> dict:
    import torch

    if not torch.cuda.is_available():
        return {}
    return {
        "vramPeakAllocatedGb": round(torch.cuda.max_memory_allocated() / GIB, 2),
        "vramPeakReservedGb": round(torch.cuda.max_memory_reserved() / GIB, 2),
        "device": torch.cuda.get_device_name(0),
    }


# ------------------------------------------------------------------------------------------------
# Loading upstream modules without `wan/__init__.py` (which imports every pipeline, FSDP and a T5
# whose default argument calls torch.cuda.current_device() at import time).
# ------------------------------------------------------------------------------------------------

def wan_module(repo: Path, name: str):
    """Import `wan.<name>` from the pinned clone, registering bare parent packages."""
    wan_dir = repo / "wan"
    if not (wan_dir / "modules").is_dir():
        raise FileNotFoundError(f"Wan-Alpha source not found at {repo}. Run Setup-Wan-Alpha.bat.")
    for package, directory in (("wan", wan_dir), ("wan.modules", wan_dir / "modules"), ("wan.utils", wan_dir / "utils")):
        existing = sys.modules.get(package)
        if existing is None or getattr(existing, "__path__", None) != [str(directory)]:
            module = types.ModuleType(package)
            module.__path__ = [str(directory)]
            sys.modules[package] = module
    return importlib.import_module(f"wan.{name}")


def sdpa_attention(q, k, v, q_lens=None, k_lens=None, dropout_p=0.0, softmax_scale=None, q_scale=None,
                   causal=False, window_size=(-1, -1), deterministic=False, dtype=None, version=None):
    """Drop-in for upstream `flash_attention` ([B, L, N, C] in, [B, L, N, C] out).

    Upstream only passes lengths that cover the whole sequence for T2V at batch 1 (self-attention
    uses the full token count; cross-attention gets `k_lens=None`), so no mask is needed. Anything
    else is refused rather than silently attended.
    """
    import torch

    if window_size != (-1, -1):
        raise NotImplementedError("windowed attention is not used by Wan2.1 T2V")
    for lens, tensor in ((q_lens, q), (k_lens, k)):
        if lens is not None and bool((lens != tensor.shape[1]).any()):
            raise NotImplementedError("padded sequences are not supported by the SDPA stand-in")
    half = dtype or torch.bfloat16
    out_dtype = q.dtype
    if q_scale is not None:
        q = q * q_scale
    qh, kh, vh = (t.to(half).transpose(1, 2) for t in (q, k, v))
    out = torch.nn.functional.scaled_dot_product_attention(qh, kh, vh, dropout_p=dropout_p, is_causal=causal, scale=softmax_scale)
    return out.transpose(1, 2).contiguous().to(out_dtype)


# ------------------------------------------------------------------------------------------------
# Pure helpers (unit-tested without CUDA).
# ------------------------------------------------------------------------------------------------

def latent_shape(width: int, height: int, frames: int) -> tuple[int, int, int, int]:
    if (frames - 1) % 4:
        raise ValueError(f"frames must be 4n+1, got {frames}")
    if width % 16 or height % 16:
        raise ValueError(f"width and height must be multiples of 16, got {width}x{height}")
    return (16, (frames - 1) // VAE_STRIDE[0] + 1, height // VAE_STRIDE[1], width // VAE_STRIDE[2])


def sequence_length(shape: tuple[int, int, int, int]) -> int:
    _, t, h, w = shape
    return math.ceil((h * w) / (PATCH_SIZE[1] * PATCH_SIZE[2]) * t)


def shaped_mask(mask, latent_frames: int, height: int, width: int):
    """Upstream resizes the (T, 96, 170) mask to the latent's (H, W) bilinearly. Its mask has
    T=21 (81 frames); for other lengths it is first resampled linearly in time (our addition)."""
    import torch

    if mask.shape[0] != latent_frames:
        dtype = mask.dtype
        mask = torch.nn.functional.interpolate(mask.float()[None, None], size=(latent_frames, mask.shape[1], mask.shape[2]),
                                               mode="trilinear", align_corners=True)[0, 0].to(dtype)
    resized = torch.nn.functional.interpolate(mask.unsqueeze(1), size=(height, width), mode="bilinear", align_corners=False)
    return resized.squeeze(1)


def rgba_frames(fgr, pha):
    """Upstream `render_video`: (3, F, H, W) in [-1, 1] -> uint8 RGBA frames, straight alpha.

    Colour: clamp, (x+1)/2*255, truncate. Alpha: the same per channel, then the mean of the three
    truncated channels, truncated again -- exactly what upstream writes to its PNGs.
    """
    import numpy as np

    def to_uint8(tensor):
        tensor = tensor.clamp(-1, 1)
        return ((tensor + 1) / 2 * 255).to(dtype=__import__("torch").uint8).permute(1, 2, 3, 0).cpu().numpy()

    rgb = to_uint8(fgr)
    alpha_channels = to_uint8(pha).astype(np.float64)
    alpha = ((alpha_channels[..., 0:1] + alpha_channels[..., 1:2] + alpha_channels[..., 2:3]) / 3.0).astype(np.uint8)
    return np.concatenate([rgb, alpha], axis=3)


def checkerboard_composite(rgba, square: int = 8):
    """Upstream's preview: straight alpha over a 255/200 grey checkerboard."""
    import numpy as np

    height, width = rgba.shape[1:3]
    ys, xs = np.mgrid[0:height, 0:width]
    board = np.where(((xs // square) + (ys // square)) % 2 == 0, 200, 255).astype(np.float32)[..., None]
    alpha = rgba[..., 3:4].astype(np.float32) / 255.0
    return (rgba[..., :3].astype(np.float32) * alpha + board * (1 - alpha)).astype(np.uint8)


# ------------------------------------------------------------------------------------------------
# Stages.
# ------------------------------------------------------------------------------------------------

def stage_bake(args) -> dict:
    import torch
    from accelerate import init_empty_weights

    from .wan_alpha_bake import bake, baked_files

    p = paths(Path(args.models))
    if baked_files(p["baked"]) and not args.force:
        log(f"bake already present in {p['baked']}")
        return {"skipped": True}
    model_module = wan_module(Path(args.repo), "modules.model_dora_lightx2v")
    config = model_config(p["config"])
    with init_empty_weights():
        meta_model = model_module.WanModel_dora_lightx2v(**config)
    from .wan_alpha_bake import fp8_parameter_names

    fp8_names = fp8_parameter_names(meta_model)
    del meta_model
    log(f"{len(fp8_names)} parameters will be stored as float8_e4m3fn")
    started = time.time()
    manifest = bake(p["base"], p["dora"], p["lightx2v"], p["baked"], fp8_names, dora_ratio=args.lora_ratio, log=log)
    manifest["seconds"] = round(time.time() - started, 1)
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return manifest


def model_config(config_path: Path) -> dict:
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    keys = ("model_type", "patch_size", "text_len", "in_dim", "dim", "ffn_dim", "freq_dim", "text_dim",
            "out_dim", "num_heads", "num_layers", "window_size", "qk_norm", "cross_attn_norm", "eps")
    config = {key: raw[key] for key in keys if key in raw}
    for key in ("patch_size", "window_size"):
        if key in config:
            config[key] = tuple(config[key])
    return config


def stage_encode(args) -> dict:
    import torch
    from safetensors.torch import save_file

    p = paths(Path(args.models))
    t5 = wan_module(Path(args.repo), "modules.t5")
    tokenizers = wan_module(Path(args.repo), "modules.tokenizers")
    device = torch.device("cuda" if torch.cuda.is_available() and args.t5_device != "cpu" else "cpu")
    started = time.time()
    # Built on the meta device and filled with assign=True: one copy of the 11.4 GB weights,
    # not upstream's random init plus the loaded state dict.
    model = t5.umt5_xxl(encoder_only=True, return_tokenizer=False, dtype=torch.bfloat16, device="meta").eval().requires_grad_(False)
    state = torch.load(str(p["t5"]), map_location="cpu", weights_only=True)
    model.load_state_dict(state, assign=True)
    del state
    model.to(device)
    loaded = time.time()
    tokenizer = tokenizers.HuggingfaceTokenizer(name=str(p["tokenizer"]), seq_len=TEXT_LEN, clean="whitespace")
    with torch.no_grad():
        ids, mask = tokenizer([args.prompt], return_mask=True, add_special_tokens=True)
        ids, mask = ids.to(device), mask.to(device)
        seq_len = int(mask.gt(0).sum(dim=1).long()[0])
        context = model(ids, mask)[0][:seq_len]
    save_file({"context": context.to(torch.bfloat16).cpu().contiguous()}, args.context)
    return {"device": str(device), "loadSeconds": round(loaded - started, 1), "encodeSeconds": round(time.time() - loaded, 2), "tokens": seq_len}


def stage_denoise(args) -> dict:
    import torch
    from accelerate import init_empty_weights
    from diffusers.hooks import apply_group_offloading

    from .safetensors_reader import describe_backend, load_state_dict
    from .wan_alpha_bake import SKIP_LAYERWISE_CASTING, baked_files

    if not torch.cuda.is_available():
        raise RuntimeError("the Wan-Alpha denoise stage needs CUDA")
    p = paths(Path(args.models))
    files = baked_files(p["baked"])
    if not files:
        raise FileNotFoundError(f"no baked DiT in {p['baked']}; run: wan_alpha.py bake")
    repo = Path(args.repo)
    model_module = wan_module(repo, "modules.model_dora_lightx2v")
    model_module.flash_attention = sdpa_attention
    unipc = wan_module(repo, "utils.fm_solvers_unipc")

    cuda = torch.device("cuda")
    started = time.time()
    with init_empty_weights():
        model = model_module.WanModel_dora_lightx2v(**model_config(p["config"]))
    state: dict = {}
    for path in files:
        state.update(load_state_dict(path))
    fp8_names = {name for name, tensor in state.items() if tensor.dtype == torch.float8_e4m3fn}
    model.load_state_dict(state, strict=True, assign=True)
    del state
    meta = [name for name, param in model.named_parameters() if param.is_meta]
    if meta:
        raise RuntimeError(f"{len(meta)} DiT parameters were not loaded, e.g. {meta[:3]}")
    model.eval().requires_grad_(False)
    # Already FP8 where it matters (the bake asked Diffusers which ones), so this only installs the
    # per-layer upcast-to-BF16 hooks; then stream one block at a time (the spritesheet pattern).
    model.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16, non_blocking=True,
                                   skip_modules_pattern=SKIP_LAYERWISE_CASTING)
    wrong = [name for name, param in model.named_parameters() if (name in fp8_names) != (param.dtype == torch.float8_e4m3fn)]
    if wrong:
        raise RuntimeError(f"baked FP8 layout disagrees with layerwise casting for {len(wrong)} tensors, e.g. {wrong[:3]}")
    apply_group_offloading(model, onload_device=cuda, offload_device=torch.device("cpu"), offload_type="block_level",
                           num_blocks_per_group=max(1, min(4, args.blocks_per_group)), use_stream=True,
                           record_stream=False, low_cpu_mem_usage=True)
    loaded = time.time()
    log(f"DiT ready in {loaded - started:.1f}s ({describe_backend()})")

    shape = latent_shape(args.width, args.height, args.frames)
    seq_len = sequence_length(shape)
    context = load_state_dict(args.context)["context"].to(cuda)
    generator = torch.Generator(device=cuda)
    generator.manual_seed(args.seed)
    noise = torch.randn(*shape, dtype=torch.float32, device=cuda, generator=generator)
    mask = torch.load(str(p["gauss_mask"]), map_location="cpu", weights_only=True)
    # Upstream casts to bf16 on the GPU first and interpolates in bf16; do the same.
    mask = shaped_mask(torch.as_tensor(mask).to(cuda, dtype=torch.bfloat16), shape[1], shape[2], shape[3])
    alpha_shift = torch.tensor(args.alpha_shift_mean, dtype=torch.bfloat16)
    latents = noise + (1 - mask) * alpha_shift

    scheduler = unipc.FlowUniPCMultistepScheduler(num_train_timesteps=NUM_TRAIN_TIMESTEPS, shift=1, use_dynamic_shifting=False)
    scheduler.set_timesteps(args.steps, device=cuda, shift=args.shift)
    step_seconds = []
    with torch.amp.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
        for index, t in enumerate(scheduler.timesteps):
            step_started = time.time()
            prediction = model([latents], t=torch.stack([t]), context=[context], seq_len=seq_len)[0]
            latents = scheduler.step(prediction.unsqueeze(0), t, latents.unsqueeze(0), return_dict=False, generator=generator)[0].squeeze(0)
            torch.cuda.synchronize()
            step_seconds.append(round(time.time() - step_started, 2))
            emit("step", step=index + 1, total=len(scheduler.timesteps), seconds=step_seconds[-1])
    torch.save(latents.float().cpu(), args.latents)
    return {"loadSeconds": round(loaded - started, 1), "stepSeconds": step_seconds, "latentShape": list(shape), "seqLen": seq_len}


def stage_decode(args) -> dict:
    import numpy as np
    import torch
    from PIL import Image

    p = paths(Path(args.models))
    vae_module = wan_module(Path(args.repo), "modules.vae")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.time()
    # Upstream's WanVAE calls torch.load on decoder.bin with the default (weights_only=True since
    # torch 2.6). Check that first so a pickle problem is named here, not deep inside the VAE.
    try:
        torch.load(str(p["decoder"]), map_location="cpu", weights_only=True)
    except Exception as error:
        raise RuntimeError(f"decoder.bin does not load with weights_only=True ({error}); refusing to unpickle it") from error
    vae = vae_module.WanVAE(vae_pth=str(p["vae"]), vae_lora_path=str(p["decoder"]), device=device)
    loaded = time.time()
    latents = torch.load(args.latents, map_location=device, weights_only=True)
    with torch.no_grad():
        fgr, pha = vae.decode([latents])
    frames = rgba_frames(fgr[0], pha[0])
    decoded = time.time()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(frames):
        Image.fromarray(frame, mode="RGBA").save(out / f"frame_{index:04d}.png", compress_level=1)
    preview = checkerboard_composite(frames)
    write_preview(preview, out / "checkerboard.mp4", fps=16)
    alpha = frames[..., 3]
    return {
        "loadSeconds": round(loaded - started, 1), "decodeSeconds": round(decoded - loaded, 1),
        "frames": int(frames.shape[0]), "size": f"{frames.shape[2]}x{frames.shape[1]}",
        # Soft alpha is the whole point: the share of pixels strictly between 0 and 255.
        "softAlphaFraction": round(float(((alpha > 0) & (alpha < 255)).mean()), 4),
        "opaqueFraction": round(float((alpha == 255).mean()), 4),
        "clearFraction": round(float((alpha == 0).mean()), 4),
        "alphaMean": round(float(alpha.mean()), 2),
    }


def write_preview(frames, path: Path, fps: int) -> None:
    height, width = frames.shape[1:3]
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-crf", "16",
               "-pix_fmt", "yuv420p", "-vf", "scale=out_color_matrix=bt709:out_range=tv",
               "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", str(path)]
    try:
        subprocess.run(command, input=frames.tobytes(), check=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError) as error:
        log(f"checkerboard preview skipped: {error}")


def stage_download(args) -> dict:
    """Pinned downloads into models/Wan-Alpha (resumable; already-complete files are skipped)."""
    from huggingface_hub import hf_hub_download, snapshot_download

    from .wan_alpha_bake import baked_files

    models = Path(args.models)
    started = time.time()
    # The fp32 DiT shards (57 GB) are only read by the bake. Once a valid bake exists they may have
    # been deleted on purpose, so a rerun of setup must not fetch them again unless asked.
    need_base = args.with_base or not baked_files(paths(models)["baked"])
    patterns = ["config.json", "Wan2.1_VAE.pth", "models_t5_umt5-xxl-enc-bf16.pth", "google/umt5-xxl/*",
                "diffusion_pytorch_model.safetensors.index.json"]
    if need_base:
        patterns.append("diffusion_pytorch_model-*.safetensors")
    else:
        log("bake present: skipping the fp32 DiT shards (pass --with-base to fetch them)")
    snapshot_download("Wan-AI/Wan2.1-T2V-14B", revision=args.base_revision, local_dir=models / "Wan2.1-T2V-14B",
                      allow_patterns=patterns)
    for name in ("t2v.safetensors", "decoder.bin", "gauss_mask"):
        hf_hub_download("htdong/Wan-Alpha-v2.0", name, revision=args.alpha_revision, local_dir=models / "Wan-Alpha-v2.0")
    hf_hub_download("Kijai/WanVideo_comfy", f"Lightx2v/{LIGHTX2V_FILE}", revision=args.lightx2v_revision, local_dir=models)
    missing = [str(path) for key, path in paths(models).items() if key != "baked" and not path.exists()]
    if missing:
        raise FileNotFoundError(f"downloads finished but these are missing: {missing}")
    return {"seconds": round(time.time() - started, 1), "baseShards": need_base}


STAGES = {"download": stage_download, "bake": stage_bake, "encode": stage_encode, "denoise": stage_denoise, "decode": stage_decode}


def run_stage(args) -> int:
    """Runs one stage in this process and writes its stats (plus memory) to --stats."""
    started = time.time()
    with MemoryWatch() as watch:
        try:
            result = STAGES[args.command](args)
            ok, error = True, None
        except Exception as exc:  # reported, then re-raised as a non-zero exit
            import traceback

            traceback.print_exc()
            result, ok, error = {}, False, f"{type(exc).__name__}: {exc}"
    stats = {"stage": args.command, "ok": ok, "error": error, "seconds": round(time.time() - started, 1), **result}
    try:
        stats.update(cuda_report())
    except Exception:  # torch missing or CUDA unusable: the error above already says why
        pass
    stats.update(watch.report())
    if args.stats:
        Path(args.stats).write_text(json.dumps(stats, indent=2), encoding="utf-8")
    emit("stage", **stats)
    return 0 if ok else 1


def generate(args) -> int:
    """Orchestrates bake (if needed) -> encode -> denoise -> decode, one process each."""
    from .wan_alpha_bake import baked_files

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    work = out / "work"
    work.mkdir(exist_ok=True)
    common = ["--models", args.models, "--repo", args.repo]
    geometry = ["--width", str(args.width), "--height", str(args.height), "--frames", str(args.frames)]
    plan = []
    if not baked_files(paths(Path(args.models))["baked"]):
        plan.append(["bake", *common, "--lora-ratio", str(args.lora_ratio)])
    plan += [
        ["encode", *common, "--prompt", args.prompt, "--context", str(work / "context.safetensors"), "--t5-device", args.t5_device],
        ["denoise", *common, *geometry, "--context", str(work / "context.safetensors"), "--latents", str(work / "latents.pt"),
         "--steps", str(args.steps), "--shift", str(args.shift), "--seed", str(args.seed),
         "--alpha-shift-mean", str(args.alpha_shift_mean), "--blocks-per-group", str(args.blocks_per_group)],
        ["decode", *common, "--latents", str(work / "latents.pt"), "--out-dir", str(out)],
    ]
    summary = {"prompt": args.prompt, "width": args.width, "height": args.height, "frames": args.frames,
               "steps": args.steps, "shift": args.shift, "seed": args.seed, "alphaShiftMean": args.alpha_shift_mean,
               "blocksPerGroup": args.blocks_per_group, "stages": []}
    started = time.time()
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    for stage in plan:
        stats_path = work / f"stage-{stage[0]}.json"
        emit("stage-start", stage=stage[0])
        command = [sys.executable, "-m", "aiv_runtime.wan_alpha", *stage, "--stats", str(stats_path)]
        code = subprocess.run(command, env=env, cwd=str(PROJECT_ROOT / "ai-runtime")).returncode
        stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.is_file() else {"stage": stage[0], "ok": False}
        stats["exitCode"] = code
        summary["stages"].append(stats)
        if code != 0:
            summary["error"] = stats.get("error") or f"{stage[0]} exited with {code}"
            break
    summary["seconds"] = round(time.time() - started, 1)
    summary["ok"] = "error" not in summary
    stage_values = [s for s in summary["stages"] if s.get("stage") != "bake"]
    summary["peakVramReservedGb"] = max((s.get("vramPeakReservedGb") or 0 for s in stage_values), default=None)
    summary["peakProcessCommitGb"] = max((s.get("processPeakCommitGb") or 0 for s in stage_values), default=None)
    summary["minFreeCommitGb"] = min((s["systemMinFreeCommitGb"] for s in stage_values if s.get("systemMinFreeCommitGb") is not None), default=None)
    (out / "stats.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    emit("done", **{k: summary[k] for k in ("ok", "seconds", "peakVramReservedGb", "peakProcessCommitGb", "minFreeCommitGb")}, outDir=str(out),
         error=summary.get("error"))
    return 0 if summary["ok"] else 1


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = root.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--models", default=str(DEFAULT_MODELS))
        p.add_argument("--repo", default=str(DEFAULT_REPO))
        p.add_argument("--stats", default=None)

    def geometry(p):
        p.add_argument("--width", type=int, default=832)
        p.add_argument("--height", type=int, default=480)
        p.add_argument("--frames", type=int, default=81)

    download = sub.add_parser("download", help="fetch the pinned weights into models/Wan-Alpha")
    common(download)
    download.add_argument("--base-revision", required=True)
    download.add_argument("--alpha-revision", required=True)
    download.add_argument("--lightx2v-revision", required=True)
    download.add_argument("--with-base", action="store_true",
                          help="fetch the fp32 DiT shards even when a bake exists (needed only to re-bake)")

    bake = sub.add_parser("bake", help="merge DoRA + LightX2V into FP8 shards (once)")
    common(bake)
    bake.add_argument("--lora-ratio", type=float, default=1.0)
    bake.add_argument("--force", action="store_true")

    encode = sub.add_parser("encode")
    common(encode)
    encode.add_argument("--prompt", required=True)
    encode.add_argument("--context", required=True)
    encode.add_argument("--t5-device", choices=["auto", "cpu"], default="auto")

    denoise = sub.add_parser("denoise")
    common(denoise)
    geometry(denoise)
    denoise.add_argument("--context", required=True)
    denoise.add_argument("--latents", required=True)
    denoise.add_argument("--steps", type=int, default=4)
    denoise.add_argument("--shift", type=float, default=DEFAULT_SHIFT)
    denoise.add_argument("--seed", type=int, default=42)
    denoise.add_argument("--alpha-shift-mean", type=float, default=DEFAULT_ALPHA_SHIFT_MEAN)
    denoise.add_argument("--blocks-per-group", type=int, default=1)

    decode = sub.add_parser("decode")
    common(decode)
    decode.add_argument("--latents", required=True)
    decode.add_argument("--out-dir", required=True)

    gen = sub.add_parser("generate", help="bake if needed, then encode -> denoise -> decode")
    common(gen)
    geometry(gen)
    gen.add_argument("--prompt", required=True)
    gen.add_argument("--out-dir", required=True)
    gen.add_argument("--steps", type=int, default=4)
    gen.add_argument("--shift", type=float, default=DEFAULT_SHIFT)
    gen.add_argument("--seed", type=int, default=42)
    gen.add_argument("--alpha-shift-mean", type=float, default=DEFAULT_ALPHA_SHIFT_MEAN)
    gen.add_argument("--lora-ratio", type=float, default=1.0)
    gen.add_argument("--blocks-per-group", type=int, default=1)
    gen.add_argument("--t5-device", choices=["auto", "cpu"], default="auto")
    return root


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if args.command == "generate":
        return generate(args)
    return run_stage(args)


if __name__ == "__main__":
    sys.exit(main())
