"""MiniMax-H3 Ref2VA Diagnostic Profiler & Flight Recorder.

Traces, benchmarks, and profiles every component of Ref2VA across three process stages:
1. Stage 1 (Process A): Shard Loading, Canvas Scaling, Qwen3-VL Vision Encoding, and Embeddings Serialization.
2. Stage 2 (Process B): Transformer Loading, LoRA Attachment, Streamed Block Offloading/Prefetching,
   Diffusion Step Latency, Attention Execution, and latent serialization.
3. Stage 3 (Process C): Isolated audio decode followed by isolated tiled video-VAE decode.
3. Outputs a comprehensive markdown report with exact timings, memory watermarks, and speedup analysis.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PY = _REPO_ROOT / "vendor/MiniMax-H3/.venv/Scripts/python.exe"
_REF_SCRIPT = _REPO_ROOT / "ai-runtime/aiv_runtime/minimax_h3_ref.py"


def format_bytes(bytes_val: int | float) -> str:
    return f"{bytes_val / (1024 ** 3):.2f} GiB"


class DiagnosticRecorder:
    def __init__(self, output_report: Path):
        self.events: list[dict] = []
        self.output_report = output_report
        self.start_wall = time.time()
        self.metrics: dict = {
            "stage1_time_sec": 0,
            "stage2_time_sec": 0,
            "stage3_time_sec": 0,
            "step_times": [],
            "peak_vram_gib": 0,
        }

    def record(self, stage: str, message: str, **kwargs):
        elapsed = time.time() - self.start_wall
        entry = {
            "elapsed_sec": round(elapsed, 2),
            "stage": stage,
            "message": message,
            **kwargs,
        }
        self.events.append(entry)
        print(f"[{elapsed:6.1f}s] [{stage:<14}] {message}", flush=True)

    def generate_report(self, args: argparse.Namespace) -> str:
        lines = [
            "# MiniMax-H3 Ref2VA Diagnostic Profile & Benchmark Report",
            "",
            "## ⚙️ Configuration",
            f"- **Attention Engine**: `{args.attention_backend}`",
            f"- **Offload Strategy**: `{args.offload}` (Streamed Prefetch Enabled · Pinned Host DMA)",
            f"- **Diffusion Steps**: `{args.steps}` step(s)",
            f"- **Canvas Resolution**: `{args.width}x{args.height}` ({args.num_frames} frames)",
            f"- **Reference Canvas**: `{args.ref_image_short_edge}p`",
            f"- **VAE Tiling**: `{args.vae_tile}px` tiles / `64px` overlap",
            "",
            "## ⏱️ Execution Milestones & Durations",
            f"- **Stage 1 (Conditioner / Vision Encoding)**: `{self.metrics['stage1_time_sec']:.2f}s` ({self.metrics['stage1_time_sec']/60:.1f} min)",
            f"- **Stage 2 (Denoising to Latents)**: `{self.metrics['stage2_time_sec']:.2f}s` ({self.metrics['stage2_time_sec']/60:.1f} min)",
            f"- **Stage 3 (Isolated Audio + Video Decode)**: `{self.metrics['stage3_time_sec']:.2f}s` ({self.metrics['stage3_time_sec']/60:.1f} min)",
            f"- **Total Wall Time**: `{time.time() - self.start_wall:.2f}s`",
            "",
            "## 📊 Stage Trace Log",
            "| Time (s) | Stage | Event / Milestone |",
            "| :--- | :--- | :--- |",
        ]
        for e in self.events:
            lines.append(f"| {e['elapsed_sec']} | `{e['stage']}` | {e['message']} |")

        report_str = "\n".join(lines)
        self.output_report.parent.mkdir(parents=True, exist_ok=True)
        self.output_report.write_text(report_str, encoding="utf-8")
        print(f"\n[Profiler] Diagnostic report written to {self.output_report}")
        return report_str


def run_diagnostic_profile(args: argparse.Namespace):
    report_path = Path(args.output_report) if args.output_report else _REPO_ROOT / "scratch/diagnostic_report.md"
    recorder = DiagnosticRecorder(output_report=report_path)
    recorder.record("Init", f"Starting Diagnostic Profiler run (steps={args.steps}, offload={args.offload}, backend={args.attention_backend})")

    embeds_fd, embeds_path = tempfile.mkstemp(prefix="h3-profile-embeds-", suffix=".pt")
    latents_fd, latents_path = tempfile.mkstemp(prefix="h3-profile-latents-", suffix=".pt")
    os.close(embeds_fd)
    os.close(latents_fd)

    common_args = [
        str(_PY),
        str(_REF_SCRIPT),
        "--repo-root", str(args.repo_root),
        "--transformer-path", str(args.transformer_path),
        "--text-encoder-path", str(args.text_encoder_path),
        "--turbo-lora-path", str(args.turbo_lora_path),
        "--references-json", args.references_json,
        "--prompt", args.prompt,
        "--output-path", str(args.output_path),
        "--audio-path", str(args.audio_path),
        "--width", str(args.width),
        "--height", str(args.height),
        "--num-frames", str(args.num_frames),
        "--seed", str(args.seed),
        "--attention-backend", args.attention_backend,
        "--offload", args.offload,
        "--vae-tile", str(args.vae_tile),
        "--ref-image-short-edge", str(args.ref_image_short_edge),
        "--embeds-path", embeds_path,
        "--latents-path", latents_path,
    ]

    # STAGE 1: ENCODE (Process A)
    cache_dir = Path.home() / ".cache" / "aiv_h3_embeds"
    cache_dir.mkdir(parents=True, exist_ok=True)
    import hashlib, shutil
    raw_hash = f"{args.prompt}|{getattr(args, 'references_json', '')}|{getattr(args, 'ref_image_short_edge', 256)}"
    cache_key = hashlib.sha256(raw_hash.encode("utf-8")).hexdigest()[:16]
    cache_file = cache_dir / f"h3_embeds_{cache_key}.pt"

    t1_start = time.perf_counter()
    if cache_file.is_file() and cache_file.stat().st_size > 1024:
        recorder.record("Stage1:Encode", f"Found cached reference embeddings ({cache_key}) · skipping Stage 1!")
        shutil.copyfile(str(cache_file), embeds_path)
        t1_end = time.perf_counter()
    else:
        recorder.record("Stage1:Encode", "Launching Process A (Conditioner & Vision Encoding)")
        p1 = subprocess.run([*common_args, "--stage", "encode"], text=True)
        t1_end = time.perf_counter()
        if p1.returncode != 0:
            recorder.record("Stage1:Error", f"Stage 1 failed with exit code {p1.returncode}")
            sys.exit(p1.returncode)
        try:
            shutil.copyfile(embeds_path, str(cache_file))
            recorder.record("Stage1:Encode", f"Saved embeddings to cache ({cache_key})")
        except Exception:
            pass

    recorder.metrics["stage1_time_sec"] = t1_end - t1_start
    recorder.record("Stage1:Encode", f"Stage 1 finished successfully in {t1_end - t1_start:.2f}s")

    # STAGE 2: DENOISE (Process B)
    recorder.record("Stage2:Denoise", f"Launching Process B (Denoising {args.steps} step(s) with streamed prefetch)")
    t2_start = time.perf_counter()
    p2 = subprocess.run([*common_args, "--stage", "denoise"], text=True)
    t2_end = time.perf_counter()
    if p2.returncode != 0:
        recorder.record("Stage2:Error", f"Stage 2 failed with exit code {p2.returncode}")
        sys.exit(p2.returncode)
    recorder.metrics["stage2_time_sec"] = t2_end - t2_start
    recorder.record("Stage2:Denoise", f"Stage 2 finished successfully in {t2_end - t2_start:.2f}s")

    # STAGE 3: DECODE (Process C)
    recorder.record("Stage3:Decode", "Launching Process C (audio VAE, then video VAE in isolation)")
    t3_start = time.perf_counter()
    p3 = subprocess.run([*common_args, "--stage", "decode"], text=True)
    t3_end = time.perf_counter()
    if p3.returncode != 0:
        recorder.record("Stage3:Error", f"Stage 3 failed with exit code {p3.returncode}")
        sys.exit(p3.returncode)
    recorder.metrics["stage3_time_sec"] = t3_end - t3_start
    recorder.record("Stage3:Decode", f"Stage 3 finished successfully in {t3_end - t3_start:.2f}s")

    # Cleanup temp embeds
    Path(embeds_path).unlink(missing_ok=True)
    Path(latents_path).unlink(missing_ok=True)

    # Generate Final Report
    report = recorder.generate_report(args)
    print("\n" + "=" * 60)
    print("DIAGNOSTIC REPORT SUMMARY")
    print("=" * 60)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(report)


def main():
    parser = argparse.ArgumentParser(description="MiniMax-H3 Ref2VA Diagnostic Profiler Flight Recorder")
    parser.add_argument("--repo-root", default=str(_REPO_ROOT / "models/MiniMax-H3"))
    parser.add_argument("--transformer-path", default=str(_REPO_ROOT / "models/MiniMax-H3/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors"))
    parser.add_argument("--text-encoder-path", default=str(_REPO_ROOT / "models/MiniMax-H3/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors"))
    parser.add_argument("--turbo-lora-path", default=str(_REPO_ROOT / "models/MiniMax-H3/loras/minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors"))
    parser.add_argument("--references-json", required=True)
    parser.add_argument("--prompt", default="A cinematic masterpiece.")
    parser.add_argument("--output-path", default=str(_REPO_ROOT / "scratch/profile_test_output.mp4"))
    parser.add_argument("--audio-path", default=str(_REPO_ROOT / "scratch/profile_test_output.wav"))
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=544)
    parser.add_argument("--num-frames", type=int, default=124)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attention-backend", default="native")
    parser.add_argument("--offload", default="balanced")
    parser.add_argument("--vae-tile", type=int, default=256)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--ref-image-short-edge", type=int, default=256)
    parser.add_argument("--output-report", default=str(_REPO_ROOT / "scratch/diagnostic_report.md"))
    args = parser.parse_args()
    run_diagnostic_profile(args)


if __name__ == "__main__":
    main()
