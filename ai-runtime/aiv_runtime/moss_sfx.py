"""Standalone MOSS-SoundEffect v2 worker for the AI Clip Editor.

The worker deliberately owns no server or persistent process. Rust's AI execution gate starts one
process per take, so switching back to LTX releases every MOSS allocation and the two large models
can never occupy the GPU together.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a local sound-effect WAV with MOSS v2")
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--sigma-shift", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    model_root = Path(args.model_root).resolve()
    output_path = Path(args.output_path).resolve()
    if not (model_root / "model_index.json").is_file():
        raise FileNotFoundError(f"MOSS model pack is incomplete: {model_root}")
    if output_path.suffix.lower() != ".wav":
        raise ValueError("MOSS output must use the .wav extension")
    if not 0.1 <= args.seconds <= 30.0:
        raise ValueError("duration must be between 0.1 and 30 seconds")

    # The transferred package is a sibling of aiv_runtime. Add ai-runtime itself, never a Digital
    # Forge directory, so the installed editor remains completely standalone.
    runtime_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(runtime_root))

    # MOSS's DAC dependency still uses the legacy weight_norm helper. Its checkpoint layout is
    # tied to that helper, so suppress the upstream warning until MOSS ships a compatible update.
    warnings.filterwarnings(
        "ignore",
        message=r"`torch\.nn\.utils\.weight_norm` is deprecated.*",
        category=FutureWarning,
    )

    import torch
    from moss_soundeffect_v2 import MossSoundEffectPipeline

    if not torch.cuda.is_available():
        raise RuntimeError("MOSS requires a CUDA-capable PyTorch installation")

    print(f"[moss-sfx] loading {model_root}", flush=True)
    pipeline = MossSoundEffectPipeline.from_pretrained(
        str(model_root), torch_dtype=torch.bfloat16, device="cuda"
    )
    print(
        f"[moss-sfx] generating {args.seconds:.1f}s, {args.steps} steps, seed {args.seed}",
        flush=True,
    )
    audio = pipeline(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        seconds=args.seconds,
        num_inference_steps=args.steps,
        cfg_scale=args.cfg_scale,
        sigma_shift=args.sigma_shift,
        seed=args.seed,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pipeline.save_audio(audio, output_path, sample_rate=48_000)
    channels = int(audio.shape[1] if audio.ndim == 3 else audio.shape[0] if audio.ndim == 2 else 1)
    result = {
        "outputPath": str(output_path),
        "sampleRate": 48_000,
        "channels": channels,
        "durationSeconds": float(audio.shape[-1]) / 48_000.0,
        "seed": args.seed,
    }
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
