"""Whisper large-v3-turbo transcription adapter.

Emits progress on stderr and a single JSON document on stdout so the Rust command can parse
the result without depending on log formatting:

    {"ok": true, "language": "en", "cues": [{"start": 0.0, "end": 1.8, "text": "..."}]}
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

TARGET_SAMPLE_RATE = 16_000
# Whisper's receptive field is 30 s; longer media is processed in chunks with overlap so word
# timings stay stable across the seams.
CHUNK_SECONDS = 30
OVERLAP_SECONDS = 5


def force_utf8_io() -> None:
    """transformers prints emoji in some warnings, and Windows consoles default to cp1252, where
    encoding one raises UnicodeEncodeError and aborts the run."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Whisper transcription for the caption track.")
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--media-path", required=True)
    parser.add_argument("--language", default=None)
    parser.add_argument("--translate", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def validate(args: argparse.Namespace) -> None:
    if not Path(args.model_root).is_dir():
        raise FileNotFoundError(args.model_root)
    if not Path(args.media_path).is_file():
        raise FileNotFoundError(args.media_path)


def extract_audio(media_path: str, destination: Path) -> None:
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-i", media_path, "-vn", "-ac", "1", "-ar", str(TARGET_SAMPLE_RATE),
            "-c:a", "pcm_s16le", str(destination),
        ],
        check=True,
    )


def transcribe(args: argparse.Namespace, audio_path: Path) -> dict:
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"[whisper] loading large-v3-turbo on {device}", file=sys.stderr)

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model_root, torch_dtype=dtype, low_cpu_mem_usage=True, use_safetensors=True,
    ).to(device)
    processor = AutoProcessor.from_pretrained(args.model_root)
    asr = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=dtype,
        device=device,
        chunk_length_s=CHUNK_SECONDS,
        stride_length_s=OVERLAP_SECONDS,
    )

    generate_kwargs = {"task": "translate" if args.translate else "transcribe"}
    if args.language:
        generate_kwargs["language"] = args.language

    print("[whisper] transcribing", file=sys.stderr)
    result = asr(str(audio_path), return_timestamps=True, generate_kwargs=generate_kwargs)

    cues = []
    for chunk in result.get("chunks") or []:
        start, end = (chunk.get("timestamp") or (None, None))
        text = (chunk.get("text") or "").strip()
        if start is None or end is None or not text:
            continue
        if end <= start:
            continue
        cues.append({"start": float(start), "end": float(end), "text": text})

    if not cues and (result.get("text") or "").strip():
        # Some builds return only the flat transcript; keep it rather than losing the pass.
        cues.append({"start": 0.0, "end": 5.0, "text": result["text"].strip()})

    del asr, model
    if device == "cuda":
        torch.cuda.empty_cache()
    return {"ok": True, "language": args.language or "auto", "cues": cues}


def main() -> int:
    force_utf8_io()
    args = build_parser().parse_args()
    validate(args)
    if args.dry_run:
        print(json.dumps({"ok": True, "cues": []}))
        return 0
    with tempfile.TemporaryDirectory(prefix="aiv-whisper-") as scratch:
        audio_path = Path(scratch, "audio.wav")
        extract_audio(args.media_path, audio_path)
        payload = transcribe(args, audio_path)
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
