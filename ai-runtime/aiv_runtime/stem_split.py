"""Demucs stem separation for the clip audio editor.

Splits a clip's audio into independent stems so each one can land on its own timeline track.
Two roles, one adapter:

  * music  -- htdemucs / htdemucs_6s: drums, bass, other, vocals (+ guitar, piano on 6s).
    Cutting to an isolated drum stem, ducking a music bed under dialogue, muting a vocal.
  * speech -- the vocals stem doubles as a dialogue isolate, which is what Whisper should
    transcribe when the clip has music under it. Transcribing a mixed track is the usual reason
    subtitle timings drift.

Emits a single JSON document on stdout so the Rust command can parse it without depending on log
formatting; progress goes to stderr:

    {"ok": true, "sampleRate": 44100, "model": "htdemucs",
     "stems": [{"name": "vocals", "path": "..."}, ...]}

Runs in the project `.venv` (with the CUDA torch build), not the LTX vendor venv. Demucs is small
-- roughly 84 M parameters, well under 1 GB of VRAM per chunk -- so unlike the LTX adapters this
one can run while something else holds the GPU. Memory scales with the segment length rather than
the model, which is why `--segment` is the recovery knob rather than any kind of offload.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

# Demucs is trained at 44.1 kHz stereo. Feeding it anything else silently resamples inside the
# loader, so normalise up front where the cost is visible and the ffmpeg call is already ours.
MODEL_SAMPLE_RATE = 44_100

MODELS = {
    # name -> (checkpoint, stem count)
    "htdemucs": 4,
    "htdemucs_ft": 4,
    "htdemucs_6s": 6,
}


def force_utf8_io() -> None:
    """Windows consoles default to cp1252 and some progress output is non-ASCII, which raises
    UnicodeEncodeError from inside a print and takes the run down with it."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Demucs stem separation for a timeline clip.")
    parser.add_argument("--media-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", choices=tuple(MODELS), default="htdemucs")
    # Clip range in *source* seconds. Separating only the visible range keeps the cost proportional
    # to what the user selected rather than to the whole source file.
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float, default=None)
    # Comma-separated subset, e.g. "vocals" or "vocals,drums". Empty means every stem the model has.
    parser.add_argument("--stems", default="")
    # Seconds per chunk. Lower trades speed for peak memory; this is the OOM recovery knob.
    parser.add_argument("--segment", type=float, default=None)
    # Demucs blends overlapping windows; 0.25 is upstream's default and removes seam artefacts.
    parser.add_argument("--overlap", type=float, default=0.25)
    # >1 averages several random time-shifts. Materially slower for a modest quality gain.
    parser.add_argument("--shifts", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def validate(args: argparse.Namespace) -> None:
    if not Path(args.media_path).is_file():
        raise FileNotFoundError(args.media_path)
    if args.end is not None and args.end <= args.start:
        raise ValueError("end must be greater than start")
    if not 0 <= args.overlap < 1:
        raise ValueError("overlap must be in [0, 1)")
    if args.shifts < 0:
        raise ValueError("shifts cannot be negative")


def extract_audio(args: argparse.Namespace, destination: Path) -> None:
    """Decode the clip's visible range to 44.1 kHz stereo PCM.

    `-ss` before `-i` seeks on the input, which is fast but snaps to a keyframe on long-GOP video.
    Audio is decoded independently of video keyframes, so for an audio-only extraction this is both
    fast and sample-accurate.
    """
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y"]
    if args.start > 0:
        command += ["-ss", f"{args.start:.6f}"]
    command += ["-i", args.media_path]
    if args.end is not None:
        command += ["-t", f"{args.end - args.start:.6f}"]
    command += ["-vn", "-ac", "2", "-ar", str(MODEL_SAMPLE_RATE), "-c:a", "pcm_s16le", str(destination)]
    subprocess.run(command, check=True)


def separate(args: argparse.Namespace, audio_path: Path) -> dict:
    import torch
    import torchaudio
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[demucs] loading {args.model} on {device}", file=sys.stderr)
    model = get_model(args.model)
    model.to(device)
    model.eval()

    wav, sample_rate = torchaudio.load(str(audio_path))
    if sample_rate != model.samplerate:
        wav = torchaudio.functional.resample(wav, sample_rate, model.samplerate)
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)

    # Demucs normalises against the mixture's own statistics; skipping this makes quiet clips
    # separate noticeably worse.
    reference = wav.mean(0)
    mean, std = reference.mean(), reference.std()
    wav = (wav - mean) / (std + 1e-8)

    segment = args.segment
    while True:
        try:
            if segment is not None:
                model.segment = segment
            print(f"[demucs] separating ({'auto' if segment is None else f'{segment:g}s'} segments)", file=sys.stderr)
            with torch.no_grad():
                sources = apply_model(
                    model, wav[None], device=device, shifts=args.shifts,
                    split=True, overlap=args.overlap, progress=True,
                )[0]
            break
        except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
            # Only OOM is recoverable here; anything else is a real bug and must not be masked by
            # a retry loop that eventually reports the wrong cause.
            if "out of memory" not in str(error).lower():
                raise
            torch.cuda.empty_cache()
            segment = 10.0 if segment is None else segment / 2
            if segment < 1.0:
                raise
            print(f"[demucs] out of memory - retrying at {segment:g}s segments", file=sys.stderr)

    sources = sources * (std + 1e-8) + mean

    wanted = {name.strip() for name in args.stems.split(",") if name.strip()}
    unknown = wanted - set(model.sources)
    if unknown:
        raise ValueError(f"unknown stems {sorted(unknown)}; {args.model} provides {list(model.sources)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem_name = Path(args.media_path).stem

    written = []
    for name, audio in zip(model.sources, sources):
        if wanted and name not in wanted:
            continue
        # FLAC: lossless, and roughly half the size of the WAV these would otherwise be. Stems are
        # intermediates the editor re-reads, so a lossy round trip here would compound on export.
        path = output_dir / f"{stem_name}-{name}.flac"
        torchaudio.save(str(path), audio.cpu(), model.samplerate, format="flac")
        written.append({"name": name, "path": str(path)})
        print(f"[demucs] wrote {name}", file=sys.stderr)

    # Read the rate off the model before it is freed; the caller needs it to build the audio
    # clips, and it is the model's rate that the files were actually written at.
    sample_rate_out = model.samplerate
    del model, sources
    if device == "cuda":
        torch.cuda.empty_cache()

    return {"ok": True, "sampleRate": sample_rate_out, "model": args.model, "stems": written}


def main() -> int:
    force_utf8_io()
    args = build_parser().parse_args()
    validate(args)
    if args.dry_run:
        print(json.dumps({"ok": True, "model": args.model, "stems": []}))
        return 0
    with tempfile.TemporaryDirectory(prefix="aiv-demucs-") as scratch:
        audio_path = Path(scratch, "mix.wav")
        extract_audio(args, audio_path)
        payload = separate(args, audio_path)
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
