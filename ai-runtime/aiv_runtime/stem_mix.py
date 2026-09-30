"""Mix a subset of separated stems back down to one track.

The clip audio editor is non-destructive: `stem_split.py` writes stems beside the source and never
touches it, and this step mixes the stems the user kept into a *new* file. The clip then points at
that file. Dropping the mixdown restores the original audio, because the original was never
modified -- reverting is a pointer change, not a re-render.

Removing a stem is therefore just excluding it from this mix: drop `vocals` for an instrumental,
keep only `vocals` for a dialogue isolate, drop `other` to strip a music bed from an interview.

Input is a JSON spec on argv rather than a pile of flags, because the stem set is variable and
carries a per-stem gain:

    {"stems": [{"path": "clip-vocals.flac", "gainDb": 0.0},
               {"path": "clip-drums.flac",  "gainDb": -3.5}],
     "outputPath": "clip-mix.flac"}

Emits: {"ok": true, "outputPath": "...", "stemCount": 2}
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mix selected Demucs stems into one track.")
    parser.add_argument("--spec", required=True, help="JSON spec, or @path to a file containing it")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def load_spec(raw: str) -> dict:
    # Windows command lines mangle embedded quotes, and a stem set with many paths gets long, so
    # accept @file as well as inline JSON.
    if raw.startswith("@"):
        raw = Path(raw[1:]).read_text(encoding="utf-8")
    spec = json.loads(raw)
    stems = spec.get("stems") or []
    if not stems:
        raise ValueError("at least one stem must be kept; an empty mix would silence the clip")
    for stem in stems:
        path = stem.get("path")
        if not path:
            raise ValueError("every stem needs a path")
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    if not spec.get("outputPath"):
        raise ValueError("outputPath is required")
    return spec


def build_command(spec: dict) -> list[str]:
    """One ffmpeg invocation: per-stem gain, then a summing mix.

    `amix` divides by the input count unless told otherwise, so with the default the same stem
    would come out quieter in a 4-stem mix than in a 2-stem one -- removing a stem would change
    the loudness of every stem that remained. `normalize=0` makes it a true sum, which is what
    stems are: they were produced by splitting one signal and add back to it.

    `dropout_transition=0` stops amix from ramping gain when a shorter input ends; Demucs stems are
    all the same length, so any such ramp would be an artefact rather than a fade.
    """
    stems = spec["stems"]
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y"]
    for stem in stems:
        command += ["-i", stem["path"]]

    parts, labels = [], []
    for index, stem in enumerate(stems):
        gain = float(stem.get("gainDb", 0.0))
        label = f"a{index}"
        parts.append(f"[{index}:a]volume={gain:.3f}dB[{label}]")
        labels.append(f"[{label}]")

    if len(stems) == 1:
        graph = f"{parts[0]};[a0]anull[out]"
    else:
        graph = ";".join(parts) + ";" + "".join(labels) + \
            f"amix=inputs={len(stems)}:normalize=0:dropout_transition=0[out]"

    # FLAC keeps the mixdown lossless. Summing several stems can exceed unity even when none of
    # them clip alone, so this must not be an 8/16-bit path that would hard-clip the overshoot.
    command += ["-filter_complex", graph, "-map", "[out]", "-c:a", "flac", spec["outputPath"]]
    return command


def main() -> int:
    args = build_parser().parse_args()
    spec = load_spec(args.spec)
    command = build_command(spec)
    if args.dry_run:
        print(json.dumps({"ok": True, "command": command}))
        return 0

    Path(spec["outputPath"]).parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr.strip(), file=sys.stderr)
        raise SystemExit(f"ffmpeg mixdown failed with exit code {result.returncode}")

    print(json.dumps({
        "ok": True,
        "outputPath": spec["outputPath"],
        "stemCount": len(spec["stems"]),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
