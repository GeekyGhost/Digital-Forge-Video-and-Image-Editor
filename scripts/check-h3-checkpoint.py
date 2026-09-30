"""Pre-flight: does the FP8 checkpoint's state dict match the transformer that will load it?

Why this exists
---------------
The FP8 denoiser is a 20 GiB **torchao pre-quantized** state dict from a third party (unsloth), and
the model class that consumes it is `MiniMaxH3Transformer3DModel` from a diffusers branch. Nothing
guarantees the two agree on parameter names. Finding out the slow way costs a full stage-one run
plus a 20 GiB read before the mismatch surfaces.

This reads the key names **without loading a single tensor**. A `.pt` is a zip archive whose
`data.pkl` member holds the object graph; the storages are separate members. Unpickling `data.pkl`
with every class stubbed out yields the dict structure — so the whole check touches a few megabytes
of a 20 GiB file and finishes in seconds.

    vendor\\MiniMax-H3\\.venv\\Scripts\\python.exe scripts\\check-h3-checkpoint.py
"""

from __future__ import annotations

import argparse
import io
import pickle
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class _Stub:
    """Stands in for every class in the pickle. Never touches a storage."""

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        pass


class _KeysOnlyUnpickler(pickle.Unpickler):
    """Rebuilds the dict structure of a torch save file and nothing else.

    `find_class` returning a callable stub is what keeps `torch._utils._rebuild_tensor_v2` from
    running, and `persistent_load` returning None is what keeps the storages unread -- those two
    together are the whole trick.
    """

    def find_class(self, module: str, name: str):
        return _Stub

    def persistent_load(self, saved_id):
        return None


def checkpoint_keys(path: Path) -> list[str]:
    with zipfile.ZipFile(path) as archive:
        members = [name for name in archive.namelist() if name.endswith("data.pkl")]
        if not members:
            raise ValueError(f"{path.name} is not a zip-serialized torch checkpoint")
        raw = archive.read(sorted(members, key=len)[0])
    obj = _KeysOnlyUnpickler(io.BytesIO(raw)).load()
    if not isinstance(obj, dict):
        raise ValueError(f"expected a state dict at the top level, got {type(obj).__name__}")
    # Some publishers wrap the state dict. Unwrap one level rather than reporting a false mismatch.
    for wrapper in ("state_dict", "model", "transformer"):
        inner = obj.get(wrapper)
        if isinstance(inner, dict):
            print(f"note: the state dict is wrapped in {wrapper!r}")
            obj = inner
            break
    return [key for key in obj if isinstance(key, str)]


def expected_keys(pack: Path, subfolder: str = "transformer") -> list[str]:
    import torch  # noqa: F401  - diffusers needs it imported first on Windows
    from accelerate import init_empty_weights
    from diffusers import MiniMaxH3Transformer3DModel

    config = MiniMaxH3Transformer3DModel.load_config(pack, subfolder=subfolder)
    with init_empty_weights():
        model = MiniMaxH3Transformer3DModel.from_config(config)
    return list(model.state_dict().keys())


def sample(names, limit=6):
    listed = sorted(names)[:limit]
    return "\n".join(f"    {name}" for name in listed) + (
        f"\n    … and {len(names) - limit} more" if len(names) > limit else ""
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Check the H3 FP8 checkpoint against the model.")
    parser.add_argument("--pack", default=str(ROOT / "models" / "MiniMax-H3"))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--subfolder", choices=("transformer", "transformer_ref"), default="transformer")
    args = parser.parse_args()

    pack = Path(args.pack)
    checkpoint = Path(args.checkpoint) if args.checkpoint else pack / "quantized" / "MiniMax-H3-FP8.pt"
    if not checkpoint.is_file():
        print(f"Checkpoint not found: {checkpoint}", file=sys.stderr)
        return 2

    print(f"Reading key names from {checkpoint.name} ({checkpoint.stat().st_size / 1024 ** 3:.1f} GiB)")
    found = set(checkpoint_keys(checkpoint))
    print(f"  {len(found)} tensors in the checkpoint")

    try:
        wanted = set(expected_keys(pack, args.subfolder))
    except Exception as error:  # noqa: BLE001 - the key list is still useful on its own
        print(f"\nCould not build the model to compare against: {error}", file=sys.stderr)
        print("Checkpoint keys look like:\n" + sample(found), file=sys.stderr)
        return 3
    print(f"  {len(wanted)} tensors expected by MiniMaxH3Transformer3DModel")

    # Unsloth's released H3 FP8 partitions replace Diffusers' dense two-layer timestep MLP with
    # the sampled AdaLN curve recorded as `time_embedder.table`. The runtime performs the same
    # architecture restoration before assign-loading; mirror that key contract here so a valid
    # pruned checkpoint does not report the four deliberately removed dense weights as missing.
    if "time_embedder.table" in found:
        wanted.difference_update({
            "time_embedder.linear_1.weight", "time_embedder.linear_1.bias",
            "time_embedder.linear_2.weight", "time_embedder.linear_2.bias",
        })
        wanted.add("time_embedder.table")
        print("  detected the pruned AdaLN curve form used by the runtime")

    missing = wanted - found
    unexpected = found - wanted
    if not missing:
        print("\nOK: the checkpoint supplies every weight the transformer needs.")
        if unexpected:
            print(f"{len(unexpected)} extra tensors will be ignored (torchao scales usually land here):")
            print(sample(unexpected))
        return 0

    print(f"\nMISMATCH: {len(missing)} weights the transformer needs are not in the checkpoint.")
    print(sample(missing))
    if unexpected:
        print(f"\n{len(unexpected)} checkpoint tensors have no home in the model:")
        print(sample(unexpected))
    # The most likely cause by far, and the one worth naming.
    shared = len(found & wanted)
    if shared == 0:
        print("\nNothing matches at all, so this is a naming convention difference rather than a\n"
              "partial checkpoint -- most likely the wrong partition (ref2va instead of fl2va), or\n"
              "a ComfyUI-format file rather than a diffusers one.")
    else:
        print(f"\n{shared} keys do match, so the layouts broadly agree and this is a partial or\n"
              "differently-pruned checkpoint rather than the wrong file entirely.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
