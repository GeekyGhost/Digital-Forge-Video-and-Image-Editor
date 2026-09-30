from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "models" / "manifests"
INSTALL_ROOT = Path(os.environ.get("AIV_MODELS", ROOT / "models"))
STATE_PATH = INSTALL_ROOT / "installed.json"


def manifests() -> dict[str, dict[str, Any]]:
    return {path.stem: json.loads(path.read_text(encoding="utf-8")) for path in MANIFESTS.glob("*.json")}


def disk_free_gib(path: Path) -> float:
    path.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(path).free / 1024**3


def load_state() -> dict[str, Any]:
    if not STATE_PATH.exists():
        return {"schemaVersion": 1, "packs": {}}
    return json.loads(STATE_PATH.read_text(encoding="utf-8"))


def save_state(state: dict[str, Any]) -> None:
    temporary = STATE_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    temporary.replace(STATE_PATH)


def list_packs(items: dict[str, dict[str, Any]]) -> int:
    state = load_state()
    print(f"Models: {INSTALL_ROOT}")
    print(f"Free disk: {disk_free_gib(INSTALL_ROOT):.1f} GiB\n")
    for pack_id, item in sorted(items.items()):
        installed = state["packs"].get(pack_id)
        status = "installed" if installed else "not installed"
        print(f"{pack_id:18} {item['estimatedSizeGiB']:>4} GiB  {status:13} {item['name']}")
        print(f"{'':25}{item['description']}")
    return 0


def download_files(files: list[dict[str, Any]], destination: Path) -> int:
    """Fetch individually-listed files, verifying each against its recorded SHA-256.

    Used for upstreams that publish to GitHub Releases rather than to the Hugging Face Hub. The
    hash is not decoration: these are pickled PyTorch state dicts loaded straight into the process,
    so a truncated download or a substituted file is worth catching before it is ever opened. A
    file already present and already matching is left alone, which makes re-running install cheap.
    """
    import hashlib
    import urllib.request

    for item in files:
        target = destination / item["name"]
        expected = item.get("sha256")

        def digest(path: Path) -> str:
            hasher = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()

        if target.is_file() and expected and digest(target) == expected:
            print(f"Already present, hash matches: {item['name']}")
            continue

        print(f"Downloading {item['name']} -> {target}")
        partial = target.with_suffix(target.suffix + ".part")
        try:
            urllib.request.urlretrieve(item["url"], partial)
        except Exception as error:  # noqa: BLE001 - the URL and the reason both matter here
            print(f"Failed to download {item['url']}: {error}", file=sys.stderr)
            partial.unlink(missing_ok=True)
            return 1

        if expected:
            actual = digest(partial)
            if actual != expected:
                partial.unlink(missing_ok=True)
                print(
                    f"Hash mismatch for {item['name']}:\n  expected {expected}\n  got      {actual}",
                    file=sys.stderr,
                )
                return 1
        partial.replace(target)
    return 0


def install(pack: dict[str, Any], accept_licenses: bool) -> int:
    license_info = pack["license"]
    if license_info.get("requiresAcceptance") and not accept_licenses:
        print(f"License acceptance required: {license_info['name']}", file=sys.stderr)
        print(license_info["url"], file=sys.stderr)
        print(license_info.get("note", ""), file=sys.stderr)
        print("Re-run with --accept-licenses after reviewing these terms.", file=sys.stderr)
        return 2

    estimated = float(pack["estimatedSizeGiB"])
    reserve = float(pack["minimumFreeAfterInstallGiB"])
    free = disk_free_gib(INSTALL_ROOT)
    if free - estimated < reserve:
        print(
            f"Refusing download: {free:.1f} GiB free; pack needs about {estimated:.1f} GiB "
            f"and requires {reserve:.1f} GiB to remain for media/cache.",
            file=sys.stderr,
        )
        return 3

    needs_hub = any("repoId" in source for source in pack["sources"])
    if needs_hub:
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            print("huggingface_hub is missing. Run scripts/bootstrap.ps1 first.", file=sys.stderr)
            return 4

    for source in pack["sources"]:
        destination = INSTALL_ROOT / source["destination"]
        destination.mkdir(parents=True, exist_ok=True)
        # Two kinds of source. Most packs are Hugging Face repositories; some upstreams publish
        # only to GitHub Releases, which snapshot_download cannot reach, so those are declared as
        # an explicit file list with a hash instead.
        if "files" in source:
            if download_files(source["files"], destination) != 0:
                return 5
            continue
        print(f"Downloading {source['repoId']} -> {destination}")
        snapshot_download(
            repo_id=source["repoId"],
            revision=source.get("revision"),
            local_dir=destination,
            allow_patterns=source.get("allowPatterns"),
            max_workers=4,
        )

    state = load_state()
    state["packs"][pack["id"]] = {
        "name": pack["name"],
        "installedAt": datetime.now(timezone.utc).isoformat(),
        "license": license_info,
        "capabilities": pack["capabilities"],
    }
    save_state(state)
    print(f"Installed {pack['name']}. Free disk: {disk_free_gib(INSTALL_ROOT):.1f} GiB")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="AIV Studio model-pack manager")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list")
    install_parser = subparsers.add_parser("install")
    install_parser.add_argument("pack")
    install_parser.add_argument("--accept-licenses", action="store_true")
    args = parser.parse_args()
    items = manifests()
    if args.command == "list":
        return list_packs(items)
    if args.pack not in items:
        print(f"Unknown pack: {args.pack}. Choices: {', '.join(sorted(items))}", file=sys.stderr)
        return 1
    return install(items[args.pack], args.accept_licenses)


if __name__ == "__main__":
    raise SystemExit(main())

