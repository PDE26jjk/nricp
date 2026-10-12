"""Prepare the benchmark dataset locally.

The datasets are not distributed with this repository. Point ``--source`` at a
directory that holds the high-resolution pairs (``<name>.vert`` / ``<name>.tri``,
optionally gzip-compressed) and the files are verified against
``config/data_manifest.json`` — file size and sha256 — before being copied into
``bench/data/hires``. Without a usable source the script says exactly what is
missing instead of failing obscurely, and the harnesses in this directory refuse
to run.

    # first time, from an existing copy of the dataset
    python bench/prepare_data.py --source <dir> --write-manifest

    # afterwards (verification only, or after moving the source)
    python bench/prepare_data.py --source <dir>
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

from geometry import load_config

BENCH_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = BENCH_DIR / "config" / "bench_config.json"
DEFAULT_MANIFEST = BENCH_DIR / "config" / "data_manifest.json"
DEFAULT_DEST = BENCH_DIR / "data" / "hires"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(os.environ["NRICP_DATA_SOURCE"])
        if os.environ.get("NRICP_DATA_SOURCE")
        else None,
        help="directory holding the high-resolution pairs (or NRICP_DATA_SOURCE)",
    )
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--write-manifest",
        action="store_true",
        help="record the source files' sizes and hashes as the manifest",
    )
    parser.add_argument("--force", action="store_true", help="re-copy existing files")
    return parser.parse_args(argv)


def required_names(config: dict) -> list[str]:
    names: list[str] = []
    for pair in config["pairs"]:
        for role in ("template", "target"):
            if pair[role] not in names:
                names.append(pair[role])
    return sorted(names)


def locate(source: Path, name: str, suffix: str) -> tuple[Path, bool]:
    """Find ``name.suffix`` (or its .gz form); returns (path, is_gzipped)."""
    plain = source / f"{name}{suffix}"
    if plain.is_file():
        return plain, False
    compressed = source / f"{name}{suffix}.gz"
    if compressed.is_file():
        return compressed, True
    raise FileNotFoundError(plain.name)


def read_bytes(path: Path, is_gzipped: bool) -> bytes:
    if not is_gzipped:
        return path.read_bytes()
    with gzip.open(path, "rb") as handle:
        return handle.read()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def destination_matches(destination: Path, expected: dict) -> bool:
    """Is the file already in place, and still exactly the expected one?"""
    if not destination.is_file():
        return False
    if destination.stat().st_size != expected["size"]:
        return False
    return digest(destination.read_bytes()) == expected["sha256"]


def build_manifest(source: Path, names: list[str]) -> dict:
    files: dict[str, dict] = {}
    for name in names:
        for suffix in (".vert", ".tri"):
            path, is_gzipped = locate(source, name, suffix)
            data = read_bytes(path, is_gzipped)
            files[f"{name}{suffix}"] = {
                "size": len(data),
                "sha256": digest(data),
                "source": path.name,
            }
    return {"files": files}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    names = required_names(config)
    wanted = [f"{name}{suffix}" for name in names for suffix in (".vert", ".tri")]

    if args.source is None or not args.source.is_dir():
        print("The benchmark dataset is not part of the repository.")
        print(f"Missing: a source directory (--source <dir> or NRICP_DATA_SOURCE).")
        print(f"Required files ({len(wanted)}): {', '.join(wanted)}")
        print("They must be the high-resolution pairs, gzip-compressed or not.")
        return 2

    manifest: dict
    if args.write_manifest or not args.manifest.is_file():
        manifest = build_manifest(args.source, names)
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        with open(args.manifest, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
        print(f"manifest -> {args.manifest}")
    else:
        with open(args.manifest, encoding="utf-8") as handle:
            manifest = json.load(handle)

    args.dest.mkdir(parents=True, exist_ok=True)
    copied = verified = 0
    problems: list[str] = []
    for entry in wanted:
        expected = manifest["files"].get(entry)
        if expected is None:
            problems.append(f"{entry}: not in the manifest")
            continue
        name, suffix = entry.rsplit(".", 1)
        try:
            path, is_gzipped = locate(args.source, name, f".{suffix}")
        except FileNotFoundError:
            problems.append(f"{entry}: missing at the source")
            continue
        data = read_bytes(path, is_gzipped)
        if len(data) != expected["size"] or digest(data) != expected["sha256"]:
            problems.append(f"{entry}: size or hash does not match the manifest")
            continue
        destination = args.dest / entry
        # The destination is verified too, not only the source: a partial copy or
        # a hand-edited file left in the data directory would otherwise be used
        # silently by every later run while still being attributed to the pinned
        # dataset.
        if args.force or not destination_matches(destination, expected):
            destination.write_bytes(data)
            copied += 1
        else:
            verified += 1
        print(f"  ok  {entry}  ({len(data)} bytes)")

    print(f"\n{copied} copied, {verified} already present, at {args.dest}")
    if problems:
        print("\nproblems:")
        for item in problems:
            print(f"  - {item}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
