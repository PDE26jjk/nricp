"""Regenerate the golden fixtures the native solver is compared against.

For every pinned pair and reference method this writes
``bench/fixtures/<pair>__<method>.npz`` with the prepared inputs, the landmark
data, the rigid transform, the reference result, and the pinned configuration
plus the versions that produced it. The fixtures are regenerated locally and
never committed: they are derived from datasets that are not ours to
redistribute.

Each fixture also gets a content hash (over the arrays, not the file bytes, so
the archive timestamps do not matter); ``fixtures/index.json`` collects them and
is what the determinism check compares between two runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import solvers
from geometry import build_case, load_config
from run_bench import DEFAULT_CONFIG, DEFAULT_DATA, select_pairs, versions

BENCH_DIR = Path(__file__).resolve().parent
DEFAULT_FIXTURES = BENCH_DIR / "fixtures"

ARRAY_FIELDS = (
    "template_vertices",
    "template_faces",
    "target_vertices",
    "target_faces",
    "ground_truth_vertices",
    "rigid_transform",
    "aligned_template_vertices",
    "landmark_indices",
    "landmark_targets",
    "result_vertices",
)


def content_hash(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in ARRAY_FIELDS:
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", default="amberg,sumner")
    parser.add_argument("--pairs", default="all")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_FIXTURES)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    pairs = select_pairs(config, args.pairs)
    methods = [method.strip() for method in args.methods.split(",") if method.strip()]
    args.out.mkdir(parents=True, exist_ok=True)
    if not args.data.is_dir():
        raise SystemExit(
            f"dataset directory {args.data} does not exist; "
            "run bench/prepare_data.py first (see bench/README.md)"
        )

    version_info = versions(None)
    config_json = json.dumps(config, sort_keys=True)
    index: dict[str, dict] = {}

    for pair in pairs:
        case = build_case(config, args.data, pair)
        for method in methods:
            result = solvers.solve("reference", case, method, config, None)
            arrays = {
                "template_vertices": case.template_vertices,
                "template_faces": case.template_faces,
                "target_vertices": case.target_vertices,
                "target_faces": case.target_faces,
                "ground_truth_vertices": case.ground_truth_vertices,
                "rigid_transform": case.rigid_transform,
                "aligned_template_vertices": case.aligned_template_vertices,
                "landmark_indices": case.landmark_indices,
                "landmark_targets": case.landmark_targets,
                "result_vertices": result,
            }
            slug = f"{pair['name'].replace('->', '_')}__{method}"
            path = args.out / f"{slug}.npz"
            np.savez_compressed(
                path,
                distance_threshold=np.float64(case.distance_threshold),
                diagonal=np.float64(case.diagonal),
                config_json=np.array(config_json),
                versions_json=np.array(json.dumps(version_info, sort_keys=True)),
                **arrays,
            )
            index[slug] = {
                "path": path.name,
                "pair": pair["name"],
                "method": method,
                "content_sha256": content_hash(arrays),
                "template_vertices": int(len(case.template_vertices)),
                "target_vertices": int(len(case.target_vertices)),
                "distance_threshold": float(case.distance_threshold),
            }
            print(f"wrote {path.name}  ({index[slug]['content_sha256'][:12]})")

    with open(args.out / "index.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "created": datetime.now(timezone.utc).isoformat(),
                "config_path": str(args.config),
                "versions": version_info,
                "fixtures": index,
            },
            handle,
            indent=2,
        )
    print(f"index -> {args.out / 'index.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
