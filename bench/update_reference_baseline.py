"""Record the reference baseline that the acceptance gates compare against.

Reads a reference results table produced by ``run_bench.py --backend reference``
and writes ``config/reference_baseline.json``: for every pair, each method's
metrics plus the best of them — that best result is the bar the native solver
has to match within the configured tolerance.

The table, not the baseline file, is the expensive artefact: this script only
transcribes an existing run, so refreshing the baseline after a reference run
costs a second.

    python bench/update_reference_baseline.py --results bench/out/results_reference.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

METRICS = (
    "corr_mean",
    "corr_p95",
    "corr_max",
    "corr_mean_fraction_of_diagonal",
    "chamfer_A2B",
    "stretch_mean",
    "stretch_p95",
    "foldover",
)

# The baseline is the anchor every acceptance decision hangs on, so it is only
# written from a run that exercised the pinned reference methods on every pair.
REQUIRED_METHODS = ("amberg", "sumner")

# These parts of the configuration define what the numbers mean; a change to any
# of them invalidates a recorded baseline.
FINGERPRINT_KEYS = (
    "pairs",
    "template_vertices",
    "target_vertices",
    "landmarks",
    "rigid_init",
    "distance_threshold",
    "reference",
)

BENCH_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS = BENCH_DIR / "out" / "results_reference.csv"
DEFAULT_META = BENCH_DIR / "out" / "results_reference.meta.json"
DEFAULT_OUT = BENCH_DIR / "config" / "reference_baseline.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--meta", type=Path, default=DEFAULT_META)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--config", type=Path, default=BENCH_DIR / "config" / "bench_config.json")
    parser.add_argument(
        "--manifest", type=Path, default=BENCH_DIR / "config" / "data_manifest.json"
    )
    return parser.parse_args(argv)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def config_fingerprint(config: dict) -> str:
    relevant = {key: config[key] for key in FINGERPRINT_KEYS if key in config}
    return hashlib.sha256(
        json.dumps(relevant, sort_keys=True).encode("utf-8")
    ).hexdigest()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.results.is_file():
        raise SystemExit(
            f"no reference results at {args.results}; run "
            "bench/run_bench.py --backend reference first"
        )

    with open(args.results, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"{args.results} has no rows")

    with open(args.config, encoding="utf-8") as handle:
        config = json.load(handle)
    expected_pairs = {pair["name"] for pair in config["pairs"]}

    pairs: dict[str, dict] = {}
    for row in rows:
        pair = pairs.setdefault(row["case"], {"methods": {}})
        pair["methods"][row["method"]] = {
            **{metric: float(row[metric]) for metric in METRICS},
            "solve_s": float(row["solve_s"]),
        }

    for pair_name, entry in pairs.items():
        # The bar is the best correspondence error any reference method reached
        # on this pair; the other metrics are taken from that same method.
        best_method = min(
            entry["methods"], key=lambda name: entry["methods"][name]["corr_mean"]
        )
        entry["best"] = {"method": best_method, **entry["methods"][best_method]}

    # Refuse to anchor acceptance on a partial run: a table produced with
    # `--methods rigid` alone would make the rigid baseline the bar and let
    # anything pass.
    missing_pairs = expected_pairs - set(pairs)
    if missing_pairs:
        raise SystemExit(
            f"{args.results} does not cover every pinned pair; missing "
            f"{sorted(missing_pairs)}. Re-run the reference backend for all pairs."
        )
    for pair_name, entry in pairs.items():
        missing = [m for m in REQUIRED_METHODS if m not in entry["methods"]]
        if missing:
            raise SystemExit(
                f"{args.results} is missing the reference method(s) {missing} for "
                f"{pair_name}; the baseline must come from a full reference run."
            )

    versions = {}
    if args.meta.is_file():
        with open(args.meta, encoding="utf-8") as handle:
            versions = json.load(handle).get("versions", {})

    payload = {
        "created": datetime.now(timezone.utc).isoformat(),
        "source_results": args.results.name,
        "versions": versions,
        "config_sha256": config_fingerprint(config),
        "data_manifest_sha256": file_sha256(args.manifest)
        if args.manifest.is_file()
        else None,
        "pairs": pairs,
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)

    for pair_name, entry in sorted(pairs.items()):
        best = entry["best"]
        print(
            f"{pair_name:22s} best {best['method']:8s} corr_mean {best['corr_mean']:.4f} "
            f"foldover {best['foldover']:.4f} stretch_p95 {best['stretch_p95']:.4f}"
        )
    print(f"\nbaseline -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
