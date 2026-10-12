"""Run the wrapping benchmark over the pinned pairs and report quality.

Every row is the result of one prepared case solved by one backend method, with
the metrics the gates are expressed in, the timings, and the gate verdict. The
run fails (non-zero exit) when any gate is violated, naming the pair and the
gate.

    python bench/run_bench.py --backend reference --methods amberg,sumner
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import solvers
from geometry import build_case, evaluate, gate_failures, load_config, save_error_map
from update_reference_baseline import config_fingerprint

BENCH_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = BENCH_DIR / "config" / "bench_config.json"
DEFAULT_DATA = BENCH_DIR / "data" / "hires"
DEFAULT_OUT = BENCH_DIR / "out"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="reference", choices=("reference", "module"))
    parser.add_argument("--methods", default="amberg,sumner")
    parser.add_argument("--pairs", default="all")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--tag", default="")
    parser.add_argument("--module-dir", type=Path, default=None)
    return parser.parse_args(argv)


def load_module(module_dir: Path | None):
    """Import the project's extension module when a module backend is asked for."""
    if module_dir is not None:
        sys.path.insert(0, str(module_dir))
    try:
        import nricp_dp

        return nricp_dp
    except ImportError as exc:  # pragma: no cover - depends on local build
        raise SystemExit(
            f"nricp_dp is not importable ({exc}); pass --module-dir <build dir>"
        ) from exc


def versions(module) -> dict:
    import scipy
    import trimesh

    info = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "trimesh": trimesh.__version__,
        "platform": platform.platform(),
    }
    if module is not None:
        info["nricp_dp"] = module.__version__
        info["build_info"] = dict(module.build_info())
    return info


def select_pairs(config: dict, selection: str) -> list[dict]:
    pairs = config["pairs"]
    if selection == "all":
        return pairs
    wanted = {name.strip() for name in selection.split(",") if name.strip()}
    chosen = [pair for pair in pairs if pair["name"] in wanted]
    missing = wanted - {pair["name"] for pair in chosen}
    if missing:
        raise SystemExit(f"unknown pair(s): {', '.join(sorted(missing))}")
    return chosen


def load_reference_baseline(config: dict) -> dict:
    """Per-pair reference results the gates are anchored on."""
    path = BENCH_DIR / config["gates"].get(
        "baseline", "config/reference_baseline.json"
    )
    if not path.is_file():
        raise SystemExit(
            f"no reference baseline at {path}; produce one with "
            "bench/run_bench.py --backend reference then "
            "bench/update_reference_baseline.py"
        )
    with open(path, encoding="utf-8") as handle:
        baseline = json.load(handle)

    # The baseline only means something for the configuration that produced it.
    if baseline.get("config_sha256") != config_fingerprint(config):
        raise SystemExit(
            f"{path} was recorded for a different benchmark configuration; "
            "re-run the reference backend and bench/update_reference_baseline.py"
        )
    return baseline["pairs"]


def pretty(value: float) -> str:
    return f"{value:.5g}"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    pairs = select_pairs(config, args.pairs)
    methods = [method.strip() for method in args.methods.split(",") if method.strip()]
    baseline = load_reference_baseline(config)
    args.out.mkdir(parents=True, exist_ok=True)

    if not args.data.is_dir():
        raise SystemExit(
            f"dataset directory {args.data} does not exist; "
            "run bench/prepare_data.py first (see bench/README.md)"
        )

    module = load_module(args.module_dir) if args.backend == "module" else None
    # The reference backend *is* the baseline: its own rows cannot be judged
    # against itself, so its gates are reported rather than enforced.
    enforce_gates = args.backend != "reference"
    tag = args.tag or args.backend
    csv_path = args.out / f"results_{tag}.csv"
    meta_path = args.out / f"results_{tag}.meta.json"

    fieldnames = [
        "case",
        "backend",
        "method",
        "corr_mean",
        "corr_p95",
        "corr_max",
        "corr_mean_fraction_of_diagonal",
        "chamfer_A2B",
        "stretch_mean",
        "stretch_p95",
        "foldover",
        "setup_s",
        "solve_s",
        "gate_pass",
        "failures",
    ]

    rows: list[dict] = []
    failures: list[str] = []
    for pair in pairs:
        started = time.perf_counter()
        case = build_case(config, args.data, pair)
        setup_s = time.perf_counter() - started
        print(
            f"{case.name}: template {len(case.template_vertices)}v/"
            f"{len(case.template_faces)}f, target {len(case.target_vertices)}v, "
            f"diagonal {case.diagonal:.1f}, threshold {case.distance_threshold:.2f}, "
            f"{len(case.landmark_indices)} landmarks (prep {setup_s:.1f}s)"
        )
        for method in methods:
            started = time.perf_counter()
            result, run_report = solvers.solve(args.backend, case, method, config, module)
            solve_s = time.perf_counter() - started
            if run_report is not None:
                # The native backend's own view of the run is kept next to the
                # table: the report is what a stored number is traced through.
                report_path = args.out / f"report_{case.name.replace('->', '_')}__{method}.json"
                report_path.write_text(run_report, encoding="utf-8")
            metrics = evaluate(result, case)
            violated = gate_failures(metrics, config, baseline.get(case.name))
            failures.extend(f"{case.name} [{method}]: {item}" for item in violated)
            save_error_map(
                args.out / f"error_{case.name.replace('->', '_')}__{method}",
                result,
                case,
                f"{case.name} — {args.backend}/{method}",
            )
            rows.append(
                {
                    "case": case.name,
                    "backend": args.backend,
                    "method": method,
                    **{key: metrics[key] for key in (
                        "corr_mean",
                        "corr_p95",
                        "corr_max",
                        "corr_mean_fraction_of_diagonal",
                        "chamfer_A2B",
                        "stretch_mean",
                        "stretch_p95",
                        "foldover",
                    )},
                    "setup_s": setup_s,
                    "solve_s": solve_s,
                    "gate_pass": not violated,
                    "failures": "; ".join(violated),
                }
            )
            print(
                f"  {method:8s} corr_mean {pretty(metrics['corr_mean'])} "
                f"({metrics['corr_mean_fraction_of_diagonal'] * 100:.2f}% diag) "
                f"p95 {pretty(metrics['corr_p95'])} "
                f"stretch_p95 {pretty(metrics['stretch_p95'])} "
                f"foldover {metrics['foldover']:.4f} "
                f"chamfer {pretty(metrics['chamfer_A2B'])} "
                f"in {solve_s:.1f}s"
                + ("" if not violated else "  GATE FAIL")
            )

    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "created": datetime.now(timezone.utc).isoformat(),
                "backend": args.backend,
                "methods": methods,
                "pairs": [pair["name"] for pair in pairs],
                "config_path": args.config.name,
                "config": config,
                "config_sha256": config_fingerprint(config),
                "versions": versions(module),
                "results_csv": csv_path.name,
                "gate_failures": failures,
                "passed": not failures,
            },
            handle,
            indent=2,
        )

    print(f"\nresults -> {csv_path}")
    print(f"metadata -> {meta_path}")
    if failures and enforce_gates:
        print("\nGATE FAILURES:")
        for item in failures:
            print(f"  - {item}")
        return 1
    if failures:
        print("\ngates reported (reference backend, not enforced):")
        for item in failures:
            print(f"  - {item}")
        return 0
    print("all gates passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
