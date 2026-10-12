"""Compare the native solver with the recorded reference, from prepared fixtures.

This is the acceptance run that costs nothing but the native solve: the fixtures
already carry the prepared inputs (rigidly aligned template, target, ground
truth, landmarks, rigid transform and threshold) and the reference result, and
the gates compare the native run against the recorded reference baseline. No
dataset preparation and no reference solve are needed.

    python bench/compare_to_reference.py --pairs victoria0->victoria2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import solvers
from geometry import Case, evaluate, gate_failures, load_config
from run_bench import load_module, load_reference_baseline, select_pairs

BENCH_DIR = Path(__file__).resolve().parent


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", default="all")
    parser.add_argument("--fixtures", type=Path, default=BENCH_DIR / "fixtures")
    parser.add_argument(
        "--config", type=Path, default=BENCH_DIR / "config" / "bench_config.json"
    )
    parser.add_argument("--out", type=Path, default=BENCH_DIR / "out")
    parser.add_argument("--module-dir", type=Path, default=None)
    parser.add_argument(
        "--method",
        default="route_a",
        choices=("route_a", "route_b"),
        help="which native route to run (route_b is the accelerated one)",
    )
    parser.add_argument(
        "--reference-method",
        default="sumner",
        help="which stored reference result to compare against",
    )
    return parser.parse_args(argv)


def case_from_fixture(config: dict, path: Path, name: str) -> Case:
    with np.load(path, allow_pickle=False) as data:
        return Case(
            name=name,
            template_vertices=data["template_vertices"],
            template_faces=data["template_faces"],
            target_vertices=data["target_vertices"],
            target_faces=data["target_faces"],
            ground_truth_vertices=data["ground_truth_vertices"],
            rigid_transform=data["rigid_transform"],
            aligned_template_vertices=data["aligned_template_vertices"],
            landmark_indices=data["landmark_indices"],
            landmark_targets=data["landmark_targets"],
            distance_threshold=float(data["distance_threshold"]),
            diagonal=float(data["diagonal"]),
            config=config,
        )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    pairs = select_pairs(config, args.pairs)
    baseline = load_reference_baseline(config)
    module = load_module(args.module_dir)
    args.out.mkdir(parents=True, exist_ok=True)

    print(
        f"{'pair':22s} {'method':8s} {'corr_mean':>9s} {'%diag':>6s} {'p95':>7s} "
        f"{'stretch95':>9s} {'fold':>7s} {'chamfer':>8s} {'sec':>6s}  gate"
    )
    failures: list[str] = []
    rows: list[dict] = []
    for pair in pairs:
        name = pair["name"]
        path = args.fixtures / f"{name.replace('->', '_')}__{args.reference_method}.npz"
        if not path.is_file():
            print(f"{name:22s} missing fixture {path.name}")
            failures.append(f"{name}: no fixture for {args.reference_method}")
            continue
        case = case_from_fixture(config, path, name)
        with np.load(path, allow_pickle=False) as data:
            reference_result = np.asarray(data["result_vertices"], dtype=np.float64)

        started = time.perf_counter()
        native, report = solvers.solve_module(case, args.method, config["solver"], module)
        solve_s = time.perf_counter() - started
        (args.out / f"report_{name.replace('->', '_')}__{args.method}.json").write_text(
            report, encoding="utf-8"
        )

        metrics = evaluate(native, case)
        reference_metrics = evaluate(reference_result, case)
        violated = gate_failures(metrics, config, baseline.get(name))
        failures.extend(f"{name} [{args.method}]: {item}" for item in violated)
        rows.append(
            {
                "pair": name,
                "method": args.method,
                "reference_method": args.reference_method,
                "solve_s": solve_s,
                **{
                    key: metrics[key]
                    for key in (
                        "corr_mean",
                        "corr_p95",
                        "corr_max",
                        "corr_mean_fraction_of_diagonal",
                        "chamfer_A2B",
                        "stretch_mean",
                        "stretch_p95",
                        "foldover",
                    )
                },
                "reference_corr_mean": reference_metrics["corr_mean"],
                "gate_pass": not violated,
                "failures": "; ".join(violated),
            }
        )
        print(
            f"{name:22s} {args.method:8s} {metrics['corr_mean']:9.4f} "
            f"{100 * metrics['corr_mean_fraction_of_diagonal']:6.2f} "
            f"{metrics['corr_p95']:7.3f} {metrics['stretch_p95']:9.4f} "
            f"{metrics['foldover']:7.4f} {metrics['chamfer_A2B']:8.4f} "
            f"{solve_s:6.1f}  {'pass' if not violated else 'FAIL'}"
            f"   (reference corr_mean {reference_metrics['corr_mean']:.4f})"
        )

    summary = args.out / f"module_comparison_{args.method}.json"
    summary.write_text(
        json.dumps(
            {
                "method": args.method,
                "reference_method": args.reference_method,
                "version": module.__version__,
                "build_info": dict(module.build_info()),
                "rows": rows,
                "gate_failures": failures,
                "passed": not failures,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nsummary -> {summary}")
    if failures:
        print("\nGATE FAILURES:")
        for item in failures:
            print(f"  - {item}")
        return 1
    print("all gates passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
