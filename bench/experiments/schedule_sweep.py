"""Measure how the native solver's schedule responds, on one prepared fixture.

The reference baseline sets the bar; the schedule has to be found, not guessed.
This sweeps the levers that decide whether the fit resists sliding (the
point-to-point share, the regularizer, the anchor) and prints the metrics the
gates are written in. It runs only the native solver on one fixture, so a whole
sweep costs seconds.

    python bench/experiments/schedule_sweep.py --pair victoria0->victoria2
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import solvers  # noqa: E402
from compare_to_reference import case_from_fixture  # noqa: E402
from geometry import evaluate, load_config  # noqa: E402
from run_bench import load_module  # noqa: E402

BENCH_DIR = Path(__file__).resolve().parents[1]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", default="victoria0->victoria2")
    parser.add_argument("--fixtures", type=Path, default=BENCH_DIR / "fixtures")
    parser.add_argument(
        "--config", type=Path, default=BENCH_DIR / "config" / "bench_config.json"
    )
    parser.add_argument("--module-dir", type=Path, default=BENCH_DIR / "build" / "Release")
    parser.add_argument("--reference-method", default="sumner")
    parser.add_argument("--p2p", default="0.1,1.0,3.0", help="point-to-point ratio grid")
    parser.add_argument(
        "--smoothness", default="1,10,100", help="smoothness weight multiplier grid"
    )
    parser.add_argument("--robust", default="0.02", help="robust threshold grid (fraction)")
    parser.add_argument("--anchor", default="0.001", help="anchor weight grid")
    parser.add_argument("--levels", type=int, default=None, help="override level count")
    parser.add_argument("--inner", type=int, default=None, help="override inner iterations")
    parser.add_argument("--slope", type=float, default=None, help="override min_slope")
    parser.add_argument(
        "--landmark", default="1", help="landmark weight multiplier grid"
    )
    parser.add_argument("--outer", type=int, default=None, help="override outer iterations")
    return parser.parse_args(argv)


def _floats(text: str) -> list[float]:
    return [float(item) for item in text.split(",") if item.strip()]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    path = args.fixtures / (
        f"{args.pair.replace('->', '_')}__{args.reference_method}.npz"
    )
    case = case_from_fixture(config, path, args.pair)
    module = load_module(args.module_dir)

    print(
        f"{'p2p':>5s} {'smooth':>7s} {'anchor':>8s} {'robust':>7s} "
        f"{'corr':>8s} {'%diag':>6s} {'p95':>8s} {'stretch95':>10s} "
        f"{'fold':>7s} {'chamfer':>8s} {'sec':>6s}"
    )
    best: tuple[float, dict] | None = None
    for (
        point_to_point_ratio,
        smoothness_scale,
        anchor,
        robust_fraction,
        landmark_scale,
    ) in itertools.product(
        _floats(args.p2p),
        _floats(args.smoothness),
        _floats(args.anchor),
        _floats(args.robust),
        _floats(args.landmark),
    ):
        base = dict(config["solver"])
        if args.levels is not None:
            base["levels"] = args.levels
        if args.inner is not None:
            base["max_inner_iterations"] = args.inner
        if args.slope is not None:
            base["min_slope"] = args.slope
        if args.outer is not None:
            base["max_outer_iterations"] = args.outer
        solver_cfg = {
            **base,
            "smoothness_weight": [
                value * smoothness_scale for value in base["smoothness_weight"]
            ],
            "landmark_weight": [
                value * landmark_scale for value in base["landmark_weight"]
            ],
            "point_to_point_ratio": point_to_point_ratio,
            "anchor_weight": anchor,
            "robust_threshold_fraction_of_diagonal": robust_fraction,
        }
        started = time.perf_counter()
        native, _ = solvers.solve_module(case, "route_a", solver_cfg, module)
        elapsed = time.perf_counter() - started
        metrics = evaluate(native, case)
        print(
            f"{point_to_point_ratio:5.1f} {smoothness_scale:7.1f} {anchor:8.1e} "
            f"{robust_fraction:7.3f} {metrics['corr_mean']:8.3f} "
            f"{100 * metrics['corr_mean_fraction_of_diagonal']:6.2f} "
            f"{metrics['corr_p95']:8.3f} {metrics['stretch_p95']:10.4f} "
            f"{metrics['foldover']:7.4f} {metrics['chamfer_A2B']:8.4f} {elapsed:6.1f}"
        )
        score = metrics["corr_mean"] + 10.0 * metrics["stretch_p95"]
        if best is None or score < best[0]:
            best = (
                score,
                {
                    "point_to_point_ratio": point_to_point_ratio,
                    "smoothness_scale": smoothness_scale,
                    "anchor_weight": anchor,
                    "robust_threshold_fraction_of_diagonal": robust_fraction,
                    "corr_mean": metrics["corr_mean"],
                    "stretch_p95": metrics["stretch_p95"],
                },
            )
    print(f"\nbest by corr_mean + 10 * stretch_p95: {best[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
