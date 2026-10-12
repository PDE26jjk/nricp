"""Measure what the rigid initialization is worth.

Compares the pinned landmark-based (Kabsch) initialization against the
surface-ICP alternative, at the pinned schedule and at shortened schedules, so
the claim "a landmark fit saves iterations" is a measured statement rather than
an assumption.

    python bench/experiments/init_alignment.py --pairs victoria0->victoria2
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import solvers  # noqa: E402
from geometry import build_case, evaluate, load_config  # noqa: E402
from run_bench import DEFAULT_CONFIG, DEFAULT_DATA, select_pairs  # noqa: E402

# Shortened variants of the pinned reference schedules. Each entry keeps the
# weight shape of the pinned schedule and only cuts the iteration budget.
SHORT_SCHEDULES = {
    "pinned": None,
    "half_iterations": {
        "amberg": [
            [0.01, 100.0, 0.5, 5],
            [0.02, 50.0, 0.5, 5],
            [0.03, 20.0, 0.5, 5],
            [0.01, 10.0, 0.0, 5],
        ],
        "sumner": [[1, 0.001, 1.0, 1000, 0], [10, 0.001, 1.0, 1000, 0]],
    },
    "half_levels": {
        "amberg": [[0.02, 50.0, 0.5, 10], [0.03, 20.0, 0.5, 10]],
        "sumner": [[10, 0.001, 1.0, 1000, 0], [100, 0.001, 1.0, 1000, 0]],
    },
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", default="victoria0->victoria2,david0->david1")
    parser.add_argument("--methods", default="amberg,sumner")
    parser.add_argument("--inits", default="landmark_procrustes,icp")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    base_config = load_config(args.config)
    pairs = select_pairs(base_config, args.pairs)
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    inits = [item.strip() for item in args.inits.split(",") if item.strip()]

    print(
        f"{'pair':22s} {'init':20s} {'schedule':16s} {'method':8s} "
        f"{'corr_mean':>10s} {'p95':>8s} {'stretch':>8s} {'fold':>7s} {'time':>7s}"
    )
    for pair in pairs:
        for init in inits:
            config = copy.deepcopy(base_config)
            config["rigid_init"]["method"] = init
            case = build_case(config, args.data, pair)
            for schedule_name, override in SHORT_SCHEDULES.items():
                reference = copy.deepcopy(base_config["reference"])
                if override is not None:
                    reference["amberg"]["steps"] = override["amberg"]
                    reference["sumner"]["steps"] = override["sumner"]
                config["reference"] = reference
                for method in methods:
                    started = time.perf_counter()
                    result = solvers.solve_reference(
                        case, method, config["reference"]
                    )
                    elapsed = time.perf_counter() - started
                    metrics = evaluate(result, case)
                    print(
                        f"{case.name:22s} {init:20s} {schedule_name:16s} "
                        f"{method:8s} {metrics['corr_mean']:10.4f} "
                        f"{metrics['corr_p95']:8.4f} {metrics['stretch_p95']:8.4f} "
                        f"{metrics['foldover']:7.4f} {elapsed:7.1f}s"
                    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
