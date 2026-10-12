"""The gate logic itself, which is what every acceptance decision rests on.

It is a pure function of the metrics, the configuration, and the recorded
reference baseline, so all of it is testable without a dataset or a solve.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

BENCH_DIR = Path(__file__).resolve().parents[1] / "bench"
sys.path.insert(0, str(BENCH_DIR))

from geometry import gate_failures, load_config  # noqa: E402

REFERENCE = {
    "best": {
        "method": "sumner",
        "corr_mean": 2.0,
        "corr_p95": 5.0,
        "stretch_p95": 0.4,
        "foldover": 0.05,
    }
}


def _metrics(**overrides) -> dict:
    values = {
        "corr_mean": 2.0,
        "corr_p95": 5.0,
        "stretch_p95": 0.4,
        "foldover": 0.05,
    }
    values.update(overrides)
    return values


@pytest.fixture(scope="module")
def config() -> dict:
    return load_config(BENCH_DIR / "config" / "bench_config.json")


def test_matching_the_reference_passes(config):
    assert gate_failures(_metrics(), config, REFERENCE) == []


def test_non_finite_metrics_fail_instead_of_passing_silently(config):
    """NaN compares false against every limit, so it has to be caught explicitly."""
    for value in (float("nan"), float("inf"), float("-inf")):
        failures = gate_failures(_metrics(corr_mean=value), config, REFERENCE)
        assert failures, f"{value} slipped through the gates"
        assert "not finite" in failures[0]


def test_a_missing_baseline_is_not_a_pass(config):
    failures = gate_failures(_metrics(), config, None)
    assert failures and "baseline" in failures[0]


def test_each_metric_can_fail_and_names_the_reference(config):
    relative = config["gates"]["relative"]
    over_the_line = {
        "corr_mean": REFERENCE["best"]["corr_mean"] * relative["corr_mean_ratio"] * 1.01,
        "corr_p95": REFERENCE["best"]["corr_p95"] * relative["corr_p95_ratio"] * 1.01,
        "stretch_p95": REFERENCE["best"]["stretch_p95"]
        + relative["stretch_p95_margin"] * 2.0,
        "foldover": REFERENCE["best"]["foldover"] + relative["foldover_margin"] * 2.0,
    }
    for name, value in over_the_line.items():
        failures = gate_failures(_metrics(**{name: value}), config, REFERENCE)
        assert any(name in failure for failure in failures), name
        assert any("reference" in failure for failure in failures), name


def test_the_boundary_itself_passes(config):
    """Exactly at the limit is inside the gate: the comparison is strict."""
    relative = config["gates"]["relative"]
    at_the_line = {
        "corr_mean": REFERENCE["best"]["corr_mean"] * relative["corr_mean_ratio"],
        "stretch_p95": REFERENCE["best"]["stretch_p95"]
        + relative["stretch_p95_margin"],
        "foldover": REFERENCE["best"]["foldover"] + relative["foldover_margin"],
    }
    assert gate_failures(_metrics(**at_the_line), config, REFERENCE) == []


def test_the_recorded_baseline_comes_from_a_full_reference_run():
    """A partial run must never become the anchor (it would pass everything)."""
    path = BENCH_DIR / "config" / "reference_baseline.json"
    if not path.is_file():
        pytest.skip("no reference baseline yet")
    with open(path, encoding="utf-8") as handle:
        baseline = json.load(handle)
    assert baseline.get("config_sha256"), "baseline has no configuration fingerprint"
    config = load_config(BENCH_DIR / "config" / "bench_config.json")
    assert {pair["name"] for pair in config["pairs"]} <= set(baseline["pairs"])
    for name, entry in baseline["pairs"].items():
        assert "amberg" in entry["methods"] and "sumner" in entry["methods"], name
        best = entry["best"]
        assert best["corr_mean"] == min(
            entry["methods"][method]["corr_mean"] for method in entry["methods"]
        ), name
