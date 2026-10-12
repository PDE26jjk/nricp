"""The benchmark harness has to be pinned, reproducible, and honest about data.

These tests need no dataset: they exercise the configuration contract, the
dataset-missing path, and — when fixtures happen to exist locally — that they
still match the recorded content hashes.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = REPO_ROOT / "bench"
CONFIG_PATH = BENCH_DIR / "config" / "bench_config.json"


@pytest.fixture(scope="module")
def config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as handle:
        return json.load(handle)


def test_pinned_configuration_covers_the_acceptance_setup(config):
    """Every quantity the gates depend on must be pinned in the config file."""
    assert [pair["name"] for pair in config["pairs"]] == [
        "cat0->cat3",
        "david0->david1",
        "michael0->michael2",
        "victoria0->victoria2",
    ]
    assert config["landmarks"] == 25
    assert config["template_vertices"] and config["target_vertices"]
    assert config["distance_threshold"]["fraction_of_diagonal"] == 0.05

    # The landmark schedule is what separates an acceptable result from a
    # several-times-worse one, so it is pinned for both backends.
    amberg_steps = config["reference"]["amberg"]["steps"]
    assert [step[1] for step in amberg_steps] == [100.0, 50.0, 20.0, 10.0]
    # The native solver's own schedule is under tuning, so only its shape is a
    # contract: one entry per level, positive, and consistent across the arrays.
    levels = config["solver"]["levels"]
    for key in ("smoothness_weight", "point_to_plane_weight", "landmark_weight"):
        assert len(config["solver"][key]) == levels, key
        assert all(value > 0.0 for value in config["solver"][key]), key

    # Acceptance is anchored on the recorded reference baseline; the absolute
    # targets are reported but do not gate.
    assert config["gates"]["baseline"] == "config/reference_baseline.json"
    for key in ("corr_mean_ratio", "corr_p95_ratio", "stretch_p95_margin", "foldover_margin"):
        assert key in config["gates"]["relative"]
    for key in ("corr_mean", "corr_mean_fraction_of_diagonal", "stretch_p95", "foldover"):
        assert key in config["gates"]["targets"]


def test_reference_baseline_is_usable(config):
    """The gates cannot be evaluated without a per-pair reference record."""
    path = BENCH_DIR / config["gates"]["baseline"]
    if not path.is_file():
        pytest.skip("no reference baseline yet; run a reference pass and update_reference_baseline.py")
    with open(path, encoding="utf-8") as handle:
        pairs = json.load(handle)["pairs"]
    assert {pair["name"] for pair in config["pairs"]} <= set(pairs)
    for name, entry in pairs.items():
        assert "best" in entry and "method" in entry["best"], name
        assert entry["best"]["corr_mean"] > 0, name


def test_prepare_data_explains_what_is_missing(tmp_path):
    """With no source the script must say what to supply, not crash."""
    result = subprocess.run(
        [sys.executable, str(BENCH_DIR / "prepare_data.py"), "--dest", str(tmp_path / "d")],
        capture_output=True,
        text=True,
        # A developer shell that points NRICP_DATA_SOURCE somewhere would
        # otherwise turn this into a normal run and change the outcome.
        env={**os.environ, "NRICP_DATA_SOURCE": ""},
    )
    assert result.returncode == 2
    assert "not part of the repository" in result.stdout
    assert "Required files" in result.stdout


def test_data_manifest_matches_the_pinned_pairs(config):
    """The committed manifest must describe exactly the files the pairs need."""
    manifest_path = BENCH_DIR / "config" / "data_manifest.json"
    if not manifest_path.is_file():
        pytest.skip("no data manifest yet; run bench/prepare_data.py --write-manifest")
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    expected = {
        f"{pair[role]}{suffix}"
        for pair in config["pairs"]
        for role in ("template", "target")
        for suffix in (".vert", ".tri")
    }
    assert expected == set(manifest["files"])
    for name, entry in manifest["files"].items():
        assert entry["size"] > 0, name
        assert len(entry["sha256"]) == 64, name


def test_golden_fixtures_match_their_recorded_hashes():
    """Fixtures are regenerated locally; if present they must be self-consistent."""
    index_path = BENCH_DIR / "fixtures" / "index.json"
    if not index_path.is_file():
        pytest.skip("no fixtures yet; run bench/golden_regen.py")
    sys.path.insert(0, str(BENCH_DIR))
    import numpy as np
    from golden_regen import ARRAY_FIELDS, content_hash

    with open(index_path, encoding="utf-8") as handle:
        index = json.load(handle)
    assert index["fixtures"], "fixture index is empty"
    for slug, entry in index["fixtures"].items():
        path = BENCH_DIR / "fixtures" / entry["path"]
        assert path.is_file(), f"{slug}: fixture file is missing"
        with np.load(path, allow_pickle=False) as data:
            arrays = {name: data[name] for name in ARRAY_FIELDS}
            assert content_hash(arrays) == entry["content_sha256"], slug
