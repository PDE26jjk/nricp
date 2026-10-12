"""The run report has to survive serialization without losing anything."""

from __future__ import annotations

import json


def test_report_round_trips_through_json(nricp_dp):
    energy = {
        "point_to_point": 1.5,
        "point_to_plane": 2.25,
        "smoothness": 0.125,
        "anchor": 0.0,
        "landmark": 4.5,
    }
    correspondence = {
        "total": 4000,
        "accepted": 3980,
        "rejected_distance": 12,
        "rejected_normal": 5,
        "rejected_slope": 3,
        "rejected_masked": 0,
        "distance_mean": 0.031,
        "distance_p95": 0.12,
        "distance_max": 0.4,
    }
    levels = [
        {
            "level": 0,
            "nodes": 100,
            "outer_iterations": 10,
            "inner_iterations": 4,
            "energy_start": 12.0,
            "energy_end": 3.5,
            "correspondence_mean": 0.9,
            "converged": True,
        },
        {
            "level": 1,
            "nodes": 200,
            "outer_iterations": 10,
            "inner_iterations": 6,
            "energy_start": 3.5,
            "energy_end": 1.25,
            "correspondence_mean": 0.3,
            "converged": False,
        },
    ]
    settings = {"levels": 2.0, "landmark_weight": 100.0}

    text = nricp_dp.serialize_report(
        "route_a", "double", 0.25, 1.75, energy, correspondence, levels, settings
    )
    report = json.loads(text)

    assert report["route"] == "route_a"
    assert report["precision"] == "double"
    assert report["version"] == nricp_dp.__version__
    assert report["setup_seconds"] == 0.25
    assert report["solve_seconds"] == 1.75
    assert report["energy"] == {**energy, "total": sum(energy.values())}
    for key, value in correspondence.items():
        assert report["correspondence"][key] == value
    assert len(report["levels"]) == 2
    assert report["levels"][1]["nodes"] == 200
    assert report["levels"][1]["converged"] is False
    assert report["levels"][0]["energy_end"] == 3.5
    assert report["settings"] == settings


def test_non_finite_values_serialize_as_null(nricp_dp):
    """An unreached tolerance must be visible, not silently turned into a number."""
    text = nricp_dp.serialize_report(
        "route_a",
        "float",
        0.0,
        0.0,
        {
            "point_to_point": float("nan"),
            "point_to_plane": 0.0,
            "smoothness": 0.0,
            "anchor": 0.0,
            "landmark": float("inf"),
        },
        {
            "total": 0,
            "accepted": 0,
            "rejected_distance": 0,
            "rejected_normal": 0,
            "rejected_slope": 0,
            "rejected_masked": 0,
            "distance_mean": 0.0,
            "distance_p95": 0.0,
            "distance_max": 0.0,
        },
        [],
        {},
    )
    report = json.loads(text)
    assert report["energy"]["point_to_point"] is None
    assert report["energy"]["landmark"] is None
    assert report["levels"] == []
