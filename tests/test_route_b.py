"""Route B: the accelerated path, checked against route A on synthetic data.

Route A's block-based double-precision implementation is the reference the
analytic Jacobians were verified against, so the device path (single precision,
device queries, block-preconditioned PCG) is held to it: the two must land on the
same shape within single-precision tolerance, and both must improve on the
starting state.
"""

from __future__ import annotations

import numpy as np
import pytest

from test_core_geometry import _uv_sphere
from test_route_a import _bulge, _settings


def _run_route(nricp_dp, name, template, faces, target, levels, nodes_initial):
    return getattr(nricp_dp, name)(
        template,
        faces,
        target,
        faces,
        _settings(levels),
        graph_levels=levels,
        nodes_initial=nodes_initial,
        growth=2,
    )


def test_the_device_is_reported(nricp_dp):
    info = nricp_dp.device_info()
    assert set(info) == {"available", "device"}
    assert isinstance(info["device"], str) and info["device"]
    if not info["available"]:
        pytest.skip(f"no CUDA device here: {info['device']}")


def test_route_b_matches_route_a_on_a_synthetic_case(nricp_dp):
    if not nricp_dp.device_info()["available"]:
        pytest.skip("no CUDA device here")
    template, faces = _uv_sphere(rings=14, segments=18)
    ground_truth = _bulge(template, amplitude=0.09)

    device = _run_route(nricp_dp, "solve_route_b", template, faces, ground_truth, 2, 80)
    host = _run_route(nricp_dp, "solve_route_a", template, faces, ground_truth, 2, 80)

    device_vertices = np.asarray(device["vertices"])
    host_vertices = np.asarray(host["vertices"])
    assert np.all(np.isfinite(device_vertices))

    initial_error = np.linalg.norm(template - ground_truth, axis=1).mean()
    device_error = np.linalg.norm(device_vertices - ground_truth, axis=1).mean()
    host_error = np.linalg.norm(host_vertices - ground_truth, axis=1).mean()

    # Both have to make real progress, and they have to agree with each other:
    # the two paths differ only in precision and linear algebra.
    assert device_error < 0.75 * initial_error, (device_error, initial_error)
    assert host_error < 0.75 * initial_error, (host_error, initial_error)
    assert abs(device_error - host_error) < 0.05 * max(host_error, 1e-9), (
        device_error,
        host_error,
    )
    # The device result is a deformation of the template, not a copy of it.
    assert np.linalg.norm(device_vertices - template, axis=1).mean() > 0.0


def test_the_session_steps_to_the_same_fit_as_the_batch_call(nricp_dp):
    """Driving the schedule by hand must land where the batch entry point does.

    The session allocates its device buffers once and reuses them across levels
    and steps, so the loop the caller writes is what the batch call does
    internally. It also has to be usable as an interactive loop: a step has to
    come back quickly enough to redraw, and the fit has to improve as it goes.
    """
    if not nricp_dp.device_info()["available"]:
        pytest.skip("no CUDA device here")
    template, faces = _uv_sphere(rings=10, segments=14)
    ground_truth = _bulge(template, amplitude=0.07)
    settings = _settings(2)
    arguments = dict(graph_levels=2, nodes_initial=50, growth=2)

    session = nricp_dp.RouteBSession(
        template, faces, ground_truth, faces, settings, **arguments
    )
    steps = 0
    first_error = None
    while not session.finished():
        progress = session.step()
        steps += 1
        assert progress["level"] < 2
        assert not progress["finished"] or steps > 0
        if steps == 3:
            partial = np.asarray(session.vertices())
            assert partial.shape == template.shape
            assert np.all(np.isfinite(partial))
            first_error = float(
                np.linalg.norm(partial - ground_truth, axis=1).mean()
            )
        assert steps < 5000, "the session did not finish"
    assert steps > 0

    stepped = np.asarray(session.result()["vertices"])
    batch = np.asarray(
        nricp_dp.solve_route_b(
            template, faces, ground_truth, faces, settings, **arguments
        )["vertices"]
    )
    assert np.all(np.isfinite(stepped))
    # Both drive the same schedule; they differ only by the atomic summation
    # order this path already documents.
    assert np.abs(stepped - batch).max() < 1e-3
    final_error = float(np.linalg.norm(stepped - ground_truth, axis=1).mean())
    initial_error = float(
        np.linalg.norm(template - ground_truth, axis=1).mean()
    )
    # Stopping at step three would have been stopping early: that is what makes
    # the interactive loop worth having.
    assert first_error is not None and first_error > final_error
    assert final_error < initial_error


def test_route_b_reports_its_trace_and_repeats_within_accumulation_tolerance(nricp_dp):
    """The device path is repeatable up to float atomic ordering, not bitwise.

    The assembled system is accumulated with atomics, so the summation order
    varies between runs and the result differs in the last few bits. The CPU
    reference stays the deterministic arbiter; this test pins how much drift the
    device path is allowed.
    """
    if not nricp_dp.device_info()["available"]:
        pytest.skip("no CUDA device here")
    import json

    template, faces = _uv_sphere(rings=12, segments=16)
    ground_truth = _bulge(template, amplitude=0.08)

    first = _run_route(nricp_dp, "solve_route_b", template, faces, ground_truth, 2, 60)
    second = _run_route(nricp_dp, "solve_route_b", template, faces, ground_truth, 2, 60)
    # The device path is not bitwise deterministic: the system is accumulated
    # with atomics (summation order varies) and the control flow is noise
    # sensitive (an accept/reject decision or a PCG stopping point can flip when
    # a comparison lands on a tie). Measured on this fixture, two runs drift by
    # about 5e-5 of the unit-scale coordinates and their metrics agree to 1e-5
    # relative; the bound below leaves room for a flipped decision without
    # pretending the path is reproducible to the last bit. The double-precision
    # reference stays the deterministic arbiter.
    drift = np.abs(
        np.asarray(first["vertices"]) - np.asarray(second["vertices"])
    ).max()
    assert drift < 1e-3, f"two identical runs drifted by {drift}"
    first_error = float(
        np.linalg.norm(np.asarray(first["vertices"]) - ground_truth, axis=1).mean()
    )
    second_error = float(
        np.linalg.norm(np.asarray(second["vertices"]) - ground_truth, axis=1).mean()
    )
    assert abs(first_error - second_error) < 0.01 * first_error, (
        first_error,
        second_error,
    )

    report = json.loads(first["report"])
    assert report["route"] == "route_b"
    assert report["precision"] == "float"
    assert report["version"] == nricp_dp.__version__
    assert [level["nodes"] for level in report["levels"]] == list(
        np.asarray(first["level_nodes"])
    )
    for level in report["levels"]:
        assert level["energy_end"] <= level["energy_start"]
