"""Route C: the global/local alternation, checked the same way route B is.

Route C shares route A's and route B's schedule, energies, correspondence rules
and linear solver; what differs is how a node's six values are produced. The
checks here cover what is new -- the closed-form rotation fit and the
translation-only solve -- and then hold the whole route to route A, the
double-precision reference, on the same synthetic deformation route B is held to.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from test_core_geometry import _uv_sphere
from test_route_a import _bulge, _settings


def _run(nricp_dp, name, template, faces, target, levels, nodes_initial, **extra):
    return getattr(nricp_dp, name)(
        template,
        faces,
        target,
        faces,
        _settings(levels),
        graph_levels=levels,
        nodes_initial=nodes_initial,
        growth=2,
        **extra,
    )


def _rotations(result):
    return np.asarray(result["rotations"], dtype=np.float64).reshape(-1, 3, 3)


def _needs_device(nricp_dp):
    if not nricp_dp.device_info()["available"]:
        pytest.skip("no CUDA device here")


def test_the_fit_returns_the_identity_when_there_is_nothing_to_fit(nricp_dp):
    """The acceptance criterion for the local step, through the whole route.

    With the target equal to the template every residual is zero, so the only
    rotation the fit can be asked for is the identity, and the translations
    cannot move: this is the fixed point the alternation has to sit on.
    """
    _needs_device(nricp_dp)
    template, faces = _uv_sphere(rings=12, segments=16)
    result = _run(nricp_dp, "solve_route_c", template, faces, template, 2, 60)
    rotations = _rotations(result)
    vertices = np.asarray(result["vertices"], dtype=np.float64)
    translations = np.asarray(result["translations"], dtype=np.float64)

    deviation = float(np.abs(rotations - np.eye(3)[None]).max())
    assert deviation < 1e-4, f"the fit left the identity by {deviation}"
    assert float(np.abs(translations).max()) < 1e-4
    assert float(np.abs(vertices - template).max()) < 1e-4


def test_the_fitted_rotations_are_proper(nricp_dp):
    """Every node's rotation has to be orthonormal with a positive determinant.

    The fit goes through a quaternion, which cannot represent a reflection, so
    this is a property of the method rather than of a branch that could be
    missed -- but it is the property the acceptance criteria name, and it is
    cheap to hold the route to it on a deformation that actually moves things.
    """
    _needs_device(nricp_dp)
    template, faces = _uv_sphere(rings=12, segments=16)
    ground_truth = _bulge(template, amplitude=0.08)
    rotations = _rotations(
        _run(nricp_dp, "solve_route_c", template, faces, ground_truth, 2, 60)
    )

    gram = np.einsum("nij,nkj->nik", rotations, rotations)
    assert np.all(np.isfinite(rotations))
    assert float(np.abs(gram - np.eye(3)[None]).max()) < 1e-4
    determinants = np.linalg.det(rotations)
    assert determinants.min() > 0.999
    assert determinants.max() < 1.001


def test_a_translated_target_is_carried_by_the_translations(nricp_dp):
    """The one rigid motion a surface fit can be asked for without ambiguity.

    Both data terms measure a distance to the target's *surface*, so a rotation
    of a shape that is close to a sphere is nearly unidentifiable -- the caller
    is the one who aligns the two meshes, per the API's contract. A translation
    is different: it has to move the mesh along the offset, and it is the global
    step's job, with nothing for the rotations to do. So the offset direction
    has to come back, the shape must not change to get there (the vertex
    covariance's eigenvalues move by a few percent at most), and the fitted
    rotations stay small.
    """
    _needs_device(nricp_dp)
    base, faces = _uv_sphere(rings=14, segments=18)
    template = base * np.array([1.4, 1.0, 0.75])
    offset = np.array([0.0, 0.05, 0.0])
    target = template + offset

    result = _run(nricp_dp, "solve_route_c", template, faces, target, 2, 80)
    fitted = np.asarray(result["vertices"], dtype=np.float64)
    moved = fitted.mean(axis=0) - template.mean(axis=0)

    assert float(np.linalg.norm(moved)) > 0.15 * float(np.linalg.norm(offset))
    cosine = float(moved @ offset / np.linalg.norm(moved) / np.linalg.norm(offset))
    assert cosine > 0.99, (moved, cosine)

    before = np.sort(np.linalg.eigvalsh(np.cov((template - template.mean(0)).T)))
    after = np.sort(np.linalg.eigvalsh(np.cov((fitted - fitted.mean(0)).T)))
    ratios = after / before
    assert np.all(ratios > 0.95) and np.all(ratios < 1.02), ratios

    rotations = _rotations(result)
    angles = np.degrees(
        np.arccos(
            np.clip((np.trace(rotations, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
        )
    )
    assert float(np.median(angles)) < 6.0, np.percentile(angles, [50, 95])


def test_route_c_matches_route_a_on_a_synthetic_case(nricp_dp):
    """The two formulations minimise the same energy, so they must agree.

    They are different algorithms -- a simultaneous Newton step against a
    block-wise alternation -- so the agreement is looser than route B's, and it
    is stated as a fraction of the error the reference reaches rather than as a
    per-vertex equality.
    """
    _needs_device(nricp_dp)
    template, faces = _uv_sphere(rings=14, segments=18)
    ground_truth = _bulge(template, amplitude=0.09)

    native = _run(nricp_dp, "solve_route_c", template, faces, ground_truth, 2, 80)
    reference = _run(nricp_dp, "solve_route_a", template, faces, ground_truth, 2, 80)
    native_error = float(
        np.linalg.norm(np.asarray(native["vertices"]) - ground_truth, axis=1).mean()
    )
    reference_error = float(
        np.linalg.norm(np.asarray(reference["vertices"]) - ground_truth, axis=1).mean()
    )
    initial_error = float(np.linalg.norm(template - ground_truth, axis=1).mean())

    assert native_error < 0.75 * initial_error, (native_error, initial_error)
    assert native_error < 1.10 * reference_error, (native_error, reference_error)
    assert np.all(np.isfinite(np.asarray(native["vertices"])))


def test_the_route_c_session_shares_the_schedule(nricp_dp):
    """Driving route C by hand must land where the batch entry point does.

    Both device routes run the same schedule, so the session only has to be told
    which one it is; everything the caller sees -- the progress, the partial
    vertices, the report -- keeps its meaning.
    """
    _needs_device(nricp_dp)
    template, faces = _uv_sphere(rings=10, segments=14)
    ground_truth = _bulge(template, amplitude=0.07)
    settings = _settings(2)
    arguments = dict(graph_levels=2, nodes_initial=50, growth=2)

    session = nricp_dp.RouteBSession(
        template, faces, ground_truth, faces, settings, route="route_c", **arguments
    )
    steps = 0
    while not session.finished():
        progress = session.step()
        steps += 1
        assert progress["level"] < 2
        if steps == 3:
            partial = np.asarray(session.vertices())
            assert partial.shape == template.shape
            assert np.all(np.isfinite(partial))
        assert steps < 5000, "the session did not finish"

    stepped = np.asarray(session.result()["vertices"])
    batch = np.asarray(
        nricp_dp.solve_route_c(
            template, faces, ground_truth, faces, settings, **arguments
        )["vertices"]
    )
    # Route C is less repeatable than route B: the system is assembled with
    # atomics (so the summation order varies) and the local step *replaces* a
    # rotation rather than stepping it, so a difference in the last bits can
    # move a nearly degenerate fit between branches. Measured on this fixture,
    # two identical runs of the same call drift by up to 1.4e-3 of the
    # unit-scale coordinates, against route B's 5e-5; the bound leaves room for
    # one flipped branch without pretending the route is reproducible.
    assert np.abs(stepped - batch).max() < 1e-2
    report = json.loads(session.report())
    assert report["route"] == "route_c"
    assert report["precision"] == "float"


def test_the_route_c_report_names_its_route_and_descends(nricp_dp):
    """Every level has to end below where it started, and say which route it is."""
    _needs_device(nricp_dp)
    template, faces = _uv_sphere(rings=12, segments=16)
    ground_truth = _bulge(template, amplitude=0.08)
    result = _run(nricp_dp, "solve_route_c", template, faces, ground_truth, 2, 60)
    report = json.loads(result["report"])

    assert report["route"] == "route_c"
    assert report["version"] == nricp_dp.__version__
    assert [level["nodes"] for level in report["levels"]] == list(
        np.asarray(result["level_nodes"])
    )
    for level in report["levels"]:
        assert level["inner_iterations"] > 0
        assert level["energy_end"] <= level["energy_start"]


def test_the_session_refuses_an_unknown_route(nricp_dp):
    """An unknown route name is a caller error, not a silent fallback."""
    _needs_device(nricp_dp)
    template, faces = _uv_sphere(rings=6, segments=8)
    with pytest.raises(ValueError):
        nricp_dp.RouteBSession(
            template,
            faces,
            template,
            faces,
            _settings(1),
            graph_levels=1,
            nodes_initial=20,
            route="route_q",
        )
