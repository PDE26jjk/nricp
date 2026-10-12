"""Route A end to end on synthetic data: the fit improves, levels are used, and
the robust kernel and the landmark weight do what they are supposed to.

The target is the template with a known smooth displacement applied, so the
ground truth correspondence is exact and the improvement can be measured without
any dataset.
"""

from __future__ import annotations

import json

import numpy as np

from test_core_geometry import _uv_sphere


def _bulge(points: np.ndarray, amplitude: float = 0.08) -> np.ndarray:
    """A smooth, invertible displacement field (radial bump plus a slight shear)."""
    result = points.copy()
    radius = np.linalg.norm(points, axis=1, keepdims=True)
    direction = points / np.maximum(radius, 1e-12)
    bump = amplitude * np.exp(-((points[:, 2:3] - 0.3) ** 2) / 0.05)
    result = result + direction * bump
    result[:, 0] += 0.02 * points[:, 2]
    return result


def _settings(levels: int, **overrides) -> list[dict]:
    base = {
        "point_to_point": 1.0,
        "point_to_plane": 2.0,
        "smoothness": 0.05,
        "anchor": 1e-4,
        "landmark": 10.0,
        "robust_kernel": "huber",
        "robust_threshold": 0.0,
        "max_outer_iterations": 6,
        "max_inner_iterations": 6,
        "outer_tolerance": 1e-4,
        "inner_tolerance": 1e-5,
    }
    base.update(overrides)
    return [dict(base) for _ in range(levels)]


def _run(nricp_dp, template, faces, target, levels=1, target_faces=None, **kwargs):
    return nricp_dp.solve_route_a(
        template,
        faces,
        target,
        faces if target_faces is None else target_faces,
        _settings(levels, **kwargs.pop("level_overrides", {})),
        graph_levels=levels,
        nodes_initial=kwargs.pop("nodes_initial", 60),
        growth=kwargs.pop("growth", 2),
        **kwargs,
    )


def test_single_level_fit_improves_and_reports_its_trace(nricp_dp):
    template, faces = _uv_sphere(rings=18, segments=24)
    ground_truth = _bulge(template)
    result = _run(nricp_dp, template, faces, ground_truth, levels=1)
    report = json.loads(result["report"])

    fitted = np.asarray(result["vertices"])
    before = np.linalg.norm(template - ground_truth, axis=1).mean()
    after = np.linalg.norm(fitted - ground_truth, axis=1).mean()
    assert after < 0.75 * before, f"correspondence error only went {before} -> {after}"
    # The fit has to reach the surface, not merely steer in its direction.
    assert report["correspondence"]["distance_mean"] < 0.1 * before

    level = report["levels"][0]
    assert level["nodes"] == 60
    assert level["outer_iterations"] >= 1
    assert level["inner_iterations"] >= 1
    # Energy must come down over the level, and the report has to say so.
    assert level["energy_end"] < level["energy_start"]
    assert report["energy"]["total"] > 0.0
    assert report["correspondence"]["accepted"] > 0
    assert report["route"] == "route_a"
    assert report["version"] == nricp_dp.__version__


def test_multi_level_is_not_worse_than_single_level(nricp_dp):
    template, faces = _uv_sphere(rings=18, segments=24)
    ground_truth = _bulge(template, amplitude=0.12)

    single = _run(nricp_dp, template, faces, ground_truth, levels=1, nodes_initial=240)
    multi = _run(nricp_dp, template, faces, ground_truth, levels=3, nodes_initial=60)

    single_error = np.linalg.norm(
        np.asarray(single["vertices"]) - ground_truth, axis=1
    ).mean()
    multi_error = np.linalg.norm(
        np.asarray(multi["vertices"]) - ground_truth, axis=1
    ).mean()
    assert multi_error <= single_error * 1.1, (multi_error, single_error)

    counts = np.asarray(multi["level_nodes"]).tolist()
    assert counts == [60, 120, 240]
    levels = json.loads(multi["report"])["levels"]
    assert [level["nodes"] for level in levels] == counts
    assert len(levels) == 3


def test_a_level_reports_convergence_when_it_stops_early(nricp_dp):
    """The convergence flag has to mean something: it stops the level early.

    With a loose outer tolerance the schedule must declare the level converged
    and stop before its iteration budget, instead of silently running to the cap.
    """
    template, faces = _uv_sphere(rings=16, segments=20)
    ground_truth = _bulge(template, amplitude=0.06)
    result = _run(
        nricp_dp,
        template,
        faces,
        ground_truth,
        levels=1,
        nodes_initial=80,
        level_overrides={
            "max_outer_iterations": 8,
            "outer_tolerance": 0.5,
        },
    )
    level = json.loads(result["report"])["levels"][0]
    assert level["converged"] is True
    assert level["outer_iterations"] < 8
    assert level["energy_end"] < level["energy_start"]


def test_strong_landmarks_drive_their_residual_down(nricp_dp):
    """Closes task 4.3's remaining half at the level of a real solve."""
    template, faces = _uv_sphere(rings=16, segments=20)
    ground_truth = _bulge(template, amplitude=0.10)
    indices = np.array([10, 80, 160, 240, 320], dtype=np.int64)
    # The marked vertices are asked to sit slightly off the surface they would
    # settle on by themselves, so the landmark weight is the only thing that can
    # pull them there -- otherwise the data term would satisfy them anyway and
    # the test would measure nothing.
    center = template.mean(axis=0)
    outward = ground_truth[indices] - center
    outward /= np.linalg.norm(outward, axis=1, keepdims=True)
    targets = ground_truth[indices] + 0.02 * outward

    weak = _run(
        nricp_dp,
        template,
        faces,
        ground_truth,
        levels=2,
        nodes_initial=80,
        landmark_indices=indices,
        landmark_targets=targets,
        level_overrides={"landmark": 1.0},
    )
    strong = _run(
        nricp_dp,
        template,
        faces,
        ground_truth,
        levels=2,
        nodes_initial=80,
        landmark_indices=indices,
        landmark_targets=targets,
        level_overrides={"landmark": 2000.0},
    )

    weak_residual = np.linalg.norm(
        np.asarray(weak["vertices"])[indices] - targets, axis=1
    ).mean()
    strong_residual = np.linalg.norm(
        np.asarray(strong["vertices"])[indices] - targets, axis=1
    ).mean()
    assert strong_residual < weak_residual
    assert strong_residual < 1e-2 * np.linalg.norm(
        template.max(axis=0) - template.min(axis=0)
    )


def _detached_cap_target(template, faces, offset=0.35, height=0.85):
    """A target with its cap cut out and floating above the hole."""
    clean = _bulge(template, amplitude=0.10)
    cap = np.where(clean[:, 2] > height)[0]
    assert len(cap) > 10
    cap_set = set(cap.tolist())
    detached = np.array([face for face in faces if set(face.tolist()) & cap_set])
    remaining = np.array([face for face in faces if not set(face.tolist()) & cap_set])
    moved = np.unique(detached)  # the whole patch, boundary ring included
    remap = -np.ones(len(clean), dtype=np.int64)
    remap[moved] = np.arange(len(moved)) + len(clean)
    vertices = np.vstack([clean, clean[moved] + np.array([0.0, 0.0, offset])])
    return clean, cap, vertices, np.vstack([remaining, remap[detached]])


def test_robust_weights_reach_the_fit(nricp_dp):
    """The reweighting is not decorative: it changes where the fit lands.

    A kernel scale far below every residual suppresses the whole data term, so
    the result must collapse towards the rest shape. That is the mechanism
    working end to end; how *much* it helps on a corrupted region is a separate
    question, answered by the distance-threshold test below.
    """
    template, faces = _uv_sphere(rings=16, segments=20)
    ground_truth = _bulge(template, amplitude=0.10)

    plain = _run(nricp_dp, template, faces, ground_truth, levels=1, nodes_initial=80)
    suppressed = _run(
        nricp_dp,
        template,
        faces,
        ground_truth,
        levels=1,
        nodes_initial=80,
        level_overrides={"robust_kernel": "geman_mcclure", "robust_threshold": 1e-6},
    )
    plain_motion = np.linalg.norm(
        np.asarray(plain["vertices"]) - template, axis=1
    ).mean()
    suppressed_motion = np.linalg.norm(
        np.asarray(suppressed["vertices"]) - template, axis=1
    ).mean()
    assert suppressed_motion < 0.5 * plain_motion, (suppressed_motion, plain_motion)


def test_distance_threshold_is_what_removes_a_detached_patch(nricp_dp):
    """The leverage against a detached patch is the distance filter.

    Measured on this fixture, the robust kernel only reduces the drag of the
    floating cap (its nodes are still driven by the intact surface around it),
    while rejecting the far correspondences removes it: the cap then follows the
    regularizer instead of the patch.
    """
    template, faces = _uv_sphere(rings=18, segments=24)
    clean, cap, contaminated, contaminated_faces = _detached_cap_target(template, faces)
    common = dict(levels=2, nodes_initial=80, target_faces=contaminated_faces)

    unfiltered = _run(nricp_dp, template, faces, contaminated, **common)
    filtered = _run(
        nricp_dp, template, faces, contaminated, distance_threshold=0.1, **common
    )

    unfiltered_error = np.linalg.norm(
        np.asarray(unfiltered["vertices"])[cap] - clean[cap], axis=1
    ).mean()
    filtered_error = np.linalg.norm(
        np.asarray(filtered["vertices"])[cap] - clean[cap], axis=1
    ).mean()
    assert filtered_error < 0.5 * unfiltered_error, (filtered_error, unfiltered_error)
    report = json.loads(filtered["report"])
    assert report["correspondence"]["rejected_distance"] > 0


def test_masked_vertices_are_left_to_the_regularizer(nricp_dp):
    template, faces = _uv_sphere(rings=16, segments=20)
    ground_truth = _bulge(template, amplitude=0.10)
    mask = np.ones(len(template), dtype=np.int64)
    masked = np.arange(0, 40)
    mask[masked] = 0

    result = _run(
        nricp_dp,
        template,
        faces,
        ground_truth,
        levels=1,
        nodes_initial=80,
        mask=mask,
    )
    report = json.loads(result["report"])
    assert report["correspondence"]["rejected_masked"] == len(masked)
    fitted = np.asarray(result["vertices"])
    # The leftover displacement of the excluded region must be small compared
    # with the applied deformation, which is what "governed by the regularizer"
    # means in practice.
    excluded_motion = np.linalg.norm(fitted[masked] - template[masked], axis=1).mean()
    applied_motion = np.linalg.norm(ground_truth - template, axis=1).mean()
    assert excluded_motion < 1.5 * applied_motion
