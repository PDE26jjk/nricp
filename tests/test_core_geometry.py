"""Template-side preparation: cleaning, normalization, sampling, binding,
deformation evaluation, level transfer, and closest-point queries.

Everything here is checked against an independent expectation: an analytic
surface, a numpy re-implementation of the formula, or the identity the stage is
supposed to preserve.
"""

from __future__ import annotations

import numpy as np
import pytest


def _fps_reference(points: np.ndarray, count: int) -> np.ndarray:
    """Plain numpy farthest-point sampling, used as the independent reference."""
    seeds = np.empty(count, dtype=np.int64)
    seeds[0] = 0
    nearest = np.linalg.norm(points - points[0], axis=1)
    for i in range(1, count):
        seeds[i] = int(np.argmax(nearest))
        nearest = np.minimum(nearest, np.linalg.norm(points - points[seeds[i]], axis=1))
    return seeds


def _uv_sphere(rings: int = 32, segments: int = 64, radius: float = 1.0):
    vertices = []
    for ring in range(rings + 1):
        theta = np.pi * ring / rings
        for segment in range(segments):
            phi = 2 * np.pi * segment / segments
            vertices.append(
                [
                    radius * np.sin(theta) * np.cos(phi),
                    radius * np.sin(theta) * np.sin(phi),
                    radius * np.cos(theta),
                ]
            )
    faces = []

    def index(ring: int, segment: int) -> int:
        return ring * segments + (segment % segments)

    for ring in range(rings):
        for segment in range(segments):
            a, b = index(ring, segment), index(ring, segment + 1)
            c, d = index(ring + 1, segment + 1), index(ring + 1, segment)
            # Winding chosen so the face normals point outwards (area-weighted
            # vertex normals inherit their sign from this).
            faces.append([a, c, b])
            faces.append([a, d, c])
    return np.asarray(vertices), np.asarray(faces, dtype=np.int64)


# --- normalization (task 3.1) ------------------------------------------------


def test_normalization_puts_the_diagonal_at_one(nricp_dp):
    rng = np.random.default_rng(0)
    points = rng.normal(size=(500, 3)) * np.array([3.0, 1.0, 2.0]) + 5.0
    normalization = nricp_dp.normalization(points)
    center = np.asarray(normalization["center"])[0]
    normalized = (points - center) * normalization["scale"]
    diagonal = np.linalg.norm(normalized.max(axis=0) - normalized.min(axis=0))
    assert diagonal == pytest.approx(1.0, abs=1e-12)
    assert np.allclose(center, (points.min(axis=0) + points.max(axis=0)) / 2.0)

    # The inverse restores the original coordinates.
    restored = normalized / normalization["scale"] + center
    assert np.allclose(restored, points, atol=1e-12)


def test_validation_reports_unusable_input(nricp_dp):
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    good = nricp_dp.validate_mesh(points, np.array([[0, 1, 2]]))
    assert good["ok"] and not good["problems"]

    bad_index = nricp_dp.validate_mesh(points, np.array([[0, 1, 7]]))
    assert not bad_index["ok"]
    assert any("out of range" in problem for problem in bad_index["problems"])

    non_finite = points.copy()
    non_finite[1, 2] = np.nan
    reported = nricp_dp.validate_mesh(non_finite, np.array([[0, 1, 2]]))
    assert not reported["ok"]
    assert any("not finite" in problem for problem in reported["problems"])


# --- cleaning (task 3.2) -----------------------------------------------------


def test_cleaning_reports_exactly_what_it_dropped(nricp_dp):
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, 0.0, 0.0],  # collinear with 0 and 1 -> zero-area face
            [0.0, 3.0, 0.0],  # unused
        ]
    )
    faces = np.array(
        [
            [0, 1, 2],
            [1, 3, 2],
            [0, 2, 1],  # duplicate of the first face, opposite winding
            [0, 1, 0],  # repeated index
            [0, 1, 4],  # zero area
        ],
        dtype=np.int64,
    )
    cleaned = nricp_dp.clean_mesh(points, faces)
    assert cleaned["duplicate_faces"] == 1
    assert cleaned["degenerate_faces"] == 2
    assert cleaned["isolated_vertices"] == 2  # vertex 4 lost its only face
    assert cleaned["faces"].shape == (2, 3)
    assert cleaned["points"].shape == (4, 3)
    assert cleaned["faces"].max() < len(cleaned["points"])
    # The surviving winding is the first occurrence's.
    assert list(cleaned["faces"][0]) == [0, 1, 2]


def test_cleaning_keeps_faces_that_only_share_an_edge(nricp_dp):
    """Faces are duplicates only when their whole vertex set matches.

    Keying the duplicate test on the two smallest corners would treat any two
    faces sharing an edge as duplicates and delete one of them, which on a real
    scan tears holes into the target surface.
    """
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int64)
    cleaned = nricp_dp.clean_mesh(points, faces)
    assert cleaned["duplicate_faces"] == 0
    assert cleaned["degenerate_faces"] == 0
    assert cleaned["faces"].shape == (4, 3)

    # A genuine duplicate is still caught, whatever its winding.
    duplicated = np.vstack([faces, faces[:, ::-1]])
    again = nricp_dp.clean_mesh(points, duplicated)
    assert again["duplicate_faces"] == 4
    assert again["faces"].shape == (4, 3)


# --- sampling (task 3.3) -----------------------------------------------------


def test_farthest_point_seeds_match_the_reference(nricp_dp):
    rng = np.random.default_rng(1)
    points = rng.normal(size=(1000, 3))
    seeds = np.asarray(nricp_dp.farthest_point_seeds(points, 25))
    assert seeds.tolist() == _fps_reference(points, 25).tolist()
    assert len(np.unique(seeds)) == 25
    assert seeds[0] == 0

    # Reproducible, and clamped when fewer seeds are requested than points.
    assert np.array_equal(seeds, np.asarray(nricp_dp.farthest_point_seeds(points, 25)))
    assert len(nricp_dp.farthest_point_seeds(points, 5000)) == 1000


def test_farthest_point_seeds_respect_the_start(nricp_dp):
    rng = np.random.default_rng(2)
    points = rng.normal(size=(200, 3))
    seeds = np.asarray(nricp_dp.farthest_point_seeds(points, 10, 7))
    assert seeds[0] == 7


# --- binding (task 3.4) ------------------------------------------------------


def test_binding_weights_follow_the_distance_ratio(nricp_dp):
    """w = (1 - d/dmax)^2 normalized over the vertex's own support."""
    vertex = np.zeros((1, 3))
    nodes = np.array([[1.0, 0, 0], [2.0, 0, 0], [3.0, 0, 0], [4.0, 0, 0]])
    binding = nricp_dp.bind_vertices(vertex, nodes)
    weights = np.asarray(binding["weights"])[0]
    falloff = (1.0 - np.array([1.0, 2.0, 3.0, 4.0]) / 4.0) ** 2
    assert np.allclose(weights, falloff / falloff.sum())
    assert weights[0] == weights.max()


def test_binding_is_normalized_and_frozen(nricp_dp):
    rng = np.random.default_rng(3)
    points = rng.normal(size=(300, 3))
    nodes = rng.normal(size=(12, 3))
    binding = nricp_dp.bind_vertices(points, nodes)
    weights = np.asarray(binding["weights"])
    counts = np.asarray(binding["counts"])
    assert (counts == 4).all()
    assert np.allclose(weights.sum(axis=1), 1.0)

    # The largest weight belongs to the nearest node.
    distances = np.linalg.norm(points[:, None, :] - nodes[None, :, :], axis=2)
    nearest = distances.argmin(axis=1)
    chosen = np.asarray(binding["nodes"])[np.arange(len(points)), weights.argmax(axis=1)]
    assert np.array_equal(chosen, nearest)

    # Binding the same vertices twice gives the same result: nothing depends on
    # a deformed state.
    again = nricp_dp.bind_vertices(points, nodes)
    assert np.array_equal(np.asarray(binding["nodes"]), np.asarray(again["nodes"]))
    assert np.allclose(weights, np.asarray(again["weights"]))


def test_binding_with_fewer_nodes_than_the_support(nricp_dp):
    points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    nodes = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    binding = nricp_dp.bind_vertices(points, nodes)
    assert (np.asarray(binding["counts"]) == 3).all()
    assert np.allclose(np.asarray(binding["weights"]).sum(axis=1), 1.0)


# --- template graph (task 3.5) ----------------------------------------------


def test_template_graph_levels_and_normalized_nodes(nricp_dp):
    points, faces = _uv_sphere(rings=20, segments=30)
    graph = nricp_dp.build_template_graph(
        points, faces, levels=4, nodes_initial=50, growth=2
    )
    levels = graph["levels"]
    assert [len(level["node_template_indices"]) for level in levels] == [50, 100, 200, 400]

    center = np.asarray(graph["normalization"]["center"])[0]
    scale = graph["normalization"]["scale"]
    for level in levels:
        indices = np.asarray(level["node_template_indices"])
        assert len(np.unique(indices)) == len(indices)
        assert indices.min() >= 0 and indices.max() < len(points)
        expected = (points[indices] - center) * scale
        assert np.allclose(np.asarray(level["node_positions"]), expected, atol=1e-12)
        # Bindings refer to nodes of the same level.
        assert np.asarray(level["bindings"]["nodes"]).max() < len(indices)


def test_deformation_identity_and_global_rigid(nricp_dp):
    points, _ = _uv_sphere(rings=8, segments=12)
    nodes = points[::13]
    binding = nricp_dp.bind_vertices(points, nodes)
    node_count = len(nodes)

    identity = np.broadcast_to(np.eye(3), (node_count, 3, 3)).copy()
    zeros = np.zeros((node_count, 3))
    unchanged = nricp_dp.evaluate_deformation(
        points,
        binding["nodes"],
        binding["weights"],
        binding["counts"],
        identity,
        zeros,
        nodes,
    )
    assert np.allclose(unchanged, points, atol=1e-12)

    angle = 0.4
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    translation = np.array([0.3, -0.2, 0.7])
    # A node's translation is the displacement of its own rest position, so a
    # global rigid warp p -> R p + t gives t_j = R g_j + t - g_j.
    node_translations = nodes @ rotation.T + translation - nodes
    rotated = nricp_dp.evaluate_deformation(
        points,
        binding["nodes"],
        binding["weights"],
        binding["counts"],
        np.broadcast_to(rotation, (node_count, 3, 3)).copy(),
        node_translations,
        nodes,
    )
    assert np.allclose(rotated, points @ rotation.T + translation, atol=1e-12)


def test_level_transfer_reproduces_a_global_rigid_field(nricp_dp):
    """A rigid field must survive warm-starting a finer level."""
    points, faces = _uv_sphere(rings=10, segments=16)
    graph = nricp_dp.build_template_graph(
        points, faces, levels=2, nodes_initial=40, growth=3
    )
    coarse, fine = graph["levels"]
    coarse_nodes = np.asarray(coarse["node_positions"])

    angle = 0.25
    rotation = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(angle), -np.sin(angle)],
            [0.0, np.sin(angle), np.cos(angle)],
        ]
    )
    translation = np.array([0.1, 0.4, -0.3])
    coarse_translations = coarse_nodes @ rotation.T + translation - coarse_nodes
    transferred = nricp_dp.transfer_level(
        coarse_nodes,
        np.broadcast_to(rotation, (len(coarse_nodes), 3, 3)).copy(),
        coarse_translations,
        np.asarray(fine["node_positions"]),
    )
    assert np.allclose(np.asarray(transferred["rotations"]), rotation, atol=1e-12)
    fine_nodes = np.asarray(fine["node_positions"])
    assert np.allclose(
        np.asarray(transferred["translations"]),
        fine_nodes @ rotation.T + translation - fine_nodes,
        atol=1e-12,
    )

    # And evaluating with the transferred field moves the template by exactly
    # that rigid transform.
    normalized = (
        points - np.asarray(graph["normalization"]["center"])[0]
    ) * graph["normalization"]["scale"]
    evaluated = nricp_dp.evaluate_deformation(
        normalized,
        fine["bindings"]["nodes"],
        fine["bindings"]["weights"],
        fine["bindings"]["counts"],
        transferred["rotations"],
        transferred["translations"],
        fine_nodes,
    )
    assert np.allclose(evaluated, normalized @ rotation.T + translation, atol=1e-10)


def test_level_transfer_survives_opposite_neighbour_rotations(nricp_dp):
    """Neighbours a half turn apart must not blend into the identity rotation.

    Two rotations of about +/-170 degrees have quaternions on opposite sides of
    the hemisphere; a weighted sum of those is near zero, and normalizing it
    would silently produce no rotation at all for that node.
    """
    points, faces = _uv_sphere(rings=10, segments=16)
    graph = nricp_dp.build_template_graph(
        points, faces, levels=2, nodes_initial=40, growth=3
    )
    coarse, fine = graph["levels"]
    coarse_nodes = np.asarray(coarse["node_positions"])

    angle = np.deg2rad(170.0)
    rotation_positive = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rotations = np.array(
        [
            rotation_positive if index % 2 == 0 else rotation_positive.T
            for index in range(len(coarse_nodes))
        ]
    )
    transferred = nricp_dp.transfer_level(
        coarse_nodes,
        rotations,
        np.zeros((len(coarse_nodes), 3)),
        np.asarray(fine["node_positions"]),
    )
    got = np.asarray(transferred["rotations"])
    angles = np.arccos(
        np.clip((np.trace(got, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    )
    assert np.all(np.isfinite(got))
    assert np.all(angles > np.deg2rad(90.0)), (
        "a node collapsed towards the identity rotation: "
        f"min angle {np.rad2deg(angles.min()):.1f} degrees"
    )


def test_rigid_fit_recovers_a_known_transform_and_survives_a_degenerate_support(
    nricp_dp,
):
    rng = np.random.default_rng(5)
    source = rng.normal(size=(20, 3))
    angle = 0.7
    rotation = np.array(
        [
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ]
    )
    translation = np.array([-0.2, 0.6, 0.1])
    target = source @ rotation.T + translation

    fit = nricp_dp.fit_rigid(source, target, np.ones(len(source)))
    assert np.allclose(np.asarray(fit["rotation"])[0], rotation, atol=1e-12)
    assert np.allclose(np.asarray(fit["translation"])[0], translation, atol=1e-12)

    # Points on a line leave the rotation about that line undetermined; the fit
    # must stay finite and still land the points where they belong.
    line = np.stack([np.linspace(-1.0, 1.0, 6), np.zeros(6), np.zeros(6)], axis=1)
    line_target = line @ rotation.T + translation
    degenerate = nricp_dp.fit_rigid(line, line_target, np.ones(len(line)))
    fitted_rotation = np.asarray(degenerate["rotation"])[0]
    fitted_translation = np.asarray(degenerate["translation"])[0]
    assert np.all(np.isfinite(fitted_rotation)) and np.all(
        np.isfinite(fitted_translation)
    )
    residual = line @ fitted_rotation.T + fitted_translation - line_target
    assert np.abs(residual).max() < 1e-9


def test_retraction_round_trips_with_the_tangent_difference(nricp_dp):
    """Zero increment is a no-op, and a small increment is recovered exactly."""
    points, _ = _uv_sphere(rings=8, segments=12)
    nodes = points[::17]
    node_count = len(nodes)
    rotations = np.broadcast_to(np.eye(3), (node_count, 3, 3)).copy()
    translations = np.zeros((node_count, 3))

    zeros = np.zeros(node_count * 6)
    unchanged = nricp_dp.retract_deformation(rotations, translations, nodes, zeros)
    assert np.array_equal(np.asarray(unchanged["rotations"]), rotations)
    assert np.array_equal(np.asarray(unchanged["translations"]), translations)

    rng = np.random.default_rng(7)
    increment = rng.normal(scale=1e-3, size=node_count * 6)
    moved = nricp_dp.retract_deformation(rotations, translations, nodes, increment)
    recovered = np.asarray(
        nricp_dp.tangent_difference(
            rotations,
            translations,
            moved["rotations"],
            moved["translations"],
        )
    )
    assert np.allclose(recovered, increment, atol=1e-12)

    # A node's own rest position moves by exactly its translation part, because
    # the rotation acts about that position (evaluate each node through a
    # single-node support so the neighbours cannot blend in).
    single = np.zeros((node_count, 4), dtype=np.int64)
    single[:, 0] = np.arange(node_count)
    weights = np.zeros((node_count, 4))
    weights[:, 0] = 1.0
    counts = np.ones(node_count, dtype=np.int64)
    evaluated = nricp_dp.evaluate_deformation(
        nodes,
        single,
        weights,
        counts,
        moved["rotations"],
        moved["translations"],
        nodes,
    )
    expected = nodes + increment.reshape(node_count, 6)[:, 3:]
    assert np.allclose(evaluated, expected, atol=1e-12)


# --- closest-point queries (task 3.7) ---------------------------------------


def test_closest_points_on_a_sphere_match_the_analytic_answer(nricp_dp):
    points, faces = _uv_sphere(rings=32, segments=64, radius=1.0)
    rng = np.random.default_rng(4)
    directions = rng.normal(size=(300, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    radii = rng.uniform(0.4, 1.6, size=(300, 1))
    queries = directions * radii

    result = nricp_dp.closest_points(points, faces, queries)
    distances = np.asarray(result["distances"])
    closest = np.asarray(result["points"])
    normals = np.asarray(result["normals"])

    assert np.allclose(distances, np.abs(radii[:, 0] - 1.0), atol=0.02)
    assert np.allclose(np.linalg.norm(closest, axis=1), 1.0, atol=0.02)
    assert np.all(np.einsum("ij,ij->i", normals, directions) > 0.99)


def test_closest_points_on_a_flat_patch_are_exact(nricp_dp):
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    queries = np.array([[0.25, 0.25, 2.0], [0.9, 0.1, 0.5], [2.0, 0.5, 1.0]])
    result = nricp_dp.closest_points(points, faces, queries)

    distances = np.asarray(result["distances"])
    closest = np.asarray(result["points"])
    normals = np.asarray(result["normals"])
    assert distances[0] == pytest.approx(2.0, abs=1e-12)
    assert np.allclose(closest[0], [0.25, 0.25, 0.0], atol=1e-12)
    assert distances[1] == pytest.approx(0.5, abs=1e-12)
    assert np.allclose(normals, [0.0, 0.0, 1.0], atol=1e-12)
    # A query outside the patch lands on its boundary.
    assert np.allclose(closest[2], [1.0, 0.5, 0.0], atol=1e-12)

    # Querying an existing vertex returns it exactly.
    at_vertex = nricp_dp.closest_points(points, faces, points)
    assert np.allclose(np.asarray(at_vertex["distances"]), 0.0, atol=1e-12)


def test_closest_points_agree_with_a_brute_force_search(nricp_dp):
    """Independent check: densely sample the surface and search it with a KD-tree."""
    from scipy.spatial import cKDTree

    points, faces = _uv_sphere(rings=8, segments=12)
    # Barycentric grid over every triangle: the samples are on the true surface,
    # so the KD-tree distance is an upper bound that the exact answer must match
    # to the sampling resolution.
    steps = np.linspace(0.0, 1.0, 13)
    samples = []
    for face in faces:
        a, b, c = points[face[0]], points[face[1]], points[face[2]]
        for u in steps:
            for v in steps:
                if u + v > 1.0:
                    continue
                samples.append(a + (b - a) * u + (c - a) * v)
    samples = np.asarray(samples)
    tree = cKDTree(samples)

    rng = np.random.default_rng(6)
    queries = rng.normal(size=(200, 3))
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    queries *= rng.uniform(0.2, 1.8, size=(200, 1))

    result = nricp_dp.closest_points(points, faces, queries)
    exact = np.asarray(result["distances"])
    approximate, _ = tree.query(queries)
    assert np.all(exact <= approximate + 1e-12)
    # Resolution of the barycentric grid on a unit sphere of radius 1:
    # edge ~ pi/8 over 12 subdivisions, so 1e-2 is comfortably above it and far
    # below any real disagreement.
    assert np.all(approximate - exact < 1e-2)


# --- masks and landmarks (task 3.6) -----------------------------------------


def test_landmark_validation_names_the_problem(nricp_dp):
    indices = np.array([3, 9], dtype=np.int64)
    targets = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    with pytest.raises(ValueError) as error:
        nricp_dp.validate_landmarks(indices, targets, 8)
    assert "landmark 1" in str(error.value)
    assert "template vertex 9" in str(error.value)

    with pytest.raises(ValueError) as duplicated:
        nricp_dp.validate_landmarks(
            np.array([2, 2], dtype=np.int64), targets, 8
        )
    assert "twice" in str(duplicated.value)

    with pytest.raises(ValueError) as mismatched:
        nricp_dp.validate_landmarks(
            np.array([1, 2], dtype=np.int64), targets[:1], 8
        )
    assert "matching indices" in str(mismatched.value)

    # A valid set is accepted silently.
    nricp_dp.validate_landmarks(np.array([1, 2], dtype=np.int64), targets, 8)


def test_mask_validation_and_active_count(nricp_dp):
    mask = np.ones(8, dtype=np.int64)
    mask[3] = 0
    assert nricp_dp.validate_mask(mask, 8) == 7

    with pytest.raises(ValueError) as wrong_length:
        nricp_dp.validate_mask(mask[:5], 8)
    assert "mask has 5 entries" in str(wrong_length.value)

    with pytest.raises(ValueError) as empty:
        nricp_dp.validate_mask(np.zeros(8, dtype=np.int64), 8)
    assert "excludes every vertex" in str(empty.value)
