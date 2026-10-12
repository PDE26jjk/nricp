"""Energy blocks: analytic Jacobians against finite differences, block energies,
the mask's effect on the data term, and the robust kernels.

The derivative check is central differences on the *actual* residual function:
the state is retracted by +/-eps along a random tangent direction and the
measured change is compared with the analytic Jacobian times that direction.
"""

from __future__ import annotations

import numpy as np
import pytest

BLOCK_POINT_TO_POINT = 0
BLOCK_POINT_TO_PLANE = 1
BLOCK_SMOOTHNESS = 2
BLOCK_ANCHOR = 3
BLOCK_LANDMARK = 4


def _sphere_patch(scale: float = 1.0, rows: int = 6, columns: int = 7):
    """A gently curved patch: enough geometry for every block to be non-trivial."""
    u = np.linspace(-0.5, 0.5, columns)
    v = np.linspace(-0.5, 0.5, rows)
    grid_u, grid_v = np.meshgrid(u, v, indexing="ij")
    points = np.stack(
        [
            grid_u.ravel() * scale,
            grid_v.ravel() * scale,
            0.25 * scale * (grid_u**2 - grid_v**2).ravel(),
        ],
        axis=1,
    )
    faces = []
    for i in range(columns - 1):
        for j in range(rows - 1):
            a = i * rows + j
            b = (i + 1) * rows + j
            c = (i + 1) * rows + j + 1
            d = i * rows + j + 1
            faces.append([a, b, c])
            faces.append([a, c, d])
    return points, np.asarray(faces, dtype=np.int64)


def _state(nricp_dp, points, node_count, seed):
    """A random but reproducible graph, deformation, and correspondence set."""
    rng = np.random.default_rng(seed)
    node_indices = nricp_dp.farthest_point_seeds(points, node_count)
    nodes = points[np.asarray(node_indices)]
    binding = nricp_dp.bind_vertices(points, nodes)

    rotations = np.broadcast_to(np.eye(3), (node_count, 3, 3)).copy()
    translations = rng.normal(scale=0.02, size=(node_count, 3))
    # Give each node a small rotation so the rotation Jacobians are exercised.
    for node in range(node_count):
        omega = rng.normal(scale=0.05, size=3)
        theta = np.linalg.norm(omega)
        axis = omega / theta if theta > 0 else omega
        skew = np.array(
            [
                [0.0, -axis[2], axis[1]],
                [axis[2], 0.0, -axis[0]],
                [-axis[1], axis[0], 0.0],
            ]
        )
        rotations[node] = (
            np.eye(3)
            + np.sin(theta) * skew
            + (1.0 - np.cos(theta)) * (skew @ skew)
        )

    deformed = nricp_dp.evaluate_deformation(
        points,
        binding["nodes"],
        binding["weights"],
        binding["counts"],
        rotations,
        translations,
        nodes,
    )
    target = deformed + rng.normal(scale=0.01, size=deformed.shape)
    normals = nricp_dp.vertex_normals(target, _sphere_patch()[1])
    return {
        "nodes": nodes,
        "binding": binding,
        "rotations": rotations,
        "translations": translations,
        "deformed": deformed,
        "correspondence_points": target,
        "correspondence_normals": normals,
        "edges": np.asarray(
            [
                [i, (i + 1) % node_count]
                for i in range(node_count)
            ],
            dtype=np.int64,
        ),
    }


def _blocks(nricp_dp, points, state, weights, mask=None):
    count = len(state["nodes"])
    return nricp_dp.energy_blocks(
        points,
        state["binding"]["nodes"],
        state["binding"]["weights"],
        state["binding"]["counts"],
        state["rotations"],
        state["translations"],
        state["nodes"],
        state["correspondence_points"],
        state["correspondence_normals"],
        np.ones(len(points), dtype=np.int64),
        np.ones(len(points)),
        state["edges"],
        np.array([0, len(points) // 2], dtype=np.int64),
        state["correspondence_points"][[0, len(points) // 2]] + 0.01,
        weights,
        mask,
    )


def _all_weights() -> dict:
    return {
        "point_to_point": 1.0,
        "point_to_plane": 2.0,
        "smoothness": 0.5,
        "anchor": 0.25,
        "landmark": 3.0,
    }


def test_block_kinds_are_all_present(nricp_dp):
    points, _ = _sphere_patch()
    state = _state(nricp_dp, points, 13, seed=0)
    result = _blocks(nricp_dp, points, state, _all_weights())
    kinds = set(np.asarray(result["blocks"]["kind"]).tolist())
    assert kinds == {
        BLOCK_POINT_TO_POINT,
        BLOCK_POINT_TO_PLANE,
        BLOCK_SMOOTHNESS,
        BLOCK_ANCHOR,
        BLOCK_LANDMARK,
    }


def test_block_energies_match_their_rows(nricp_dp):
    points, _ = _sphere_patch()
    state = _state(nricp_dp, points, 13, seed=1)
    result = _blocks(nricp_dp, points, state, _all_weights())
    blocks = result["blocks"]
    values = np.asarray(blocks["values"])
    rows = np.asarray(blocks["rows"])
    kinds = np.asarray(blocks["kind"])

    squared = np.array(
        [np.sum(values[i, : rows[i]] ** 2) for i in range(len(rows))]
    )
    expected = {
        "point_to_point": squared[kinds == BLOCK_POINT_TO_POINT].sum(),
        "point_to_plane": squared[kinds == BLOCK_POINT_TO_PLANE].sum(),
        "smoothness": squared[kinds == BLOCK_SMOOTHNESS].sum(),
        "anchor": squared[kinds == BLOCK_ANCHOR].sum(),
        "landmark": squared[kinds == BLOCK_LANDMARK].sum(),
    }
    energies = result["energies"]
    for key, value in expected.items():
        assert energies[key] == pytest.approx(value, rel=1e-12)
    assert energies["total"] == pytest.approx(sum(expected.values()), rel=1e-12)


def test_masked_vertices_contribute_no_data_residual(nricp_dp):
    """The remaining half of task 3.6: a masked vertex is out of the data term."""
    points, _ = _sphere_patch()
    state = _state(nricp_dp, points, 13, seed=2)
    unmasked = _blocks(nricp_dp, points, state, _all_weights())

    mask = np.ones(len(points), dtype=np.int64)
    excluded = [3, 4, 5]
    mask[excluded] = 0
    masked = _blocks(nricp_dp, points, state, _all_weights(), mask=mask)

    data_kinds = (BLOCK_POINT_TO_POINT, BLOCK_POINT_TO_PLANE)
    before = np.isin(np.asarray(unmasked["blocks"]["kind"]), data_kinds).sum()
    after = np.isin(np.asarray(masked["blocks"]["kind"]), data_kinds).sum()
    # Two data blocks per vertex, and exactly the excluded ones disappeared.
    assert before - after == 2 * len(excluded)
    assert masked["energies"]["point_to_point"] < unmasked["energies"]["point_to_point"]
    assert masked["energies"]["point_to_plane"] < unmasked["energies"]["point_to_plane"]
    # The regularizers are untouched by the mask.
    for key in ("smoothness", "anchor", "landmark"):
        assert masked["energies"][key] == pytest.approx(unmasked["energies"][key])


def test_analytic_jacobians_match_central_differences(nricp_dp):
    points, _ = _sphere_patch()
    node_count = 13
    state = _state(nricp_dp, points, node_count, seed=3)
    weights = _all_weights()

    rng = np.random.default_rng(4)
    direction = rng.normal(size=node_count * 6)
    direction /= np.linalg.norm(direction)
    step = 1e-6

    def blocks_at(delta):
        moved = nricp_dp.retract_deformation(
            state["rotations"], state["translations"], state["nodes"], delta
        )
        moved_state = dict(state)
        moved_state["rotations"] = moved["rotations"]
        moved_state["translations"] = moved["translations"]
        return _blocks(nricp_dp, points, moved_state, weights)

    base = _blocks(nricp_dp, points, state, weights)["blocks"]
    forward = blocks_at(step * direction)["blocks"]
    backward = blocks_at(-step * direction)["blocks"]

    values = np.asarray(base["values"])
    rows = np.asarray(base["rows"])
    slots = np.asarray(base["slots"])
    nodes = np.asarray(base["nodes"])
    jacobians = np.asarray(base["jacobians"])

    worst_relative_error = 0.0
    worst_block = -1
    for block in range(len(rows)):
        row_count = rows[block]
        predicted = np.zeros(row_count)
        for slot in range(slots[block]):
            node = nodes[block, slot]
            predicted += (
                jacobians[block, slot, :row_count, :] @ direction[node * 6 : node * 6 + 6]
            )
        measured = (
            np.asarray(forward["values"])[block, :row_count]
            - np.asarray(backward["values"])[block, :row_count]
        ) / (2 * step)
        scale = max(np.abs(measured).max(), np.abs(predicted).max(), 1e-12)
        error = np.abs(measured - predicted).max() / scale
        if error > worst_relative_error:
            worst_relative_error = error
            worst_block = block
        # Blocks that are entirely insensitive to this direction carry no
        # information; skip them rather than dividing by zero.
        if np.abs(measured).max() > 0.0:
            assert np.allclose(measured, predicted, atol=1e-6, rtol=1e-4), (
                f"block {block} (kind {np.asarray(base['kind'])[block]}) "
                f"disagrees: measured {measured}, predicted {predicted}"
            )
    assert worst_relative_error < 1e-4, f"worst block {worst_block}"


def test_robust_weights_are_monotone_and_bounded(nricp_dp):
    squared = np.linspace(0.0, 10.0, 50)
    threshold = 0.5
    for kernel in ("huber", "geman_mcclure"):
        weights = np.asarray(
            nricp_dp.robust_weights(kernel, squared, threshold)
        )
        assert weights[0] == pytest.approx(1.0)
        assert np.all(np.diff(weights) <= 0.0)
        assert np.all(weights > 0.0) and np.all(weights <= 1.0)

    # Huber is exactly 1 inside the threshold and falls off as t/sqrt(s) after.
    huber = np.asarray(nricp_dp.robust_weights("huber", squared, threshold))
    inside = squared <= threshold * threshold
    assert np.allclose(huber[inside], 1.0)
    outside = squared > threshold * threshold
    assert np.allclose(huber[outside], threshold / np.sqrt(squared[outside]))

    # Geman-McClure decays faster and never reaches zero.
    geman = np.asarray(nricp_dp.robust_weights("geman_mcclure", squared, threshold))
    assert np.all(geman[outside] < huber[outside])

    # The neutral kernel leaves everything alone.
    assert np.allclose(nricp_dp.robust_weights("none", squared, threshold), 1.0)
