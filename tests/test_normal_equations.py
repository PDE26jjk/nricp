"""Route A's linear algebra: the assembled system and its direct solve.

The assembly is compared against an independent numpy implementation built from
the same block Jacobians, the solve is compared against a dense numpy solve, and
the step is checked to actually decrease the energy -- which is what pins the
sign convention of the right-hand side.
"""

from __future__ import annotations

import numpy as np
import pytest

from test_energy_blocks import _all_weights, _blocks, _sphere_patch, _state


def _system(nricp_dp, points, state, weights, mask=None):
    result = _blocks(nricp_dp, points, state, weights, mask)
    blocks = result["blocks"]
    system = nricp_dp.assemble_normal_equations(
        blocks["kind"],
        blocks["rows"],
        blocks["values"],
        blocks["slots"],
        blocks["nodes"],
        blocks["jacobians"],
        len(state["nodes"]),
    )
    return result, blocks, system


def _numpy_assembly(blocks, node_count):
    """Independent assembly from the returned block arrays."""
    rows = np.asarray(blocks["rows"])
    values = np.asarray(blocks["values"])
    slots = np.asarray(blocks["slots"])
    nodes = np.asarray(blocks["nodes"])
    jacobians = np.asarray(blocks["jacobians"])

    diagonal = np.zeros((node_count, 6, 6))
    rhs = np.zeros((node_count, 6))
    interactions: dict[tuple[int, int], np.ndarray] = {}
    for block in range(len(rows)):
        for slot in range(slots[block]):
            node = nodes[block, slot]
            if node < 0:
                continue
            jacobian = jacobians[block, slot, : rows[block], :]
            residual = values[block, : rows[block]]
            diagonal[node] += jacobian.T @ jacobian
            rhs[node] -= jacobian.T @ residual
            for other in range(slot + 1, slots[block]):
                other_node = nodes[block, other]
                if other_node < 0:
                    continue
                other_jacobian = jacobians[block, other, : rows[block], :]
                key = (min(node, other_node), max(node, other_node))
                if node < other_node:
                    contribution = jacobian.T @ other_jacobian
                else:
                    contribution = other_jacobian.T @ jacobian
                interactions[key] = interactions.get(key, np.zeros((6, 6))) + contribution
    return diagonal, rhs, interactions


def test_assembly_matches_an_independent_implementation(nricp_dp):
    points, _ = _sphere_patch()
    state = _state(nricp_dp, points, 13, seed=10)
    _, blocks, system = _system(nricp_dp, points, state, _all_weights())

    diagonal, rhs, interactions = _numpy_assembly(blocks, len(state["nodes"]))
    assert np.allclose(np.asarray(system["diagonal"]), diagonal, atol=1e-12)
    assert np.allclose(np.asarray(system["rhs"]), rhs, atol=1e-12)

    edges = np.asarray(system["edges"])
    edge_blocks = np.asarray(system["edge_blocks"])
    assert len(edges) == len(interactions)
    for edge, block in zip(edges, edge_blocks):
        assert tuple(edge) in interactions
        assert np.allclose(block, interactions[tuple(edge)], atol=1e-12)


def test_dense_expansion_is_symmetric_and_matches_the_blocks(nricp_dp):
    points, _ = _sphere_patch()
    state = _state(nricp_dp, points, 13, seed=11)
    _, blocks, system = _system(nricp_dp, points, state, _all_weights())

    dense = np.asarray(
        nricp_dp.normal_equations_dense(
            system["diagonal"], system["edges"], system["edge_blocks"], system["rhs"]
        )
    )
    node_count = len(state["nodes"])
    assert dense.shape == (node_count * 6, node_count * 6)
    assert np.allclose(dense, dense.T, atol=1e-12)

    diagonal, _, interactions = _numpy_assembly(blocks, node_count)
    expected = np.zeros_like(dense)
    for node in range(node_count):
        expected[node * 6 : node * 6 + 6, node * 6 : node * 6 + 6] += diagonal[node]
    for (a, b), block in interactions.items():
        expected[a * 6 : a * 6 + 6, b * 6 : b * 6 + 6] += block
        expected[b * 6 : b * 6 + 6, a * 6 : a * 6 + 6] += block.T
    assert np.allclose(dense, expected, atol=1e-12)


def test_direct_solve_matches_a_dense_solve(nricp_dp):
    points, _ = _sphere_patch()
    state = _state(nricp_dp, points, 13, seed=12)
    _, _, system = _system(nricp_dp, points, state, _all_weights())
    node_count = len(state["nodes"])

    lam = 1e-3
    dense = np.asarray(
        nricp_dp.normal_equations_dense(
            system["diagonal"], system["edges"], system["edge_blocks"], system["rhs"]
        )
    )
    damped = dense + lam * np.eye(node_count * 6)
    rhs = np.asarray(system["rhs"]).reshape(-1)
    expected = np.linalg.solve(damped, rhs)

    solved = nricp_dp.solve_normal_equations(
        system["diagonal"],
        system["edges"],
        system["edge_blocks"],
        system["rhs"],
        damping=lam,
    )
    delta = np.asarray(solved["delta"])
    assert solved["ok"]
    assert np.allclose(delta, expected, atol=1e-9)
    # The reported residual is the sparse system's own, so it must be tiny.
    assert solved["residual_norm"] < 1e-8 * max(np.linalg.norm(rhs), 1.0)


def test_relaxation_without_damping_reduces_the_energy(nricp_dp):
    """Pins the sign convention: the solved direction must be a descent direction."""
    points, _ = _sphere_patch()
    node_count = 13
    state = _state(nricp_dp, points, node_count, seed=13)
    weights = _all_weights()
    result, _, system = _system(nricp_dp, points, state, weights)

    before = result["energies"]["total"]
    # A small damping keeps the system positive definite while barely changing
    # the direction; the energy must drop for a modest step along it.
    solved = nricp_dp.solve_normal_equations(
        system["diagonal"],
        system["edges"],
        system["edge_blocks"],
        system["rhs"],
        damping=1e-6,
    )
    delta = np.asarray(solved["delta"])
    moved = nricp_dp.retract_deformation(
        state["rotations"], state["translations"], state["nodes"], delta
    )
    moved_state = dict(state)
    moved_state["rotations"] = moved["rotations"]
    moved_state["translations"] = moved["translations"]
    after = _blocks(nricp_dp, points, moved_state, weights)["energies"]["total"]
    assert after < before, f"energy rose from {before} to {after}"


def test_damping_shortens_the_step(nricp_dp):
    points, _ = _sphere_patch()
    state = _state(nricp_dp, points, 13, seed=14)
    _, _, system = _system(nricp_dp, points, state, _all_weights())

    lengths = []
    for lam in (1e-6, 1e-2, 1.0):
        solved = nricp_dp.solve_normal_equations(
            system["diagonal"],
            system["edges"],
            system["edge_blocks"],
            system["rhs"],
            damping=lam,
        )
        lengths.append(np.linalg.norm(np.asarray(solved["delta"])))
    assert lengths[0] > lengths[1] > lengths[2]


def test_diagonal_scaling_damping_keeps_the_system_solvable(nricp_dp):
    """Even a bare system (no anchor) must solve once scaled damping is added."""
    points, _ = _sphere_patch()
    state = _state(nricp_dp, points, 13, seed=15)
    weights = dict(_all_weights())
    weights["anchor"] = 0.0
    _, _, system = _system(nricp_dp, points, state, weights)

    unscaled = nricp_dp.solve_normal_equations(
        system["diagonal"],
        system["edges"],
        system["edge_blocks"],
        system["rhs"],
        damping=1e-4,
    )
    scaled = nricp_dp.solve_normal_equations(
        system["diagonal"],
        system["edges"],
        system["edge_blocks"],
        system["rhs"],
        damping=1e-4,
        diagonal_scaling=True,
    )
    assert np.all(np.isfinite(np.asarray(unscaled["delta"])))
    assert np.all(np.isfinite(np.asarray(scaled["delta"])))
    assert scaled["ok"]


def test_fused_path_matches_the_block_reference(nricp_dp):
    """The solver's single-pass path must reproduce the verified block path.

    `energy_blocks` + `assemble_normal_equations` is the reference the analytic
    Jacobians were checked against; the fused path is what the solver actually
    runs, and it may not drift from that reference in either the system or the
    energies.
    """
    points, _ = _sphere_patch()
    node_count = 13
    state = _state(nricp_dp, points, node_count, seed=21)
    weights = _all_weights()
    mask = np.ones(len(points), dtype=np.int64)
    mask[[2, 5, 6]] = 0

    for label, current_mask in (("unmasked", None), ("masked", mask)):
        reference_result, blocks, reference = _system(
            nricp_dp, points, state, weights, current_mask
        )
        accepted = np.ones(len(points), dtype=np.int64)
        fused = nricp_dp.fused_system(
            points,
            state["binding"]["nodes"],
            state["binding"]["weights"],
            state["binding"]["counts"],
            state["rotations"],
            state["translations"],
            state["nodes"],
            state["correspondence_points"],
            state["correspondence_normals"],
            accepted,
            np.ones(len(points)),
            state["edges"],
            np.array([0, len(points) // 2], dtype=np.int64),
            state["correspondence_points"][[0, len(points) // 2]] + 0.01,
            weights,
            current_mask,
        )
        assert np.allclose(
            np.asarray(fused["diagonal"]), np.asarray(reference["diagonal"]), atol=1e-12
        ), label
        assert np.allclose(
            np.asarray(fused["rhs"]), np.asarray(reference["rhs"]), atol=1e-12
        ), label
        assert np.array_equal(
            np.asarray(fused["edges"]), np.asarray(reference["edges"])
        ), label
        assert np.allclose(
            np.asarray(fused["edge_blocks"]),
            np.asarray(reference["edge_blocks"]),
            atol=1e-12,
        ), label
        for key, value in reference_result["energies"].items():
            assert fused["energies"][key] == pytest.approx(value, rel=1e-12), (
                label,
                key,
            )
