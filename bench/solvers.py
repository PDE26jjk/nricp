"""Solver backends for the benchmark.

``reference`` runs the library implementation the acceptance numbers came from;
``module`` runs this project's own extension module. Both are reached through
the same call so the harness, the gates, and the golden fixtures treat them
identically.
"""

from __future__ import annotations

import numpy as np

from geometry import Case


def _as_vertices(result) -> np.ndarray:
    """Accept either a mesh-like result or a plain vertex array."""
    return np.asarray(getattr(result, "vertices", result), dtype=np.float64)


def solve_reference(case: Case, method: str, reference_cfg: dict) -> np.ndarray:
    """Run the reference (library) implementation for one prepared case.

    The reference protocol solves *from the rigidly aligned template*: the
    config's rigid initialization is what gives the correspondence search a
    usable starting pose, and both reference solvers were measured that way.
    """
    import trimesh

    if method == "rigid":
        # The initialization itself, reported as the baseline row.
        return case.aligned_template_vertices.copy()

    # The reference implementation normalizes the source mesh *in place*, so it
    # gets private copies of the case arrays.
    source = trimesh.Trimesh(
        vertices=case.aligned_template_vertices.copy(),
        faces=case.template_faces.copy(),
        process=False,
    )
    target = trimesh.Trimesh(
        vertices=case.target_vertices.copy(),
        faces=case.target_faces.copy(),
        process=False,
    )
    common = {
        "source_landmarks": case.landmark_indices,
        "target_positions": case.landmark_targets,
        "distance_threshold": case.distance_threshold,
    }
    if method == "amberg":
        steps = reference_cfg["amberg"]["steps"]
        return _as_vertices(
            trimesh.registration.nricp_amberg(
                source, target, steps=[tuple(step) for step in steps], **common
            )
        )
    if method == "sumner":
        steps = reference_cfg["sumner"].get("steps")
        extra = {} if not steps else {"steps": [tuple(step) for step in steps]}
        return _as_vertices(
            trimesh.registration.nricp_sumner(source, target, **extra, **common)
        )
    raise ValueError(f"unknown reference method: {method!r}")


def solve_module(
    case: Case, method: str, solver_cfg: dict, module=None
) -> np.ndarray:
    """Run this project's own solver through the same prepared case.

    `method` selects the route; only `route_a` exists so far. The template is the
    rigidly aligned one, matching the protocol the reference baseline was
    measured under, and the distances are handed over in model units (the
    binding converts them into its normalized frame).
    """
    if module is None:
        raise ValueError("solve_module needs the extension module")
    if method not in ("route_a", "route_b", "rigid"):
        raise ValueError(f"unsupported native method: {method!r}")
    if method == "rigid":
        return case.aligned_template_vertices.copy()

    levels = int(solver_cfg["levels"])
    level_settings = []
    for level in range(levels):
        level_settings.append(
            {
                "point_to_plane": float(solver_cfg["point_to_plane_weight"][level]),
                "point_to_point": float(solver_cfg["point_to_plane_weight"][level])
                * float(solver_cfg["point_to_point_ratio"]),
                "smoothness": float(solver_cfg["smoothness_weight"][level]),
                "landmark": float(solver_cfg["landmark_weight"][level]),
                "anchor": float(solver_cfg["anchor_weight"]),
                "robust_kernel": solver_cfg["robust_kernel"],
                "robust_threshold": float(
                    solver_cfg["robust_threshold_fraction_of_diagonal"]
                )
                * case.diagonal,
                "max_outer_iterations": int(solver_cfg["max_outer_iterations"]),
                "max_inner_iterations": int(solver_cfg["max_inner_iterations"]),
                "outer_tolerance": float(solver_cfg["outer_tolerance"]),
                "inner_tolerance": float(solver_cfg["inner_tolerance"]),
            }
        )

    solve = getattr(module, f"solve_{method}")
    result = solve(
        case.aligned_template_vertices,
        case.template_faces,
        case.target_vertices,
        case.target_faces,
        level_settings,
        landmark_indices=case.landmark_indices.astype(np.int64),
        landmark_targets=case.landmark_targets,
        distance_threshold=float(case.distance_threshold),
        min_normal_cosine=float(solver_cfg["min_normal_cosine"]),
        min_slope=float(solver_cfg["min_slope"]),
        graph_levels=levels,
        nodes_initial=int(solver_cfg["nodes_initial"]),
        growth=int(solver_cfg["node_growth"]),
        pcg_tolerance=float(solver_cfg["pcg_tolerance"]),
        pcg_max_iterations=int(solver_cfg["pcg_max_iterations"]),
    )
    return np.asarray(result["vertices"], dtype=np.float64), result["report"]


def solve(backend: str, case: Case, method: str, config: dict, module=None):
    """Run one backend. Returns (vertices, run report or None)."""
    if backend == "reference":
        return solve_reference(case, method, config["reference"]), None
    if backend == "module":
        return solve_module(case, method, config["solver"], module)
    raise ValueError(f"unknown backend: {backend!r}")
