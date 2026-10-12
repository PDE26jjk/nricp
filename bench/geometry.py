"""Shared geometry, case construction, and metrics for the wrapping benchmark.

The harness follows ``config/bench_config.json``: the template is decimated by
farthest-point grouping, the target is decimated with its *own* grouping (so the
solver really sees cross-topology input), and exact per-vertex ground truth is
available because the high-resolution pairs of one category share their
triangulation. Landmark targets are read from that ground truth, which is how
the reference numbers were produced.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_tosca(directory: Path, name: str) -> tuple[np.ndarray, np.ndarray]:
    """Load one high-resolution TOSCA shape (``.vert`` / ``.tri``, 1-based)."""
    vertices = np.loadtxt(directory / f"{name}.vert", dtype=np.float64)
    faces = np.loadtxt(directory / f"{name}.tri", dtype=np.int64) - 1
    return vertices, faces


def fps_seeds(points: np.ndarray, count: int) -> np.ndarray:
    """Deterministic farthest-point sampling; the first seed is index 0."""
    seeds = np.empty(count, dtype=np.int64)
    seeds[0] = 0
    nearest = np.linalg.norm(points - points[0], axis=1)
    for i in range(1, count):
        seeds[i] = int(np.argmax(nearest))
        nearest = np.minimum(
            nearest, np.linalg.norm(points - points[seeds[i]], axis=1)
        )
    return seeds


def group_labels(points: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    return cKDTree(points[seeds]).query(points)[1]


def group_centroids(
    vertices: np.ndarray, labels: np.ndarray, count: int
) -> np.ndarray:
    sums = np.zeros((count, 3))
    counts = np.zeros(count)
    np.add.at(sums, labels, vertices)
    np.add.at(counts, labels, 1.0)
    return sums / np.maximum(counts, 1.0)[:, None]


def grouped_mesh(
    vertices: np.ndarray,
    labels: np.ndarray,
    count: int,
    faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collapse a mesh onto group centroids, keeping the first winding seen."""
    seen: dict[tuple[int, int, int], np.ndarray] = {}
    for triangle in labels[faces]:
        key = tuple(sorted(int(v) for v in triangle))
        if key[0] == key[1] or key[1] == key[2] or key[0] == key[2]:
            continue
        seen.setdefault(key, triangle)
    grouped_faces = np.array(list(seen.values()), dtype=np.int64)
    used = np.unique(grouped_faces)
    remap = -np.ones(count, dtype=np.int64)
    remap[used] = np.arange(len(used))
    centroids = group_centroids(vertices, labels, count)[used]
    return centroids, remap[grouped_faces], used


def unique_edges(faces: np.ndarray) -> np.ndarray:
    edges = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    edges = np.sort(edges, axis=1)
    return np.unique(edges, axis=0)


def face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    triangles = vertices[faces]
    normals = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.maximum(lengths, 1e-300)


@dataclass
class Case:
    """One benchmark pair, prepared exactly as the pinned config prescribes."""

    name: str
    template_vertices: np.ndarray
    template_faces: np.ndarray
    target_vertices: np.ndarray
    target_faces: np.ndarray
    ground_truth_vertices: np.ndarray
    rigid_transform: np.ndarray
    aligned_template_vertices: np.ndarray
    landmark_indices: np.ndarray
    landmark_targets: np.ndarray
    distance_threshold: float
    diagonal: float
    config: dict = field(repr=False, default_factory=dict)

    @property
    def template_edges(self) -> np.ndarray:
        return unique_edges(self.template_faces)


def procrustes_transform(
    source_points: np.ndarray, target_points: np.ndarray
) -> np.ndarray:
    """Rigid transform (4x4) mapping source points onto target points.

    Closed-form Kabsch fit over the landmark correspondences: the centroids and
    the covariance supply the translation and rotation, and the reflection case
    is excluded by flipping the smallest singular direction. Deterministic — no
    sampling, no iteration — which is what makes a prepared case reproducible.
    """
    source_center = source_points.mean(axis=0)
    target_center = target_points.mean(axis=0)
    covariance = (source_points - source_center).T @ (target_points - target_center)
    u, _, vt = np.linalg.svd(covariance)
    correction = np.diag([1.0, 1.0, np.sign(np.linalg.det(vt.T @ u.T))])
    rotation = vt.T @ correction @ u.T
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = target_center - rotation @ source_center
    return transform


def _apply(transform: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    return vertices @ transform[:3, :3].T + transform[:3, 3]


def _icp_transform(
    source_vertices: np.ndarray,
    source_faces: np.ndarray,
    target_vertices: np.ndarray,
    target_faces: np.ndarray,
    rigid_cfg: dict,
) -> np.ndarray:
    """Surface-based ICP initialization.

    Kept for comparison only: the library's surface sampling is *not* seedable
    through the global numpy generator, so this path is not reproducible and is
    not the pinned default.
    """
    import trimesh

    source = trimesh.Trimesh(
        vertices=source_vertices, faces=source_faces, process=False
    )
    target = trimesh.Trimesh(
        vertices=target_vertices, faces=target_faces, process=False
    )
    a = source.sample(int(rigid_cfg["source_samples"]))
    b = target.sample(int(rigid_cfg["target_samples"]))
    transform, _, _ = trimesh.registration.icp(
        a,
        b,
        threshold=float(rigid_cfg["threshold"]),
        max_iterations=int(rigid_cfg["max_iterations"]),
        scale=False,
    )
    return np.asarray(transform, dtype=np.float64)


def build_case(config: dict, data_dir: Path, pair: dict) -> Case:
    """Prepare one pair: decimation, ground truth, rigid init, and landmarks."""
    template_hi_v, template_hi_f = load_tosca(data_dir, pair["template"])
    target_hi_v, _ = load_tosca(data_dir, pair["target"])

    # Template: farthest-point grouping on the template shape.
    template_seeds = fps_seeds(template_hi_v, int(config["template_vertices"]))
    template_labels = group_labels(template_hi_v, template_seeds)
    template_v, template_f, used = grouped_mesh(
        template_hi_v, template_labels, int(config["template_vertices"]), template_hi_f
    )

    # Target: an independent grouping, so the solver cannot see the shared
    # indexing that the scoring uses.
    target_seeds = fps_seeds(target_hi_v, int(config["target_vertices"]))
    target_labels = group_labels(target_hi_v, target_seeds)
    target_v, target_f, _ = grouped_mesh(
        target_hi_v, target_labels, int(config["target_vertices"]), template_hi_f
    )

    # Ground truth: the template's grouping applied to the target's
    # high-resolution vertices, i.e. where each template vertex really is.
    ground_truth = group_centroids(
        target_hi_v, template_labels, int(config["template_vertices"])
    )[used]

    diagonal = float(np.linalg.norm(target_v.max(axis=0) - target_v.min(axis=0)))
    threshold = float(config["distance_threshold"]["fraction_of_diagonal"]) * diagonal

    # Landmarks: farthest-point spread over the template, with targets taken
    # from the ground truth — equivalent to hand-placed control points, which is
    # what the alignment below consumes.
    landmark_indices = fps_seeds(template_v, int(config["landmarks"]))
    landmark_targets = ground_truth[landmark_indices]

    rigid_cfg = config["rigid_init"]
    method = rigid_cfg.get("method", "landmark_procrustes")
    if method == "landmark_procrustes":
        transform = procrustes_transform(
            template_v[landmark_indices], landmark_targets
        )
    elif method == "icp":
        transform = _icp_transform(
            template_v, template_f, target_v, target_f, rigid_cfg
        )
    elif method == "identity":
        transform = np.eye(4)
    else:
        raise ValueError(f"unknown rigid_init method: {method!r}")
    aligned = _apply(transform, template_v)

    return Case(
        name=pair["name"],
        template_vertices=template_v,
        template_faces=template_f,
        target_vertices=target_v,
        target_faces=target_f,
        ground_truth_vertices=ground_truth,
        rigid_transform=transform,
        aligned_template_vertices=aligned,
        landmark_indices=landmark_indices,
        landmark_targets=landmark_targets,
        distance_threshold=threshold,
        diagonal=diagonal,
        config=config,
    )


def correspondence_errors(result: np.ndarray, ground_truth: np.ndarray) -> dict:
    distances = np.linalg.norm(result - ground_truth, axis=1)
    return {
        "corr_mean": float(distances.mean()),
        "corr_p95": float(np.percentile(distances, 95)),
        "corr_max": float(distances.max()),
    }


def one_sided_chamfer(result_vertices: np.ndarray, case: Case) -> float:
    """Mean distance from the fitted template to the target *surface*."""
    import trimesh

    target = trimesh.Trimesh(
        vertices=case.target_vertices, faces=case.target_faces, process=False
    )
    closest, _, _ = trimesh.proximity.closest_point(target, result_vertices)
    return float(np.linalg.norm(closest - result_vertices, axis=1).mean())


def stretch(result_vertices: np.ndarray, case: Case) -> tuple[float, float]:
    edges = case.template_edges
    rest = np.linalg.norm(
        case.template_vertices[edges[:, 0]] - case.template_vertices[edges[:, 1]], axis=1
    )
    current = np.linalg.norm(
        result_vertices[edges[:, 0]] - result_vertices[edges[:, 1]], axis=1
    )
    relative = np.abs(current / np.maximum(rest, 1e-12) - 1.0)
    return float(relative.mean()), float(np.percentile(relative, 95))


def foldover(result_vertices: np.ndarray, case: Case) -> float:
    reference = face_normals(case.aligned_template_vertices, case.template_faces)
    current = face_normals(result_vertices, case.template_faces)
    return float((np.einsum("ij,ij->i", reference, current) < 0).mean())


def evaluate(result_vertices: np.ndarray, case: Case) -> dict:
    metrics = correspondence_errors(result_vertices, case.ground_truth_vertices)
    metrics["corr_mean_fraction_of_diagonal"] = metrics["corr_mean"] / case.diagonal
    metrics["chamfer_A2B"] = one_sided_chamfer(result_vertices, case)
    stretch_mean, stretch_p95 = stretch(result_vertices, case)
    metrics["stretch_mean"] = stretch_mean
    metrics["stretch_p95"] = stretch_p95
    metrics["foldover"] = foldover(result_vertices, case)
    return metrics


def gate_failures(
    metrics: dict, config: dict, baseline: dict | None
) -> list[str]:
    """Which acceptance gates this result violates (empty means it passes).

    The gates are anchored on the recorded reference baseline for the same pair:
    each metric has to stay within a tolerance of the best result the reference
    implementation reached there. Absolute targets are reported by the harness
    alongside the metrics and never gate a result.
    """
    if baseline is None:
        return ["no reference baseline recorded for this pair"]

    gates = config["gates"]
    relative = gates["relative"]
    best = baseline["best"]
    failures = []

    limits = (
        (
            "corr_mean",
            best["corr_mean"] * float(relative["corr_mean_ratio"]),
        ),
        (
            "corr_p95",
            best["corr_p95"] * float(relative["corr_p95_ratio"]),
        ),
        (
            "stretch_p95",
            best["stretch_p95"] + float(relative["stretch_p95_margin"]),
        ),
        (
            "foldover",
            best["foldover"] + float(relative["foldover_margin"]),
        ),
    )
    for name, limit in limits:
        value = metrics[name]
        # A non-finite metric must never pass: NaN compares false against every
        # limit, which would silently turn a broken solve into "all gates passed".
        if not math.isfinite(value):
            failures.append(f"{name} is not finite ({value!r})")
            continue
        if value > limit:
            failures.append(
                f"{name} {value:.5f} > {limit:.5f} "
                f"(reference {best[name]:.5f} by {best['method']})"
            )
    return failures


def save_error_map(
    path_stem: Path, result_vertices: np.ndarray, case: Case, title: str
) -> Path:
    """Per-vertex error data, plus a rendered map when matplotlib is present."""
    distances = np.linalg.norm(result_vertices - case.ground_truth_vertices, axis=1)
    npz_path = path_stem.with_suffix(".npz")
    np.savez_compressed(
        npz_path,
        vertex_distances=distances,
        ground_truth_vertices=case.ground_truth_vertices,
        result_vertices=result_vertices,
    )
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - optional dependency
        return npz_path

    figure = plt.figure(figsize=(10, 4))
    for index, (vertices, name) in enumerate(
        ((case.template_vertices, "template"), (result_vertices, "result"))
    ):
        axis = figure.add_subplot(1, 2, index + 1, projection="3d")
        axis.scatter(
            vertices[:, 0],
            vertices[:, 1],
            vertices[:, 2],
            c=distances,
            s=1.5,
            cmap="turbo",
        )
        axis.set_title(f"{name}: distance to ground truth")
        axis.set_axis_off()
    figure.suptitle(title)
    png_path = path_stem.with_suffix(".png")
    figure.savefig(png_path, dpi=110, bbox_inches="tight")
    plt.close(figure)
    return png_path
