"""Correspondence search and its filters, each triggered on purpose."""

from __future__ import annotations

import numpy as np

REJECTED_ACCEPTED = 0
REJECTED_DISTANCE = 1
REJECTED_NORMAL = 2
REJECTED_SLOPE = 3
REJECTED_MASKED = 4


def _plane_target():
    """Unit square in the z = 0 plane, two triangles, outward normal +z."""
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    return points, faces


def test_each_filter_rejects_for_its_own_reason(nricp_dp):
    points, faces = _plane_target()
    up = np.array([0.0, 0.0, 1.0])
    queries = np.array(
        [
            [0.5, 0.5, 0.05],   # squarely above the surface: accepted
            [0.4, 0.4, 5.0],    # too far away
            [0.6, 0.6, 0.01],   # accepted geometry, but its normal is flipped
            [-0.3, 0.5, 0.0],   # outside the patch: sliding, rejected by slope
            [0.3, 0.7, 0.004],  # masked out of the data term
        ]
    )
    normals = np.array([up, up, -up, up, up])
    mask = np.ones(len(queries), dtype=np.int64)
    mask[4] = 0  # the last query is masked out of the data term

    result = nricp_dp.find_correspondences(
        points,
        faces,
        queries,
        normals,
        distance_threshold=1.0,
        min_normal_cosine=0.5,
        min_slope=0.5,
        mask=mask,
    )
    rejection = np.asarray(result["rejection"])
    assert rejection[0] == REJECTED_ACCEPTED
    assert rejection[1] == REJECTED_DISTANCE
    assert rejection[2] == REJECTED_NORMAL
    assert rejection[3] == REJECTED_SLOPE
    assert rejection[4] == REJECTED_MASKED

    counts = result["counts"]
    assert counts["total"] == 5
    assert counts["accepted"] == 1
    assert counts["rejected_distance"] == 1
    assert counts["rejected_normal"] == 1
    assert counts["rejected_slope"] == 1
    assert counts["rejected_masked"] == 1


def test_slope_filter_rejects_sliding_connections(nricp_dp):
    """The filter drops connections that run along the surface.

    |n . d| / |d| is 1 for a connection along the normal (a vertex facing its
    closest point) and 0 for one along the surface (the closest point is off to
    the side, so the vertex is sliding). Only the sliding ones are useless, so
    the threshold is a *minimum*; earlier revisions had the comparison the other
    way round, and this test is what makes that reversal visible.
    """
    points, faces = _plane_target()
    up = np.array([0.0, 0.0, 1.0])
    hovering = np.array([[0.5, 0.5, 0.2]])
    sliding = np.array([[-0.3, 0.5, 0.0]])
    normals = np.array([up])

    kept = nricp_dp.find_correspondences(
        points, faces, hovering, normals, min_slope=0.5
    )
    assert kept["counts"]["accepted"] == 1

    rejected = nricp_dp.find_correspondences(
        points, faces, sliding, normals, min_slope=0.5
    )
    assert rejected["counts"]["rejected_slope"] == 1

    # The comparison is strict, and zero disables the filter entirely.
    at_the_line = nricp_dp.find_correspondences(
        points, faces, sliding, normals, min_slope=0.0
    )
    assert at_the_line["counts"]["accepted"] == 1


def test_distance_statistics_cover_accepted_correspondences_only(nricp_dp):
    points, faces = _plane_target()
    up = np.array([0.0, 0.0, 1.0])
    heights = np.array([0.01, 0.02, 0.03, 4.0])
    queries = np.stack(
        [
            np.full(4, 0.5),
            np.full(4, 0.5),
            heights,
        ],
        axis=1,
    )
    normals = np.broadcast_to(up, (4, 3)).copy()
    result = nricp_dp.find_correspondences(
        points, faces, queries, normals, distance_threshold=1.0
    )
    counts = result["counts"]
    assert counts["accepted"] == 3
    assert counts["rejected_distance"] == 1
    assert counts["distance_mean"] == np.mean(heights[:3])
    assert counts["distance_max"] == heights[2]
    assert counts["distance_p95"] <= counted_max(result)


def counted_max(result) -> float:
    distances = np.asarray(result["distances"])
    accepted = np.asarray(result["accepted"]) == 1
    return float(distances[accepted].max())


def test_a_query_on_the_surface_is_accepted_even_with_a_tight_slope(nricp_dp):
    points, faces = _plane_target()
    query = np.array([[0.5, 0.5, 0.0]])
    normal = np.array([[0.0, 0.0, 1.0]])
    result = nricp_dp.find_correspondences(
        points, faces, query, normal, min_slope=1e-6
    )
    assert result["counts"]["accepted"] == 1
    assert np.asarray(result["distances"])[0] == 0.0


def test_closest_points_follow_the_surface_not_the_vertex_cloud(nricp_dp):
    """Sliding along a triangle is accepted; the point stays on the surface."""
    points, faces = _plane_target()
    query = np.array([[0.5, 0.5, 0.05]])
    normal = np.array([[0.0, 0.0, 1.0]])
    result = nricp_dp.find_correspondences(points, faces, query, normal)
    closest = np.asarray(result["points"])[0]
    assert np.allclose(closest, [0.5, 0.5, 0.0], atol=1e-12)
    assert np.allclose(np.asarray(result["normals"])[0], [0.0, 0.0, 1.0])
