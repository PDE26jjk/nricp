"""The device graph builder against the core layer's, bit for bit.

The accelerated path builds the template graph on the device, because the
construction is two sweeps of every vertex against every node of a level and
that -- not the solve -- is where a single wrap's time goes. The core layer's
builder stays as the reference, and this pins the two together: the seeds, the
four nodes, the weights and the edge set have to come out *identical*, not
merely close. Getting there needed the device's distance arithmetic to stop
contracting into fused multiply-adds, which is what the intrinsics in
`graph_build.cu` are for: a difference in the last bit of a distance is enough
to swap the order two nearly-tied template vertices are chosen in.
"""

from __future__ import annotations

import numpy as np
import pytest

from test_core_geometry import _uv_sphere


def test_the_device_graph_builder_matches_the_core_layers(nricp_dp):
    if not nricp_dp.device_info()["available"]:
        pytest.skip("no CUDA device here")
    for rings, segments, nodes, growth in ((12, 16, 40, 2), (24, 20, 100, 2), (32, 24, 150, 3)):
        template, faces = _uv_sphere(rings=rings, segments=segments)
        host = nricp_dp.build_template_graph(template, faces, 3, nodes, growth)
        device = nricp_dp.build_template_graph(
            template, faces, 3, nodes, growth, 4, True
        )
        assert host["normalization"]["scale"] == device["normalization"]["scale"]
        assert len(host["levels"]) == len(device["levels"])
        for level, (left, right) in enumerate(zip(host["levels"], device["levels"])):
            assert np.array_equal(
                np.asarray(left["node_template_indices"]),
                np.asarray(right["node_template_indices"]),
            ), f"level {level} sampled different nodes"
            assert np.array_equal(
                np.asarray(left["node_positions"]),
                np.asarray(right["node_positions"]),
            ), f"level {level} placed its nodes differently"
            assert repr(left["bindings"]) == repr(right["bindings"]), (
                f"level {level} bound the vertices differently"
            )
            assert np.array_equal(
                np.asarray(left["edges"]), np.asarray(right["edges"])
            ), f"level {level} connected the nodes differently"
