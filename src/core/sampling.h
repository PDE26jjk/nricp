#pragma once

#include <vector>

#include "core/vec3.h"

namespace nricp {

// Deterministic farthest-point sampling. Ties are broken by the lowest index,
// and the first seed is `first_index`, so two runs on the same input are
// identical and two implementations (host and device) can be compared exactly.
std::vector<int> farthest_point_seeds(
    const std::vector<Vec3>& points, int count, int first_index = 0);

// Binding of one vertex to the nodes that drive it.
struct VertexBinding {
    static constexpr int kMaxNodes = 4;
    int nodes[kMaxNodes] = {-1, -1, -1, -1};
    double weights[kMaxNodes] = {0.0, 0.0, 0.0, 0.0};
    int count = 0;
};

// Binds every vertex to its nearest `max_nodes` nodes, weighted by
// (1 - d/dmax)^2 normalized over that vertex's own support, where dmax is the
// largest distance inside the support (Sumner et al. 2007). The closest node
// therefore always carries the largest weight. Supports smaller than max_nodes
// (few nodes, coincident nodes) are handled without dividing by zero.
std::vector<VertexBinding> bind_vertices(
    const std::vector<Vec3>& vertices,
    const std::vector<Vec3>& nodes,
    int max_nodes = VertexBinding::kMaxNodes);

}  // namespace nricp
