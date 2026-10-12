#include "core/sampling.h"

#include <algorithm>
#include <limits>
#include <utility>

namespace nricp {

std::vector<int> farthest_point_seeds(
    const std::vector<Vec3>& points, int count, int first_index) {
    std::vector<int> seeds;
    if (points.empty() || count <= 0) {
        return seeds;
    }
    count = std::min(count, static_cast<int>(points.size()));
    first_index = std::clamp(first_index, 0, static_cast<int>(points.size()) - 1);

    std::vector<double> nearest(points.size(),
                                std::numeric_limits<double>::infinity());
    seeds.reserve(count);
    seeds.push_back(first_index);
    for (std::size_t i = 0; i < points.size(); ++i) {
        nearest[i] = norm(points[i] - points[first_index]);
    }
    while (static_cast<int>(seeds.size()) < count) {
        int best = 0;
        double best_distance = -1.0;
        for (std::size_t i = 0; i < points.size(); ++i) {
            if (nearest[i] > best_distance) {
                best_distance = nearest[i];
                best = static_cast<int>(i);
            }
        }
        seeds.push_back(best);
        for (std::size_t i = 0; i < points.size(); ++i) {
            nearest[i] = std::min(nearest[i], norm(points[i] - points[best]));
        }
    }
    return seeds;
}

std::vector<VertexBinding> bind_vertices(
    const std::vector<Vec3>& vertices,
    const std::vector<Vec3>& nodes,
    int max_nodes) {
    std::vector<VertexBinding> bindings(vertices.size());
    if (nodes.empty()) {
        return bindings;
    }
    // The binding structure has a fixed number of slots, so the requested
    // support is clamped to it as well as to the number of available nodes.
    const int support =
        std::max(1, std::min({max_nodes, static_cast<int>(nodes.size()),
                              VertexBinding::kMaxNodes}));
    std::vector<std::pair<double, int>> distances(nodes.size());

    for (std::size_t vertex = 0; vertex < vertices.size(); ++vertex) {
        for (std::size_t node = 0; node < nodes.size(); ++node) {
            distances[node] = {norm(vertices[vertex] - nodes[node]),
                               static_cast<int>(node)};
        }
        std::partial_sort(distances.begin(), distances.begin() + support,
                          distances.end());

        VertexBinding binding;
        binding.count = support;
        const double dmax = distances[support - 1].first;
        double total = 0.0;
        for (int slot = 0; slot < support; ++slot) {
            const double ratio = dmax > 0.0 ? distances[slot].first / dmax : 0.0;
            const double falloff = 1.0 - ratio;
            binding.nodes[slot] = distances[slot].second;
            binding.weights[slot] = falloff * falloff;
            total += binding.weights[slot];
        }
        for (int slot = 0; slot < support; ++slot) {
            binding.weights[slot] =
                total > 0.0 ? binding.weights[slot] / total
                            : 1.0 / static_cast<double>(support);
        }
        bindings[vertex] = binding;
    }
    return bindings;
}

}  // namespace nricp
