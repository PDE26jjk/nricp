#include "core/sampling.h"

#include <algorithm>
#include <limits>
#include <utility>

namespace nricp {
namespace {

// Both passes below only ever *rank* distances, and a square root is monotone,
// so ranking by the square selects exactly the same points. What is squared is
// only ever compared or minimised; the values that reach the binding weights are
// square-rooted at the end, which keeps them bit-identical to what taking the
// root up front produced.
double squared_distance(const Vec3& a, const Vec3& b) noexcept {
    return squared_norm(a - b);
}

// Keeps a small list of the nearest entries, sorted by (distance, index) -- the
// order a partial sort of the whole set would have left in front.
void keep_nearest(
    double* squared, int* index, int& held, int capacity, double value, int entry) {
    if (held == capacity) {
        const bool beats_worst =
            value < squared[capacity - 1] ||
            (value == squared[capacity - 1] && entry < index[capacity - 1]);
        if (!beats_worst) {
            return;
        }
    }
    int at = held < capacity ? held++ : capacity - 1;
    while (at > 0 && (squared[at - 1] > value ||
                      (squared[at - 1] == value && index[at - 1] > entry))) {
        squared[at] = squared[at - 1];
        index[at] = index[at - 1];
        --at;
    }
    squared[at] = value;
    index[at] = entry;
}

}  // namespace

std::vector<int> farthest_point_seeds(
    const std::vector<Vec3>& points, int count, int first_index) {
    std::vector<int> seeds;
    if (points.empty() || count <= 0) {
        return seeds;
    }
    count = std::min(count, static_cast<int>(points.size()));
    first_index = std::clamp(first_index, 0, static_cast<int>(points.size()) - 1);

    // Squared distances to the set chosen so far, with the farthest point found
    // in the same sweep that updates them: one pass over the points per node
    // instead of two, and no square root. The sweep keeps the first maximum, so
    // ties still go to the lower index.
    std::vector<double> nearest(points.size());
    seeds.reserve(count);
    seeds.push_back(first_index);
    int best = first_index;
    double best_squared = -1.0;
    for (std::size_t i = 0; i < points.size(); ++i) {
        nearest[i] = squared_distance(points[i], points[first_index]);
        if (nearest[i] > best_squared) {
            best_squared = nearest[i];
            best = static_cast<int>(i);
        }
    }
    while (static_cast<int>(seeds.size()) < count) {
        seeds.push_back(best);
        const Vec3& chosen = points[static_cast<std::size_t>(best)];
        best_squared = -1.0;
        for (std::size_t i = 0; i < points.size(); ++i) {
            nearest[i] = std::min(nearest[i], squared_distance(points[i], chosen));
            if (nearest[i] > best_squared) {
                best_squared = nearest[i];
                best = static_cast<int>(i);
            }
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
    // The nearest `support` nodes of each vertex, kept in a list the size of the
    // support rather than by filling and partially sorting the whole node set,
    // and ranked by squared distance rather than by the distance itself. The
    // square roots are then the support's worth per vertex instead of one per
    // (vertex, node) pair.
    //
    // A uniform grid over the nodes was tried here and measured slower, not
    // faster: a level has a few hundred nodes at most, so the exhaustive sweep
    // costs about four nanoseconds per pair while a grid adds a cell enumeration
    // and pointer chase per candidate. The win has to come from comparing fewer
    // pairs, not from making each comparison cheaper -- see the sampling above,
    // where the update is the part that can be localised.
    std::vector<double> best_squared(static_cast<std::size_t>(support));
    std::vector<int> best_index(static_cast<std::size_t>(support));

    for (std::size_t vertex = 0; vertex < vertices.size(); ++vertex) {
        const Vec3& point = vertices[vertex];
        int held = 0;
        for (std::size_t node = 0; node < nodes.size(); ++node) {
            keep_nearest(best_squared.data(), best_index.data(), held, support,
                         squared_distance(point, nodes[node]),
                         static_cast<int>(node));
        }

        VertexBinding binding;
        binding.count = support;
        const double dmax = std::sqrt(best_squared[static_cast<std::size_t>(support - 1)]);
        double total = 0.0;
        for (int slot = 0; slot < support; ++slot) {
            const double distance = std::sqrt(best_squared[static_cast<std::size_t>(slot)]);
            const double ratio = dmax > 0.0 ? distance / dmax : 0.0;
            const double falloff = 1.0 - ratio;
            binding.nodes[slot] = best_index[static_cast<std::size_t>(slot)];
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
