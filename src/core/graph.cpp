#include "core/graph.h"

#include <algorithm>
#include <cmath>
#include <set>

namespace nricp {
namespace {

// Connects every node to its nearest neighbours, keeping each undirected pair
// once and in a deterministic order. Neighbour count is fixed rather than
// radius-based so the edge set is a function of the node set alone.
std::vector<Int2> build_edges(
    const std::vector<Vec3>& node_positions, int neighbours = 4) {
    std::set<std::pair<int, int>> unique;
    const int count = static_cast<int>(node_positions.size());
    for (int node = 0; node < count; ++node) {
        std::vector<std::pair<double, int>> distances;
        distances.reserve(count);
        for (int other = 0; other < count; ++other) {
            if (other == node) {
                continue;
            }
            distances.push_back({norm(node_positions[node] - node_positions[other]),
                                 other});
        }
        const int taken = std::min<int>(neighbours, static_cast<int>(distances.size()));
        std::partial_sort(distances.begin(), distances.begin() + taken,
                          distances.end());
        for (int i = 0; i < taken; ++i) {
            const int other = distances[i].second;
            unique.insert({std::min(node, other), std::max(node, other)});
        }
    }
    std::vector<Int2> edges;
    edges.reserve(unique.size());
    for (const auto& pair : unique) {
        edges.push_back({pair.first, pair.second});
    }
    return edges;
}

}  // namespace

TemplateGraph build_template_graph(
    const Mesh& template_mesh, const GraphSettings& settings) {
    TemplateGraph graph;
    graph.normalization = compute_normalization(template_mesh);

    Mesh normalized = template_mesh;
    apply_normalization(normalized, graph.normalization);

    const int vertex_count = static_cast<int>(normalized.vertices.size());
    const int levels = std::max(1, settings.levels);
    graph.levels.reserve(levels);
    for (int level = 0; level < levels; ++level) {
        const double growth = std::pow(std::max(1, settings.node_growth),
                                       static_cast<double>(level));
        const int requested =
            std::max(1, static_cast<int>(std::lround(settings.nodes_initial * growth)));
        const int count = std::min(requested, vertex_count);

        GraphLevel level_data;
        level_data.node_template_indices =
            farthest_point_seeds(normalized.vertices, count);
        level_data.node_rest_positions.reserve(level_data.node_template_indices.size());
        for (const int index : level_data.node_template_indices) {
            level_data.node_rest_positions.push_back(normalized.vertices[index]);
        }
        level_data.bindings = bind_vertices(
            normalized.vertices, level_data.node_rest_positions,
            settings.max_nodes_per_vertex);
        level_data.edges = build_edges(level_data.node_rest_positions);
        graph.levels.push_back(std::move(level_data));
    }
    return graph;
}

Deformation transfer_to_level(
    const GraphLevel& from,
    const Deformation& from_deformation,
    const GraphLevel& to) {
    return transfer_level(from.node_rest_positions, from_deformation,
                          to.node_rest_positions);
}

}  // namespace nricp
