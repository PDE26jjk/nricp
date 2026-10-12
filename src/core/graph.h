#pragma once

#include <vector>

#include "core/deformation.h"
#include "core/mesh.h"
#include "core/sampling.h"

namespace nricp {

struct GraphSettings {
    int levels = 4;
    int nodes_initial = 100;
    int node_growth = 2;  // node count per level: nodes_initial * growth^level
    int max_nodes_per_vertex = VertexBinding::kMaxNodes;
};

// One resolution of the deformation graph: node rest positions sampled on the
// undeformed template, plus the bindings that are frozen for this level.
struct GraphLevel {
    std::vector<Vec3> node_rest_positions;
    std::vector<int> node_template_indices;
    std::vector<VertexBinding> bindings;
    std::vector<Int2> edges;  // node pairs the smoothness term connects

    int node_count() const { return static_cast<int>(node_rest_positions.size()); }
};

// The template side of a wrap. Everything here depends on the template only, so
// it is prepared once and reused for every target.
struct TemplateGraph {
    Normalization normalization;
    std::vector<GraphLevel> levels;
};

// Samples nodes on the undeformed template per level and freezes the bindings
// there. Node positions and bindings live in normalized space; the caller
// applies the returned normalization to the meshes it feeds to the solver.
TemplateGraph build_template_graph(
    const Mesh& template_mesh, const GraphSettings& settings);

// Warm-starts a finer level from a coarser one.
Deformation transfer_to_level(
    const GraphLevel& from,
    const Deformation& from_deformation,
    const GraphLevel& to);

// Connects every node to its nearest neighbours, keeping each undirected pair
// once and in a deterministic order. Exposed because the accelerated graph
// builder derives the same edge set from the same nodes, so the rule has to live
// in one place rather than in two that drift.
std::vector<Int2> build_node_edges(
    const std::vector<Vec3>& node_positions, int neighbours = 4);

}  // namespace nricp
