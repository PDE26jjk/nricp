#include "solver/fused_assembly.h"

#include <cmath>
#include <map>
#include <utility>
#include <vector>

#include "core/rotation.h"

namespace nricp {
namespace {

// Accumulates one residual row at a time. Only the row's values and one 6-vector
// per node are ever live, so nothing proportional to the number of residuals is
// allocated.
class Accumulator {
public:
    Accumulator(int node_count, bool with_jacobians)
        : with_jacobians_(with_jacobians) {
        system_.node_count = node_count;
        system_.diagonal.assign(static_cast<std::size_t>(node_count), Mat6{});
        system_.rhs.assign(static_cast<std::size_t>(node_count) * 6, 0.0);
    }

    void add_row(
        BlockKind kind,
        const int* nodes,
        const double jacobians[kMaxBlockSlots][6],
        int slots,
        double value) {
        energies_.add(kind, value * value);
        if (!with_jacobians_) {
            return;
        }
        for (int slot = 0; slot < slots; ++slot) {
            const int node = nodes[slot];
            if (node < 0 || node >= system_.node_count) {
                continue;
            }
            Mat6& diagonal = system_.diagonal[static_cast<std::size_t>(node)];
            for (int row = 0; row < 6; ++row) {
                for (int column = 0; column < 6; ++column) {
                    diagonal.at(row, column) +=
                        jacobians[slot][row] * jacobians[slot][column];
                }
                system_.rhs[static_cast<std::size_t>(node) * 6 + row] -=
                    jacobians[slot][row] * value;
            }
            for (int other = slot + 1; other < slots; ++other) {
                const int other_node = nodes[other];
                if (other_node < 0 || other_node >= system_.node_count) {
                    continue;
                }
                const std::pair<int, int> key =
                    node < other_node ? std::make_pair(node, other_node)
                                      : std::make_pair(other_node, node);
                Mat6& block = interactions_[key];
                for (int row = 0; row < 6; ++row) {
                    for (int column = 0; column < 6; ++column) {
                        block.at(row, column) +=
                            node < other_node
                                ? jacobians[slot][row] * jacobians[other][column]
                                : jacobians[other][row] * jacobians[slot][column];
                    }
                }
            }
        }
    }

    FusedSystem finish() {
        FusedSystem result;
        result.energies = energies_;
        result.system = std::move(system_);
        result.system.off_diagonal.reserve(interactions_.size());
        for (auto& entry : interactions_) {
            NormalEquations::OffDiagonal block;
            block.a = entry.first.first;
            block.b = entry.first.second;
            block.value = entry.second;
            result.system.off_diagonal.push_back(block);
        }
        return result;
    }

private:
    bool with_jacobians_;
    NormalEquations system_;
    std::map<std::pair<int, int>, Mat6> interactions_;
    BlockEnergies energies_;
};

bool participates(const VertexMask* mask, std::size_t vertex) {
    if (mask == nullptr || mask->active.empty()) {
        return true;
    }
    return vertex < mask->active.size() && mask->active[vertex] != 0;
}

VertexBinding usable_binding(const VertexBinding& binding, int node_count) {
    VertexBinding compact;
    for (int slot = 0; slot < binding.count; ++slot) {
        const int node = binding.nodes[slot];
        if (node < 0 || node >= node_count || binding.weights[slot] == 0.0) {
            continue;
        }
        compact.nodes[compact.count] = node;
        compact.weights[compact.count] = binding.weights[slot];
        compact.count += 1;
    }
    return compact;
}

Vec3 node_offset(
    const Deformation& deformation, int node, const Vec3& point) {
    return deformation.nodes[static_cast<std::size_t>(node)].rotation *
           (point - deformation.node_rest_positions[static_cast<std::size_t>(node)]);
}

void fill_linear_row(
    const Vec3& lambda, const Vec3& rotated, double weight, double row[6]) {
    const Vec3 rotation_part = cross(lambda, rotated);
    row[0] = -weight * rotation_part.x;
    row[1] = -weight * rotation_part.y;
    row[2] = -weight * rotation_part.z;
    row[3] = weight * lambda.x;
    row[4] = weight * lambda.y;
    row[5] = weight * lambda.z;
}

// One residual of the shape `scale * (bound point - target)`, evaluated and
// accumulated row by row.
void accumulate_offset_rows(
    Accumulator& accumulator,
    BlockKind kind,
    const VertexBinding& binding,
    const Deformation& deformation,
    const Vec3& vertex_rest_point,
    const Vec3& deformed,
    const Vec3& target,
    double scale) {
    static const Vec3 axes[3] = {
        {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
    const Vec3 offset = (deformed - target) * scale;
    for (int row = 0; row < 3; ++row) {
        double jacobians[kMaxBlockSlots][6] = {};
        for (int slot = 0; slot < binding.count; ++slot) {
            const Vec3 rotated =
                node_offset(deformation, binding.nodes[slot], vertex_rest_point);
            fill_linear_row(axes[row], rotated, scale * binding.weights[slot],
                            jacobians[slot]);
        }
        const double value = row == 0   ? offset.x
                             : row == 1 ? offset.y
                                        : offset.z;
        accumulator.add_row(kind, binding.nodes, jacobians, binding.count, value);
    }
}

FusedSystem run_fused(
    const EnergyInputs& inputs,
    const Deformation& deformation,
    const EnergyWeights& weights,
    bool with_jacobians) {
    const int node_count = static_cast<int>(deformation.nodes.size());
    Accumulator accumulator(node_count, with_jacobians);
    const std::size_t vertex_count = inputs.template_points.size();

    for (std::size_t vertex = 0; vertex < vertex_count; ++vertex) {
        if (vertex >= inputs.correspondences.size()) {
            break;
        }
        const DataCorrespondence& correspondence = inputs.correspondences[vertex];
        if (!correspondence.accepted || !participates(inputs.mask, vertex)) {
            continue;
        }
        const VertexBinding binding =
            usable_binding(inputs.bindings[vertex], node_count);
        if (binding.count <= 0) {
            continue;
        }
        const Vec3 deformed =
            evaluate_point(inputs.template_points[vertex], binding, deformation);
        const double robust = std::max(0.0, correspondence.robust_weight);
        if (weights.point_to_point > 0.0) {
            accumulate_offset_rows(accumulator, BlockKind::PointToPoint, binding,
                                   deformation, inputs.template_points[vertex],
                                   deformed, correspondence.point,
                                   std::sqrt(weights.point_to_point * robust));
        }
        if (weights.point_to_plane > 0.0) {
            const double scale = std::sqrt(weights.point_to_plane * robust);
            double jacobians[kMaxBlockSlots][6] = {};
            for (int slot = 0; slot < binding.count; ++slot) {
                const Vec3 rotated = node_offset(
                    deformation, binding.nodes[slot], inputs.template_points[vertex]);
                fill_linear_row(correspondence.normal, rotated,
                                scale * binding.weights[slot], jacobians[slot]);
            }
            accumulator.add_row(BlockKind::PointToPlane, binding.nodes, jacobians,
                                binding.count,
                                scale * dot(correspondence.normal,
                                            deformed - correspondence.point));
        }
    }

    if (weights.smoothness > 0.0) {
        static const Vec3 axes[3] = {
            {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
        const double scale = std::sqrt(weights.smoothness);
        for (const Int2& edge : inputs.smoothness_edges) {
            if (edge.a < 0 || edge.b < 0 || edge.a >= node_count ||
                edge.b >= node_count ||
                edge.b >= static_cast<int>(inputs.node_positions.size())) {
                continue;
            }
            const Vec3 node_a = inputs.node_positions[edge.a];
            const Vec3 node_b = inputs.node_positions[edge.b];
            const Vec3 difference =
                (deformation.apply(edge.a, node_b) - deformation.apply(edge.b, node_b)) *
                scale;
            const Vec3 rotated =
                deformation.nodes[edge.a].rotation * (node_b - node_a);
            const int nodes[2] = {edge.a, edge.b};
            for (int row = 0; row < 3; ++row) {
                double jacobians[kMaxBlockSlots][6] = {};
                fill_linear_row(axes[row], rotated, scale, jacobians[0]);
                jacobians[1][row + 3] = -scale;
                const double value = row == 0   ? difference.x
                                     : row == 1 ? difference.y
                                                : difference.z;
                accumulator.add_row(BlockKind::Smoothness, nodes, jacobians, 2,
                                    value);
            }
        }
    }

    if (weights.anchor > 0.0) {
        const double scale = std::sqrt(weights.anchor);
        for (int node = 0; node < node_count; ++node) {
            const Vec3 omega =
                log_so3(deformation.nodes[static_cast<std::size_t>(node)].rotation);
            const Mat3 rotation_jacobian = left_jacobian_inverse(omega);
            const Vec3 translation =
                deformation.nodes[static_cast<std::size_t>(node)].translation;
            const int nodes[1] = {node};
            for (int row = 0; row < 6; ++row) {
                double jacobians[kMaxBlockSlots][6] = {};
                if (row < 3) {
                    for (int column = 0; column < 3; ++column) {
                        jacobians[0][column] =
                            scale * rotation_jacobian.at(row, column);
                    }
                } else {
                    jacobians[0][row] = scale;
                }
                const double value =
                    scale * (row == 0   ? omega.x
                             : row == 1 ? omega.y
                             : row == 2 ? omega.z
                             : row == 3 ? translation.x
                             : row == 4 ? translation.y
                                        : translation.z);
                accumulator.add_row(BlockKind::Anchor, nodes, jacobians, 1, value);
            }
        }
    }

    if (weights.landmark > 0.0) {
        const double scale = std::sqrt(weights.landmark);
        for (const Landmark& landmark : inputs.landmarks) {
            if (landmark.template_vertex < 0 ||
                landmark.template_vertex >= static_cast<int>(vertex_count)) {
                continue;
            }
            const std::size_t vertex =
                static_cast<std::size_t>(landmark.template_vertex);
            const VertexBinding binding =
                usable_binding(inputs.bindings[vertex], node_count);
            if (binding.count <= 0) {
                continue;
            }
            const Vec3 deformed =
                evaluate_point(inputs.template_points[vertex], binding, deformation);
            accumulate_offset_rows(accumulator, BlockKind::Landmark, binding,
                                   deformation, inputs.template_points[vertex],
                                   deformed, landmark.target, scale);
        }
    }

    return accumulator.finish();
}

}  // namespace

FusedSystem assemble_and_energy(
    const EnergyInputs& inputs,
    const Deformation& deformation,
    const EnergyWeights& weights) {
    return run_fused(inputs, deformation, weights, true);
}

BlockEnergies accumulate_energy(
    const EnergyInputs& inputs,
    const Deformation& deformation,
    const EnergyWeights& weights) {
    return run_fused(inputs, deformation, weights, false).energies;
}

}  // namespace nricp
