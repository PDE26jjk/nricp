#include "energy/energy.h"

#include <cmath>

#include "core/rotation.h"

namespace nricp {
namespace {

// Jacobian of a linear functional of one node's transform, `lambda^T (R v + t)`,
// with respect to that node's six tangent-space unknowns. `rotated` is `R v`,
// and the rotation part is `-weight * (lambda x rotated)` because a left
// perturbation turns the increment into `delta x rotated`.
void fill_linear_row(
    double row[6], const Vec3& lambda, const Vec3& rotated, double weight) {
    const Vec3 rotation_part = cross(lambda, rotated);
    row[0] = -weight * rotation_part.x;
    row[1] = -weight * rotation_part.y;
    row[2] = -weight * rotation_part.z;
    row[3] = weight * lambda.x;
    row[4] = weight * lambda.y;
    row[5] = weight * lambda.z;
}

// Offset block: three rows of the scaled `v' - target`. Shared by the
// point-to-point data block and the landmark block.
void add_offset_block(
    std::vector<ResidualBlock>& blocks,
    BlockKind kind,
    const VertexBinding& binding,
    const Deformation& deformation,
    const Vec3& vertex_rest_point,
    const Vec3& deformed,
    const Vec3& target,
    double scale,
    bool with_jacobians) {
    static const Vec3 axes[3] = {
        {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
    ResidualBlock block;
    block.kind = kind;
    block.row_count = 3;
    block.slots = binding.count;
    const Vec3 offset = (deformed - target) * scale;
    block.values[0] = offset.x;
    block.values[1] = offset.y;
    block.values[2] = offset.z;
    for (int slot = 0; slot < binding.count; ++slot) {
        const int node = binding.nodes[slot];
        block.nodes[slot] = node;
        // `a` is the vertex offset from this node, rotated by this node: the
        // rotation of a node acts about its own rest position.
        const Vec3 rotated = deformation.nodes[node].rotation *
                             (vertex_rest_point -
                              deformation.node_rest_positions[node]);
        const double weight = scale * binding.weights[slot];
        if (with_jacobians) {
            for (int row = 0; row < 3; ++row) {
                fill_linear_row(block.jacobians[slot][row], axes[row], rotated,
                                weight);
            }
        }
    }
    blocks.push_back(block);
}

// Point-to-plane block: one row, the scaled projection of the offset onto the
// target normal.
void add_plane_block(
    std::vector<ResidualBlock>& blocks,
    const VertexBinding& binding,
    const Deformation& deformation,
    const Vec3& vertex_rest_point,
    const Vec3& deformed,
    const Vec3& normal,
    const Vec3& target,
    double scale,
    bool with_jacobians) {
    ResidualBlock block;
    block.kind = BlockKind::PointToPlane;
    block.row_count = 1;
    block.slots = binding.count;
    block.values[0] = scale * dot(normal, deformed - target);
    for (int slot = 0; slot < binding.count; ++slot) {
        const int node = binding.nodes[slot];
        block.nodes[slot] = node;
        const Vec3 rotated = deformation.nodes[node].rotation *
                             (vertex_rest_point -
                              deformation.node_rest_positions[node]);
        if (with_jacobians) {
            fill_linear_row(block.jacobians[slot][0], normal, rotated,
                            scale * binding.weights[slot]);
        }
    }
    blocks.push_back(block);
}

bool participates(const VertexMask* mask, std::size_t vertex) {
    if (mask == nullptr || mask->active.empty()) {
        return true;
    }
    return vertex < mask->active.size() && mask->active[vertex] != 0;
}

// A usable binding names only existing nodes and carries only the slots that
// actually contribute. The distance-ratio weights always zero the farthest
// support node, so keeping it would add zero-weight blocks and wasted work to
// every stage downstream (evaluation, assembly, and the device kernels later).
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

}  // namespace

std::vector<ResidualBlock> build_blocks(
    const EnergyInputs& inputs,
    const Deformation& deformation,
    const EnergyWeights& weights,
    bool with_jacobians) {
    std::vector<ResidualBlock> blocks;
    const std::size_t vertex_count = inputs.template_points.size();
    blocks.reserve(vertex_count * 2 + inputs.smoothness_edges.size() +
                   deformation.nodes.size() + inputs.landmarks.size());

    // Data blocks: one point-to-point and one point-to-plane block per accepted
    // correspondence. A masked vertex contributes nothing to the data term.
    for (std::size_t vertex = 0; vertex < vertex_count; ++vertex) {
        if (vertex >= inputs.correspondences.size()) {
            break;
        }
        const DataCorrespondence& correspondence = inputs.correspondences[vertex];
        if (!correspondence.accepted || !participates(inputs.mask, vertex)) {
            continue;
        }
        const VertexBinding binding =
            usable_binding(inputs.bindings[vertex],
                           static_cast<int>(deformation.nodes.size()));
        if (binding.count <= 0) {
            continue;
        }
        const Vec3 deformed =
            evaluate_point(inputs.template_points[vertex], binding, deformation);
        const double robust = std::max(0.0, correspondence.robust_weight);
        if (weights.point_to_point > 0.0) {
            add_offset_block(blocks, BlockKind::PointToPoint, binding, deformation,
                             inputs.template_points[vertex], deformed,
                             correspondence.point,
                             std::sqrt(weights.point_to_point * robust),
                             with_jacobians);
        }
        if (weights.point_to_plane > 0.0) {
            add_plane_block(blocks, binding, deformation,
                            inputs.template_points[vertex], deformed,
                            correspondence.normal, correspondence.point,
                            std::sqrt(weights.point_to_plane * robust),
                            with_jacobians);
        }
    }

    // Smoothness: node a's transform applied to node b's position must agree
    // with node b's own transform applied there.
    if (weights.smoothness > 0.0) {
        static const Vec3 axes[3] = {
            {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
        const double scale = std::sqrt(weights.smoothness);
        for (const Int2& edge : inputs.smoothness_edges) {
            if (edge.a < 0 || edge.b < 0 ||
                edge.a >= static_cast<int>(deformation.nodes.size()) ||
                edge.b >= static_cast<int>(deformation.nodes.size()) ||
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

            ResidualBlock block;
            block.kind = BlockKind::Smoothness;
            block.row_count = 3;
            block.slots = 2;
            block.nodes[0] = edge.a;
            block.nodes[1] = edge.b;
            block.values[0] = difference.x;
            block.values[1] = difference.y;
            block.values[2] = difference.z;
            if (with_jacobians) {
                for (int row = 0; row < 3; ++row) {
                    fill_linear_row(block.jacobians[0][row], axes[row], rotated,
                                    scale);
                    block.jacobians[1][row][3] = -scale * axes[row].x;
                    block.jacobians[1][row][4] = -scale * axes[row].y;
                    block.jacobians[1][row][5] = -scale * axes[row].z;
                }
            }
            blocks.push_back(block);
        }
    }

    // Anchor: pulls every node back towards its rest transform (identity
    // rotation, zero translation). It is also what keeps the reduced system
    // positive definite where a node sees no data at all.
    if (weights.anchor > 0.0) {
        const double scale = std::sqrt(weights.anchor);
        for (std::size_t node = 0; node < deformation.nodes.size(); ++node) {
            const Vec3 omega = log_so3(deformation.nodes[node].rotation);
            ResidualBlock block;
            block.kind = BlockKind::Anchor;
            block.row_count = 6;
            block.slots = 1;
            block.nodes[0] = static_cast<int>(node);
            block.values[0] = scale * omega.x;
            block.values[1] = scale * omega.y;
            block.values[2] = scale * omega.z;
            block.values[3] = scale * deformation.nodes[node].translation.x;
            block.values[4] = scale * deformation.nodes[node].translation.y;
            block.values[5] = scale * deformation.nodes[node].translation.z;
            if (with_jacobians) {
                // The rotation rows are the exact derivative of log(R) with
                // respect to a left perturbation, i.e. the inverse left
                // Jacobian; only its first-order term is the identity.
                const Mat3 rotation_jacobian = left_jacobian_inverse(omega);
                for (int row = 0; row < 3; ++row) {
                    for (int column = 0; column < 3; ++column) {
                        block.jacobians[0][row][column] =
                            scale * rotation_jacobian.at(row, column);
                    }
                    block.jacobians[0][row + 3][row + 3] = scale;
                }
            }
            blocks.push_back(block);
        }
    }

    // Landmarks: a marked template vertex must reach its target position.
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
                usable_binding(inputs.bindings[vertex],
                               static_cast<int>(deformation.nodes.size()));
            if (binding.count <= 0) {
                continue;
            }
            const Vec3 deformed =
                evaluate_point(inputs.template_points[vertex], binding, deformation);
            add_offset_block(blocks, BlockKind::Landmark, binding, deformation,
                             inputs.template_points[vertex], deformed,
                             landmark.target, scale, with_jacobians);
        }
    }
    return blocks;
}

BlockEnergies block_energies(const std::vector<ResidualBlock>& blocks) {
    BlockEnergies energies;
    for (const ResidualBlock& block : blocks) {
        double energy = 0.0;
        for (int row = 0; row < block.row_count; ++row) {
            energy += block.values[row] * block.values[row];
        }
        switch (block.kind) {
            case BlockKind::PointToPoint:
                energies.point_to_point += energy;
                break;
            case BlockKind::PointToPlane:
                energies.point_to_plane += energy;
                break;
            case BlockKind::Smoothness:
                energies.smoothness += energy;
                break;
            case BlockKind::Anchor:
                energies.anchor += energy;
                break;
            case BlockKind::Landmark:
                energies.landmark += energy;
                break;
        }
    }
    return energies;
}

double total_energy(const std::vector<ResidualBlock>& blocks) {
    return block_energies(blocks).total();
}

double total_energy(
    const EnergyInputs& inputs,
    const Deformation& deformation,
    const EnergyWeights& weights) {
    // The same residuals as build_blocks, without materializing the Jacobians
    // that the accept/reject test would immediately throw away.
    return total_energy(build_blocks(inputs, deformation, weights, false));
}

}  // namespace nricp
