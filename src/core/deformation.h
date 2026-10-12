#pragma once

#include <vector>

#include "core/mat3.h"
#include "core/sampling.h"
#include "core/vec3.h"

namespace nricp {

// One node's rigid transform, as the specification writes it: a point is mapped
// by p -> R (p - g) + g + t, with g the node's rest position. The translation is
// therefore the displacement of the node's own rest position, so rotating a node
// never moves g itself. That is what makes the smoothness term's derivative with
// respect to a node's own rotation vanish, and it keeps the regularizer from
// fighting the data term.
struct NodeTransform {
    Mat3 rotation;
    Vec3 translation;
};

// Plain rigid transform on point pairs, used by the geometric fits. It carries
// no notion of a node, so it is independent of how a deformation stores its
// nodes.
struct RigidTransform {
    Mat3 rotation;
    Vec3 translation;

    Vec3 apply(const Vec3& point) const { return rotation * point + translation; }
};

// A deformation of the graph: one transform per node plus the node rest
// positions the transforms are expressed against.
struct Deformation {
    std::vector<Vec3> node_rest_positions;
    std::vector<NodeTransform> nodes;

    // The field of one node at an arbitrary point.
    Vec3 apply(int node, const Vec3& point) const {
        const Vec3& rest = node_rest_positions[static_cast<std::size_t>(node)];
        const NodeTransform& transform = nodes[static_cast<std::size_t>(node)];
        return transform.rotation * (point - rest) + rest + transform.translation;
    }
};

Deformation identity_deformation(const std::vector<Vec3>& node_rest_positions);

// Applies a tangent-space increment: six values per node, the rotation
// increment first, then the translation increment. The rotation is applied on
// the left, R_j <- exp([w_j]) R_j, which is the convention the Jacobians use.
void retract(Deformation& deformation, const std::vector<double>& delta);

// Tangent-space difference between two deformations, in the same layout.
std::vector<double> tangent_difference(
    const Deformation& from, const Deformation& to);

// v'_i = sum_j w_ij (R_j (v_i - g_j) + g_j + t_j)
Vec3 evaluate_point(
    const Vec3& point,
    const VertexBinding& binding,
    const Deformation& deformation);

std::vector<Vec3> evaluate(
    const std::vector<Vec3>& vertices,
    const std::vector<VertexBinding>& bindings,
    const Deformation& deformation);

// Weighted rigid fit (reflection excluded): the transform minimizing
// sum_i w_i |R p_i + t - q_i|^2. Weights must be non-negative; zero-weight
// points are ignored. Degenerate supports fall back to translation only.
RigidTransform fit_rigid(
    const std::vector<Vec3>& source_points,
    const std::vector<Vec3>& target_points,
    const std::vector<double>& weights);

// Samples a deformation at the nodes of a finer level: each new node takes the
// blended rotation of the source nodes near it plus the field's value at its
// rest position. This is how a level is warm-started from the previous one.
Deformation transfer_level(
    const std::vector<Vec3>& from_nodes,
    const Deformation& from_deformation,
    const std::vector<Vec3>& to_nodes,
    int max_nodes = VertexBinding::kMaxNodes);

}  // namespace nricp
