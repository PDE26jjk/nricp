#include "core/deformation.h"

#include <Eigen/Dense>

#include <algorithm>

#include "core/quaternion.h"
#include "core/rotation.h"

namespace nricp {

Deformation identity_deformation(const std::vector<Vec3>& node_rest_positions) {
    Deformation deformation;
    deformation.node_rest_positions = node_rest_positions;
    deformation.nodes.resize(node_rest_positions.size());
    return deformation;
}

void retract(Deformation& deformation, const std::vector<double>& delta) {
    const std::size_t expected = deformation.nodes.size() * 6;
    if (delta.size() < expected) {
        return;
    }
    for (std::size_t node = 0; node < deformation.nodes.size(); ++node) {
        const double* values = delta.data() + node * 6;
        const Vec3 omega{values[0], values[1], values[2]};
        const Vec3 translation{values[3], values[4], values[5]};
        deformation.nodes[node].rotation =
            exp_so3(omega) * deformation.nodes[node].rotation;
        deformation.nodes[node].translation += translation;
    }
}

std::vector<double> tangent_difference(
    const Deformation& from, const Deformation& to) {
    const std::size_t count = std::min(from.nodes.size(), to.nodes.size());
    std::vector<double> delta(count * 6, 0.0);
    for (std::size_t node = 0; node < count; ++node) {
        // Left increment: exp([w]) R_from = R_to  =>  w = log(R_to R_from^T).
        const Vec3 omega = log_so3(to.nodes[node].rotation *
                                   transposed(from.nodes[node].rotation));
        delta[node * 6 + 0] = omega.x;
        delta[node * 6 + 1] = omega.y;
        delta[node * 6 + 2] = omega.z;
        const Vec3 translation =
            to.nodes[node].translation - from.nodes[node].translation;
        delta[node * 6 + 3] = translation.x;
        delta[node * 6 + 4] = translation.y;
        delta[node * 6 + 5] = translation.z;
    }
    return delta;
}

Vec3 evaluate_point(
    const Vec3& point,
    const VertexBinding& binding,
    const Deformation& deformation) {
    Vec3 result;
    for (int slot = 0; slot < binding.count; ++slot) {
        const int node = binding.nodes[slot];
        if (node < 0 || node >= static_cast<int>(deformation.nodes.size())) {
            continue;
        }
        result += deformation.apply(node, point) * binding.weights[slot];
    }
    return result;
}

std::vector<Vec3> evaluate(
    const std::vector<Vec3>& vertices,
    const std::vector<VertexBinding>& bindings,
    const Deformation& deformation) {
    // A short binding list would read out of bounds, so the loop is bounded by
    // both; the callers that take this from Python validate the pairing first.
    const std::size_t count = std::min(vertices.size(), bindings.size());
    std::vector<Vec3> result(vertices.size());
    for (std::size_t i = 0; i < count; ++i) {
        result[i] = evaluate_point(vertices[i], bindings[i], deformation);
    }
    for (std::size_t i = count; i < vertices.size(); ++i) {
        result[i] = vertices[i];
    }
    return result;
}

RigidTransform fit_rigid(
    const std::vector<Vec3>& source_points,
    const std::vector<Vec3>& target_points,
    const std::vector<double>& weights) {
    RigidTransform transform;
    const std::size_t count =
        std::min(source_points.size(), target_points.size());
    if (count == 0 || weights.size() < count) {
        return transform;
    }

    double weight_sum = 0.0;
    Vec3 source_center;
    Vec3 target_center;
    for (std::size_t i = 0; i < count; ++i) {
        const double weight = std::max(0.0, weights[i]);
        weight_sum += weight;
        source_center += source_points[i] * weight;
        target_center += target_points[i] * weight;
    }
    if (weight_sum <= 0.0) {
        return transform;
    }
    source_center = source_center / weight_sum;
    target_center = target_center / weight_sum;

    Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
    for (std::size_t i = 0; i < count; ++i) {
        const double weight = std::max(0.0, weights[i]);
        if (weight == 0.0) {
            continue;
        }
        const Vec3 p = source_points[i] - source_center;
        const Vec3 q = target_points[i] - target_center;
        const Eigen::Vector3d pe(p.x, p.y, p.z);
        const Eigen::Vector3d qe(q.x, q.y, q.z);
        covariance += weight * pe * qe.transpose();
    }

    const Eigen::JacobiSVD<Eigen::Matrix3d> svd(
        covariance, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d correction = Eigen::Matrix3d::Identity();
    correction(2, 2) =
        (svd.matrixV() * svd.matrixU().transpose()).determinant() < 0.0 ? -1.0 : 1.0;
    const Eigen::Matrix3d rotation =
        svd.matrixV() * correction * svd.matrixU().transpose();
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            transform.rotation.at(row, col) = rotation(row, col);
        }
    }
    transform.translation = target_center - transform.rotation * source_center;
    return transform;
}

Deformation transfer_level(
    const std::vector<Vec3>& from_nodes,
    const Deformation& from_deformation,
    const std::vector<Vec3>& to_nodes,
    int max_nodes) {
    Deformation transferred = identity_deformation(to_nodes);
    if (from_nodes.empty() || to_nodes.empty()) {
        return transferred;
    }

    // A new node samples the field through the closest source nodes: it takes
    // the weighted blend of their rotations (quaternion space, with every sign
    // aligned to the heaviest contributor) and the field's own value at its rest
    // position. The binding weights deliberately zero out the farthest support
    // node, which would leave a rigid fit with too few points to determine a
    // rotation; a blend stays well defined and reproduces a rigid field exactly.
    const std::vector<VertexBinding> sampling_bindings =
        bind_vertices(to_nodes, from_nodes, max_nodes);
    for (std::size_t node = 0; node < to_nodes.size(); ++node) {
        const VertexBinding& binding = sampling_bindings[node];
        if (binding.count <= 0) {
            continue;
        }
        int heaviest = binding.nodes[0];
        double heaviest_weight = binding.weights[0];
        for (int slot = 1; slot < binding.count; ++slot) {
            if (binding.weights[slot] > heaviest_weight) {
                heaviest_weight = binding.weights[slot];
                heaviest = binding.nodes[slot];
            }
        }
        const Quaternion reference =
            quaternion_from_matrix(from_deformation.nodes[heaviest].rotation);

        Quaternion blended{0.0, 0.0, 0.0, 0.0};
        Vec3 position;
        for (int slot = 0; slot < binding.count; ++slot) {
            const int source = binding.nodes[slot];
            const double weight = binding.weights[slot];
            Quaternion rotation =
                quaternion_from_matrix(from_deformation.nodes[source].rotation);
            if (dot(rotation, reference) < 0.0) {
                rotation = rotation * -1.0;
            }
            blended += rotation * weight;
            position += from_deformation.apply(source, to_nodes[node]) * weight;
        }

        // When the neighbours' rotations are spread across both quaternion
        // hemispheres the weighted sum is near zero, and normalizing that would
        // silently produce the identity rotation -- a half-turn error for that
        // node, with no warning. Fall back to the heaviest neighbour instead.
        const double blended_length = std::sqrt(dot(blended, blended));
        const Mat3 rotation =
            blended_length > 1e-8 ? matrix_from_quaternion(normalized(blended))
                                  : matrix_from_quaternion(reference);
        transferred.nodes[node].rotation = rotation;
        // The transform has to map the new node's own rest position onto the
        // sampled position: R (g - g) + g + t = g + t = position.
        transferred.nodes[node].translation = position - to_nodes[node];
    }
    return transferred;
}

}  // namespace nricp
