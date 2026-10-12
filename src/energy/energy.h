#pragma once

#include <vector>

#include "core/constraints.h"
#include "core/deformation.h"
#include "core/sampling.h"
#include "core/vec3.h"

namespace nricp {

constexpr int kMaxBlockRows = 6;
constexpr int kMaxBlockSlots = 4;

// Per-level weights of the wrapping energy. They are applied inside the
// residual, so a block's energy is the sum of its squared scaled rows and the
// total energy is directly comparable between levels and routes.
struct EnergyWeights {
    double point_to_point = 1.0;  // lambda_c (keeps the plane term rotatable)
    double point_to_plane = 1.0;  // lambda_p (the primary fitting term)
    double smoothness = 1.0;      // lambda_s (graph regularization)
    double anchor = 0.0;          // lambda_a (ties nodes to their rest transform)
    double landmark = 1.0;        // lambda_l (control points)
};

enum class BlockKind : int {
    PointToPoint = 0,
    PointToPlane = 1,
    Smoothness = 2,
    Anchor = 3,
    Landmark = 4,
};

// One residual block: `row_count` rows, each depending on at most four node
// transforms. `nodes[slot]` names the node and `jacobians[slot]` holds the
// derivative of every row with respect to that node's six tangent-space
// unknowns (rotation increment first, then translation).
struct ResidualBlock {
    BlockKind kind = BlockKind::PointToPoint;
    int row_count = 0;
    double values[kMaxBlockRows] = {};
    int slots = 0;
    int nodes[kMaxBlockSlots] = {-1, -1, -1, -1};
    double jacobians[kMaxBlockSlots][kMaxBlockRows][6] = {};
};

// What the data blocks need for one template vertex.
struct DataCorrespondence {
    Vec3 point;                  // closest point on the target surface
    Vec3 normal;                 // target normal at that point
    double robust_weight = 1.0;  // IRLS weight u_i in [0, 1]
    bool accepted = false;       // survived the correspondence filters
};

struct EnergyInputs {
    std::vector<Vec3> template_points;  // rest positions, normalized
    std::vector<VertexBinding> bindings;
    std::vector<Vec3> node_positions;   // graph node rest positions
    std::vector<DataCorrespondence> correspondences;  // one per template vertex
    std::vector<Int2> smoothness_edges;
    std::vector<Landmark> landmarks;
    const VertexMask* mask = nullptr;   // null: every vertex participates
};

struct BlockEnergies {
    double point_to_point = 0.0;
    double point_to_plane = 0.0;
    double smoothness = 0.0;
    double anchor = 0.0;
    double landmark = 0.0;

    void add(BlockKind kind, double energy) {
        switch (kind) {
            case BlockKind::PointToPoint:
                point_to_point += energy;
                break;
            case BlockKind::PointToPlane:
                point_to_plane += energy;
                break;
            case BlockKind::Smoothness:
                smoothness += energy;
                break;
            case BlockKind::Anchor:
                anchor += energy;
                break;
            case BlockKind::Landmark:
                landmark += energy;
                break;
        }
    }

    double total() const {
        return point_to_point + point_to_plane + smoothness + anchor + landmark;
    }
};

// Resolves the block set of the current state: one point-to-point and one
// point-to-plane block per accepted correspondence, one smoothness block per
// graph edge, one anchor block per node, one landmark block per landmark.
// Fills only the residual values when `with_jacobians` is false, which is what
// the accept/reject test needs and what keeps the debug interface usable at
// production scale.
std::vector<ResidualBlock> build_blocks(
    const EnergyInputs& inputs,
    const Deformation& deformation,
    const EnergyWeights& weights,
    bool with_jacobians = true);

// Energy of the current state, without materializing Jacobians.
double total_energy(
    const EnergyInputs& inputs,
    const Deformation& deformation,
    const EnergyWeights& weights);

BlockEnergies block_energies(const std::vector<ResidualBlock>& blocks);
double total_energy(const std::vector<ResidualBlock>& blocks);

}  // namespace nricp
