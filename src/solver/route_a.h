#pragma once

#include <vector>

#include "core/constraints.h"
#include "core/graph.h"
#include "core/mesh.h"
#include "core/report.h"
#include "energy/energy.h"
#include "energy/robust.h"
#include "geometry/closest_point.h"
#include "solver/correspondence.h"

namespace nricp {

// One level of the coarse-to-fine schedule.
struct LevelSettings {
    EnergyWeights weights;
    RobustKernel robust_kernel = RobustKernel::Huber;
    double robust_threshold = 0.0;  // 0 disables reweighting
    int max_outer_iterations = 10;  // correspondence updates
    int max_inner_iterations = 10;  // linear solves per correspondence set
    double outer_tolerance = 1e-3;  // relative energy change that stops a level
    double inner_tolerance = 1e-4;  // relative change that stops the inner loop
};

struct RouteSettings {
    // One entry per graph level; the last entry is reused when the graph has
    // more levels than entries.
    std::vector<LevelSettings> levels;
    CorrespondenceSettings correspondence;
    double initial_damping = 1e-3;
    double damping_decrease = 0.3;
    double damping_increase = 4.0;
    int max_damping_attempts = 6;
    bool diagonal_damping = true;
    // Only the accelerated path uses these: the relative residual the PCG drives
    // to, and how many iterations it may take. A tolerance far below
    // single-precision noise buys nothing and costs a kernel launch per
    // iteration (measured: 21k PCG iterations over one benchmark pair).
    float pcg_tolerance = 1e-5f;
    int pcg_max_iterations = 60;
};

struct RouteResult {
    std::vector<Vec3> vertices;  // fitted template in the graph's normalized space
    Deformation deformation;
    std::vector<CorrespondenceCounts> correspondence_per_level;
    RunReport report;
};

// Route A: Gauss-Newton on the wrapping energy with a sparse direct solve in
// double precision. The template must already sit in the target's frame (that is
// what the rigid initialization is for), and both must share the normalization
// the graph was built with.
RouteResult solve_route_a(
    const Mesh& normalized_template,
    const TargetQueries& target,
    const TemplateGraph& graph,
    const RouteSettings& settings,
    const VertexMask* mask,
    const std::vector<Landmark>& landmarks);

}  // namespace nricp
