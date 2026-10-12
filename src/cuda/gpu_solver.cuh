#pragma once

#include <cuda_runtime.h>

#include <memory>
#include <string>
#include <vector>

#include "core/graph.h"
#include "core/mesh.h"
#include "geometry/closest_point.h"
#include "solver/route_a.h"

namespace nricp {

// Route B: the same energy, correspondence rules and coarse-to-fine schedule as
// route A, with the device doing everything that is not a decision -- the
// closest-point hierarchy, deformation evaluation, the deformed template's
// normals, the fused accumulation of the reduced system, and a damped linear
// solve that runs as one cooperative kernel. The host keeps only the schedule:
// which level, which weights, damping, acceptance.
RouteResult solve_route_b(
    const Mesh& normalized_template,
    const Mesh& normalized_target,
    const TemplateGraph& graph,
    const RouteSettings& settings,
    const VertexMask* mask,
    const std::vector<Landmark>& landmarks);

bool device_available();
std::string device_description();

// One call's worth of progress, for a caller driving the schedule itself.
struct WrapProgress {
    int level = 0;      // which level of the schedule is running
    int outer = 0;      // outer iterations (correspondence updates) done in it
    int inner = 0;      // inner iterations (linear solves) done in the current one
    bool finished = false;
    bool level_converged = false;
    double energy = 0.0;
    double correspondence_mean = 0.0;
};

// The same schedule as solve_route_b, driven by the caller one step at a time.
// qmyidp's cloth solver has the same shape -- initialize, then step until the
// caller is satisfied -- and this exists for the same reason: a caller that can
// look at the fit between steps can see the deformation progress and can stop
// early, before the solve commits to a result that no longer makes sense,
// instead of waiting out the whole coarse-to-fine schedule.
//
// Everything the schedule needs on the device is allocated once, in the
// constructor, at the finest level's size, and reused by every level and every
// step, so driving the schedule from a loop does not reallocate.
class RouteBSession {
public:
    RouteBSession(
        const Mesh& normalized_template,
        const Mesh& normalized_target,
        const TemplateGraph& graph,
        const RouteSettings& settings,
        const VertexMask* mask,
        const std::vector<Landmark>& landmarks);
    ~RouteBSession();
    RouteBSession(RouteBSession&&) noexcept;
    RouteBSession& operator=(RouteBSession&&) noexcept;
    RouteBSession(const RouteBSession&) = delete;
    RouteBSession& operator=(const RouteBSession&) = delete;

    // Advances the schedule by one unit of work and reports where it got to.
    WrapProgress step();

    bool finished() const;

    // The template as it currently stands, evaluated and brought back: one
    // launch and one copy, so it is cheap enough to call between steps.
    std::vector<Vec3> vertices() const;

    // What has run so far; the level in progress is its last entry.
    const RunReport& report() const;
    const std::vector<CorrespondenceCounts>& correspondence_per_level() const;

    // The same result the batch entry point returns, for the state reached so
    // far -- calling it before the schedule finishes gives the partial fit.
    RouteResult result() const;

    // The graph the session is running, for callers that report the fit in the
    // caller's units (it carries the normalization the template was placed in).
    const TemplateGraph& graph() const;

private:
    struct State;
    std::unique_ptr<State> state_;
};

}  // namespace nricp
