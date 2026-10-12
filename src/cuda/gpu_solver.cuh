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

// The two routes the device path runs. They share the schedule, the energy, the
// correspondence rules, the reports and the linear solver; they differ in how a
// node's six values are obtained, and that is all.
enum class DeviceRoute {
    // Damped Gauss-Newton: one linearized step for each node's rotation and
    // translation together, with the 6x6 blocks of the reduced system.
    GaussNewton,
    // Global/local: every rotation is replaced by a closed-form fit, and the
    // solve that follows is translation-only.
    LocalGlobal,
};

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

// Route C: the same schedule, correspondence rules and energies, solved by the
// global/local alternation instead of a simultaneous Newton step. The rotations
// are fitted in closed form per node (Horn's quaternion method) and the linear
// solve that follows is translation-only, which is what makes its matrix
// independent of the rotations and therefore assembled once per correspondence
// update rather than once per solve.
RouteResult solve_route_c(
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
//
// The class is named for the route it was written for and takes the route as an
// argument now that there are two: `DeviceRoute::GaussNewton` is route B,
// `DeviceRoute::LocalGlobal` is route C. Both run the same schedule and report
// the same things, so a caller that drives the loop by hand -- which is the
// point of the class -- does not care which one it constructed.
class RouteBSession {
public:
    RouteBSession(
        const Mesh& normalized_template,
        const Mesh& normalized_target,
        const TemplateGraph& graph,
        const RouteSettings& settings,
        const VertexMask* mask,
        const std::vector<Landmark>& landmarks,
        DeviceRoute route = DeviceRoute::GaussNewton);
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
