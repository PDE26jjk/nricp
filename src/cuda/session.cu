#include "cuda/gpu_solver.cuh"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include "core/version.h"
#include "cuda/assemble.cuh"
#include "cuda/deform_ops.cuh"
#include "cuda/gpu_level.cuh"
#include "cuda/host_staging.cuh"
#include "cuda/lbvh.cuh"
#include "cuda/level_layout.cuh"
#include "cuda/pcg_solver.cuh"
#include "cuda/solver_common.cuh"
#include "solver/route_shared.h"

// Route B. The shape of the schedule is route A's -- coarse to fine,
// correspondence updates outside, damped Gauss-Newton steps inside -- but the
// state lives on the device for the whole of a level and the host only sees the
// numbers it needs to make a decision:
//
//   * the node transforms are not uploaded or downloaded inside a level. An
//     accepted step is applied by a kernel and a rejected one is undone by
//     copying the accepted state back on the device; the host learns the new
//     transforms at the level boundary, where the next level's warm start needs
//     them anyway.
//   * the deformed template's normals and the target's hierarchy come out of
//     kernels, so an outer iteration has no host-side geometry work in it.
//   * the damped linear solve is a single cooperative kernel. This is what the
//     timeline said to do: a kernel launch costs about 10.5 microseconds on the
//     reference machine (measured with an empty kernel), so the
//     host-orchestrated PCG's roughly 230 launches per solve dominated a run
//     whose own kernels add up to a few hundred milliseconds. The iteration now
//     stays on the device, its inner products are grid-wide reductions, and it
//     decides convergence itself, so a solve costs one launch instead of
//     hundreds.
//
// What stays host-side is the schedule: which energy is lower, whether a level
// has converged, which damping to try next.
namespace nricp {

using gpu::apply_increment;
using gpu::assemble_system;
using gpu::AssembleInputs;
using gpu::build_query_normals;
using gpu::cooperative_block_count;
using gpu::cooperative_launch_supported;
using gpu::download_state;
using gpu::download_vectors;
using gpu::EnergyReadout;
using gpu::evaluate_deformation;
using gpu::GpuLevel;
using gpu::kEnergyKinds;
using gpu::Lbvh;
using gpu::load_level;
using gpu::PcgOutcome;
using gpu::PcgState;
using gpu::pcg_system_of;
using gpu::PinnedFloats;
using gpu::reserve_level;
using gpu::restore_accepted_state;
using gpu::save_accepted_state;
using gpu::solve_linear_system;
using gpu::StreamGuard;
using gpu::summarize_correspondences;
using gpu::update_robust_weights;
using gpu::upload_ints;
using gpu::upload_state;
using gpu::upload_vectors;

// The session's device state and its position in the schedule. Keeping it behind
// a pointer is what lets the header stay free of the CUDA and thrust types, so a
// host translation unit that only calls the API does not pull them in.
struct RouteBSession::State {
    // Copies of the inputs, so the caller's objects do not have to outlive the
    // session.
    Mesh template_mesh;
    TemplateGraph graph;
    RouteSettings settings;
    std::vector<Landmark> landmarks;

    Lbvh hierarchy;
    StreamGuard stream_guard;
    PinnedFloats staging{std::max(kEnergyKinds, 3)};
    thrust::device_vector<float> points;
    thrust::device_vector<int> template_faces;
    thrust::device_vector<unsigned char> mask;
    const unsigned char* mask_pointer = nullptr;
    thrust::device_vector<int> landmark_vertices;
    thrust::device_vector<float> landmark_targets;
    // The landmark rows live inside the gathered pass, which walks vertices, so
    // the markers are also indexed per vertex: a flag saying which vertices are
    // markers at all, and the position each is asked to sit on. Both are fixed
    // for the run; the weight itself comes from the level's schedule, since the
    // per-level schedule anneals it.
    thrust::device_vector<float> landmark_flag;
    thrust::device_vector<float> landmark_target;
    GpuLevel level;  // reserved at the finest level's size and reused

    cudaStream_t stream = nullptr;
    int vertex_count = 0;
    int triangle_count = 0;
    int landmark_count = 0;
    int pcg_max_iterations = 30;
    int grid_blocks = 1;
    float distance_threshold = 0.0f;
    float min_normal_cosine = -1.0f;
    float min_slope = 0.0f;
    PcgState host_pcg;

    // Where the schedule got to. A step runs exactly one phase, so the caller
    // regains control between an outer iteration's correspondences, its linear
    // solves, and the level boundary.
    enum class Phase { BeginLevel, Refresh, InnerStep, CloseOuter, EndLevel, Done };
    Phase phase = Phase::BeginLevel;
    bool finished = false;
    std::size_t level_index = 0;
    int outer = 0;
    int inner = 0;
    double damping = 0.0;
    double current_energy = 0.0;
    double previous_level_energy = 0.0;
    LevelView view;
    RunReport report;
    std::vector<CorrespondenceCounts> counts_per_level;
    CorrespondenceCounts counts;
    Deformation deformation;
    EnergyReadout level_readout;
    bool evaluated = false;
    std::vector<Vec3> cached_vertices;

    LevelSettings level_settings() const {
        return level_settings_for(settings, level_index);
    }

    EnergyReadout accumulate(const LevelSettings& settings_for_level,
                             bool with_jacobians);
    CorrespondenceCounts summarize();
    PcgOutcome solve_system(float damping_value);
    void begin_level();
    void refresh();
    bool inner_step();
    bool close_outer();
    bool end_level();
    // The fitted template in the graph's normalized frame -- what a RouteResult
    // carries, since the caller converts it back to its own units.
    std::vector<Vec3> fitted_vertices();
};

// One accumulation pass: the level's blocks and right-hand side, and the five
// block energies. Everything the pass needs that is not the level itself goes
// in one request, so the schedule reads as the decision it is making.
EnergyReadout RouteBSession::State::accumulate(
    const LevelSettings& settings_for_level, bool with_jacobians) {
    AssembleInputs inputs;
    inputs.template_points = thrust::raw_pointer_cast(points.data());
    inputs.mask = mask_pointer;
    inputs.landmark_flag = thrust::raw_pointer_cast(landmark_flag.data());
    inputs.landmark_target = thrust::raw_pointer_cast(landmark_target.data());
    inputs.landmark_vertices =
        thrust::raw_pointer_cast(landmark_vertices.data());
    inputs.landmark_targets = thrust::raw_pointer_cast(landmark_targets.data());
    inputs.landmark_count = landmark_count;
    inputs.vertex_count = vertex_count;
    inputs.point_to_point =
        static_cast<float>(settings_for_level.weights.point_to_point);
    inputs.point_to_plane =
        static_cast<float>(settings_for_level.weights.point_to_plane);
    inputs.landmark = static_cast<float>(settings_for_level.weights.landmark);
    inputs.smoothness = static_cast<float>(settings_for_level.weights.smoothness);
    inputs.anchor = static_cast<float>(settings_for_level.weights.anchor);
    return assemble_system(level, inputs, with_jacobians, staging, stream);
}

// The correspondence statistics for the level, reduced on the device.
CorrespondenceCounts RouteBSession::State::summarize() {
    return summarize_correspondences(level, vertex_count, stream);
}

// One damped linear solve, handed to the solver module: the preconditioner is
// rebuilt for the current damping and the recurrence runs inside a single
// cooperative kernel, so the host only ever sees the outcome.
PcgOutcome RouteBSession::State::solve_system(float damping_value) {
    return solve_linear_system(pcg_system_of(level), damping_value,
                               settings.diagonal_damping, settings.pcg_tolerance,
                               pcg_max_iterations, grid_blocks, host_pcg, stream);
}

void RouteBSession::State::begin_level() {
    const GraphLevel& graph_level = graph.levels[level_index];
    if (level_index > 0) {
        deformation = transfer_to_level(graph.levels[level_index - 1], deformation,
                                        graph_level);
    }
    load_level(level, graph_level, vertex_count, stream);
    upload_state(deformation, level.rotations, level.translations);
    view = LevelView{};
    view.level = static_cast<int>(level_index);
    view.nodes = level.node_count;
    damping = settings.initial_damping;
    previous_level_energy = 0.0;
    outer = 0;
    inner = 0;
    current_energy = 0.0;
}

// One correspondence update: the deformed template, its normals, the closest
// point on the target for every vertex, the robust weights, and the system the
// linear solves will consume.
void RouteBSession::State::refresh() {
    const LevelSettings settings_for_level = level_settings();
    evaluate_deformation(level, thrust::raw_pointer_cast(points.data()),
                         vertex_count, stream);
    build_query_normals(level, template_faces, triangle_count, vertex_count,
                        stream);

    hierarchy.query(
        thrust::raw_pointer_cast(level.deformed.data()),
        thrust::raw_pointer_cast(level.query_normals.data()), vertex_count,
        mask_pointer, distance_threshold, min_normal_cosine, min_slope,
        thrust::raw_pointer_cast(level.correspondence_points.data()),
        thrust::raw_pointer_cast(level.correspondence_normals.data()),
        thrust::raw_pointer_cast(level.correspondence_distances.data()),
        thrust::raw_pointer_cast(level.correspondence_triangles.data()),
        thrust::raw_pointer_cast(level.rejection.data()), stream);

    update_robust_weights(level, vertex_count,
                          static_cast<float>(settings_for_level.robust_threshold),
                          stream);

    current_energy = accumulate(settings_for_level, true).total;
    inner = 0;
    if (outer == 0) {
        view.energy_start = current_energy;
    }
}

// One damped Newton step: the solve, the step, the energy, and the decision.
// Returns false when the inner loop for this correspondence set is over.
// One damped Newton step: the solve, the step, the energy, and the decision.
// Returns false when the inner loop for this correspondence set is over.
bool RouteBSession::State::inner_step() {
    const LevelSettings settings_for_level = level_settings();
    const int attempt_limit = std::max(1, settings.max_damping_attempts);
    double candidate_energy = current_energy;
    bool accepted = false;
    for (int attempt = 0; attempt < attempt_limit && !accepted; ++attempt) {
        const PcgOutcome outcome = solve_system(static_cast<float>(damping));
        if (!outcome.ok) {
            damping *= settings.damping_increase;
            continue;
        }
        // Keep the accepted state, step, measure, and only then decide: the whole
        // sequence runs on the device.
        save_accepted_state(level, level.node_count, stream);
        apply_increment(level, level.node_count, stream);
        candidate_energy = accumulate(settings_for_level, false).total;
        if (candidate_energy < current_energy) {
            accepted = true;
            break;
        }
        // Rejected: the accepted state goes back and the next attempt uses a
        // stronger damping.
        restore_accepted_state(level, level.node_count, stream);
        damping *= settings.damping_increase;
    }

    if (!accepted) {
        return false;
    }
    const double improvement = relative_change(current_energy, candidate_energy);
    damping = std::max(damping * settings.damping_decrease, 1e-12);
    inner += 1;
    view.inner_iterations += 1;
    if (improvement <= settings_for_level.inner_tolerance) {
        return false;
    }
    // The accepted state is on the device; assemble the system for the next inner
    // iteration from it.
    current_energy = accumulate(settings_for_level, true).total;
    return true;
}

// Returns true when the level is over.
bool RouteBSession::State::close_outer() {
    const LevelSettings settings_for_level = level_settings();
    level_readout = accumulate(settings_for_level, false);
    const double level_energy = level_readout.total;
    view.energy_end = level_energy;
    bool converged = false;
    view.outer_iterations += 1;
    if (outer > 0 &&
        relative_change(previous_level_energy, level_energy) <=
            settings_for_level.outer_tolerance) {
        view.converged = true;
        converged = true;
    }
    previous_level_energy = level_energy;
    outer += 1;
    if (converged) {
        return true;
    }
    return outer >= std::max(1, settings_for_level.max_outer_iterations);
}

// Returns true when that was the last level.
bool RouteBSession::State::end_level() {
    const LevelSettings settings_for_level = level_settings();
    counts = summarize();
    view.correspondence_mean = counts.distance_mean;
    counts_per_level.push_back(counts);
    report.levels.push_back(view);
    record_level_settings(report, level_index, settings_for_level.weights,
                          settings_for_level.robust_threshold);
    // The level is over: the transforms come back to the host, for the warm start
    // of the next level and for the caller.
    deformation = download_state(level.rotations, level.translations,
                                 graph.levels[level_index].node_rest_positions);
    if (level_index + 1 == graph.levels.size()) {
        // Evaluate once more so the cached vertices belong to the final accepted
        // state rather than to the start of the last outer pass.
        evaluate_deformation(level, thrust::raw_pointer_cast(points.data()),
                             vertex_count, stream);

        cached_vertices = download_vectors(level.deformed);
        evaluated = true;
        report.energy.point_to_point = level_readout.point_to_point;
        report.energy.point_to_plane = level_readout.point_to_plane;
        report.energy.smoothness = level_readout.smoothness;
        report.energy.anchor = level_readout.anchor;
        report.energy.landmark = level_readout.landmark;
        report.correspondence.total = counts.total;
        report.correspondence.accepted = counts.accepted;
        report.correspondence.rejected_distance = counts.rejected_distance;
        report.correspondence.distance_mean = counts.distance_mean;
        report.correspondence.distance_p95 = counts.distance_p95;
        report.correspondence.distance_max = counts.distance_max;
        return true;
    }
    ++level_index;
    return false;
}

RouteBSession::RouteBSession(
    const Mesh& normalized_template,
    const Mesh& normalized_target,
    const TemplateGraph& graph,
    const RouteSettings& settings,
    const VertexMask* mask,
    const std::vector<Landmark>& landmarks)
    : state_(std::make_unique<State>()) {
    State& session = *state_;
    session.report.route = "route_b";
    session.report.precision = "float";
    session.report.version = NRICP_VERSION_STRING;
    session.template_mesh = normalized_template;
    session.graph = graph;
    session.settings = settings;
    session.landmarks = landmarks;
    session.vertex_count = static_cast<int>(normalized_template.vertices.size());
    session.triangle_count = static_cast<int>(normalized_template.faces.size());
    if (session.graph.levels.empty() || session.vertex_count <= 0 ||
        session.triangle_count <= 0) {
        session.finished = true;
        session.phase = State::Phase::Done;
        return;
    }
    if (!cooperative_launch_supported()) {
        throw std::runtime_error(
            "route B needs a device that supports cooperative launches (the linear "
            "solve is a persistent kernel); this device does not report the "
            "attribute");
    }
    session.stream = session.stream_guard.get();
    session.pcg_max_iterations = std::max(1, settings.pcg_max_iterations);
    session.grid_blocks = cooperative_block_count();
    session.distance_threshold =
        static_cast<float>(settings.correspondence.distance_threshold);
    session.min_normal_cosine =
        static_cast<float>(settings.correspondence.min_normal_cosine);
    session.min_slope = static_cast<float>(settings.correspondence.min_slope);
    session.deformation =
        identity_deformation(session.graph.levels.front().node_rest_positions);

    // The target's hierarchy, built on the device and never read back. The
    // accelerated route takes the target as a mesh rather than as the core
    // layer's query structure, because it builds its own: asking for one that
    // this route never reads would be a whole host-side BVH per call.
    session.hierarchy.build(normalized_target, session.stream);
    if (session.hierarchy.empty()) {
        session.finished = true;
        session.phase = State::Phase::Done;
        return;
    }

    // The template's fixed inputs, uploaded once.
    session.points = upload_vectors(normalized_template.vertices);
    std::vector<int> template_faces(
        static_cast<std::size_t>(session.triangle_count) * 3);
    for (int triangle = 0; triangle < session.triangle_count; ++triangle) {
        const Int3& face =
            normalized_template.faces[static_cast<std::size_t>(triangle)];
        template_faces[triangle * 3 + 0] = face.v0;
        template_faces[triangle * 3 + 1] = face.v1;
        template_faces[triangle * 3 + 2] = face.v2;
    }
    session.template_faces = upload_ints(template_faces);
    if (mask != nullptr && !mask->active.empty()) {
        session.mask.assign(mask->active.begin(), mask->active.end());
        session.mask_pointer = thrust::raw_pointer_cast(session.mask.data());
    }
    std::vector<int> landmark_vertices;
    std::vector<Vec3> landmark_targets;
    for (const Landmark& landmark : landmarks) {
        landmark_vertices.push_back(landmark.template_vertex);
        landmark_targets.push_back(landmark.target);
    }
    session.landmark_vertices = upload_ints(landmark_vertices);
    session.landmark_targets = upload_vectors(landmark_targets);
    session.landmark_count = static_cast<int>(landmark_vertices.size());
    std::vector<float> landmark_weight(static_cast<std::size_t>(session.vertex_count),
                                       0.0f);
    std::vector<float> landmark_target(static_cast<std::size_t>(session.vertex_count) * 3,
                                       0.0f);
    for (std::size_t index = 0; index < landmark_vertices.size(); ++index) {
        const int vertex = landmark_vertices[index];
        if (vertex < 0 || vertex >= session.vertex_count) {
            continue;
        }
        landmark_weight[static_cast<std::size_t>(vertex)] = 1.0f;
        landmark_target[static_cast<std::size_t>(vertex) * 3 + 0] =
            static_cast<float>(landmark_targets[index].x);
        landmark_target[static_cast<std::size_t>(vertex) * 3 + 1] =
            static_cast<float>(landmark_targets[index].y);
        landmark_target[static_cast<std::size_t>(vertex) * 3 + 2] =
            static_cast<float>(landmark_targets[index].z);
    }
    session.landmark_flag.assign(landmark_weight.begin(), landmark_weight.end());
    session.landmark_target.assign(landmark_target.begin(), landmark_target.end());

    // Every buffer at the finest level's size: from here on a level only assigns
    // into storage that already exists, so neither a level change nor a step
    // reallocates.
    const GraphLevel& finest = session.graph.levels.back();
    reserve_level(session.level, finest.node_count(),
                  static_cast<int>(finest.edges.size()), session.vertex_count,
                  session.pcg_max_iterations);
}

RouteBSession::~RouteBSession() = default;
RouteBSession::RouteBSession(RouteBSession&&) noexcept = default;
RouteBSession& RouteBSession::operator=(RouteBSession&&) noexcept = default;

WrapProgress RouteBSession::step() {
    State& session = *state_;
    if (!session.finished) {
        switch (session.phase) {
            case State::Phase::BeginLevel:
                session.begin_level();
                session.phase = State::Phase::Refresh;
                break;
            case State::Phase::Refresh:
                session.refresh();
                session.phase = State::Phase::InnerStep;
                break;
            case State::Phase::InnerStep:
                session.phase = session.inner_step() ? State::Phase::InnerStep
                                                     : State::Phase::CloseOuter;
                break;
            case State::Phase::CloseOuter:
                session.phase =
                    session.close_outer() ? State::Phase::EndLevel : State::Phase::Refresh;
                break;
            case State::Phase::EndLevel:
                session.phase =
                    session.end_level() ? State::Phase::Done : State::Phase::BeginLevel;
                break;
            case State::Phase::Done:
                session.finished = true;
                break;
        }
    }
    WrapProgress progress;
    progress.level = static_cast<int>(session.level_index);
    progress.outer = session.outer;
    progress.inner = session.inner;
    progress.finished = session.finished;
    progress.level_converged = session.view.converged;
    progress.energy = session.current_energy;
    progress.correspondence_mean = session.counts.distance_mean;
    return progress;
}

bool RouteBSession::finished() const { return state_->finished; }

std::vector<Vec3> RouteBSession::vertices() const {
    State& session = *state_;
    // The schedule works in the normalized frame the graph was built in; a caller
    // drawing the result wants the units it handed in, which is also what the
    // batch entry point reports.
    Mesh restored;
    restored.vertices = session.fitted_vertices();
    undo_normalization(restored, session.graph.normalization);
    return restored.vertices;
}

// Evaluated on demand: one launch and one copy, so a caller can redraw as often
// as it likes without the session having to keep anything in step.
std::vector<Vec3> RouteBSession::State::fitted_vertices() {
    if (evaluated && !cached_vertices.empty()) {
        return cached_vertices;
    }
    if (vertex_count <= 0) {
        return {};
    }

    evaluate_deformation(level, thrust::raw_pointer_cast(points.data()),
                         vertex_count, stream);

    return download_vectors(level.deformed);
}

const RunReport& RouteBSession::report() const { return state_->report; }

const std::vector<CorrespondenceCounts>& RouteBSession::correspondence_per_level()
    const {
    return state_->counts_per_level;
}

const TemplateGraph& RouteBSession::graph() const { return state_->graph; }

RouteResult RouteBSession::result() const {
    State& session = *state_;
    RouteResult out;
    out.report = session.report;
    out.correspondence_per_level = session.counts_per_level;
    out.deformation = session.deformation;
    if (session.vertex_count > 0) {
        out.vertices = session.fitted_vertices();
    }
    if (!session.evaluated) {
        // The schedule is still running: report the last energy pass and
        // correspondence set it saw, so a partial result is still a result.
        out.report.energy.point_to_point = session.level_readout.point_to_point;
        out.report.energy.point_to_plane = session.level_readout.point_to_plane;
        out.report.energy.smoothness = session.level_readout.smoothness;
        out.report.energy.anchor = session.level_readout.anchor;
        out.report.energy.landmark = session.level_readout.landmark;
        out.report.correspondence.total = session.counts.total;
        out.report.correspondence.accepted = session.counts.accepted;
        out.report.correspondence.rejected_distance =
            session.counts.rejected_distance;
        out.report.correspondence.distance_mean = session.counts.distance_mean;
        out.report.correspondence.distance_p95 = session.counts.distance_p95;
        out.report.correspondence.distance_max = session.counts.distance_max;
    }
    return out;
}

}  // namespace nricp
