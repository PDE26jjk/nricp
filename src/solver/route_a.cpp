#include "solver/route_a.h"

#include <algorithm>
#include <cmath>
#include <string>

#include "solver/direct_solve.h"
#include "solver/fused_assembly.h"
#include "solver/normal_equations.h"
#include "solver/route_shared.h"
#include "core/version.h"

namespace nricp {
namespace {

// IRLS weights from the current mismatch between a fitted vertex and its
// correspondence. With reweighting disabled every accepted correspondence keeps
// weight one, which leaves plain least squares available for comparison.
std::vector<DataCorrespondence> build_correspondences(
    const std::vector<Vec3>& deformed,
    const CorrespondenceSet& found,
    const LevelSettings& settings) {
    std::vector<DataCorrespondence> correspondences(deformed.size());
    for (std::size_t i = 0; i < deformed.size(); ++i) {
        const Correspondence& item = found.items[i];
        if (!item.accepted()) {
            continue;
        }
        const double squared = squared_norm(deformed[i] - item.point);
        const double weight =
            settings.robust_threshold > 0.0
                ? robust_weight(settings.robust_kernel, squared,
                                settings.robust_threshold)
                : 1.0;
        correspondences[i] = {item.point, item.normal, weight, true};
    }
    return correspondences;
}

void fill_final_statistics(
    RouteResult& result,
    const Mesh& normalized_template,
    const GraphLevel& finest,
    const Deformation& deformation,
    const TargetQueries& target,
    const RouteSettings& settings,
    const VertexMask* mask) {
    const std::vector<Vec3> deformed =
        evaluate(normalized_template.vertices, finest.bindings, deformation);
    Mesh deformed_mesh{deformed, normalized_template.faces};
    const CorrespondenceSet found = find_correspondences(
        target, deformed, vertex_normals(deformed_mesh), settings.correspondence,
        mask);
    result.report.correspondence.total = found.counts.total;
    result.report.correspondence.accepted = found.counts.accepted;
    result.report.correspondence.rejected_distance = found.counts.rejected_distance;
    result.report.correspondence.rejected_normal = found.counts.rejected_normal;
    result.report.correspondence.rejected_slope = found.counts.rejected_slope;
    result.report.correspondence.rejected_masked = found.counts.rejected_masked;
    result.report.correspondence.distance_mean = found.counts.distance_mean;
    result.report.correspondence.distance_p95 = found.counts.distance_p95;
    result.report.correspondence.distance_max = found.counts.distance_max;
}

}  // namespace

RouteResult solve_route_a(
    const Mesh& normalized_template,
    const TargetQueries& target,
    const TemplateGraph& graph,
    const RouteSettings& settings,
    const VertexMask* mask,
    const std::vector<Landmark>& landmarks) {
    RouteResult result;
    result.report.route = "route_a";
    result.report.precision = "double";
    result.report.version = NRICP_VERSION_STRING;
    if (graph.levels.empty()) {
        return result;
    }

    Deformation deformation =
        identity_deformation(graph.levels.front().node_rest_positions);

    EnergyInputs inputs;
    inputs.template_points = normalized_template.vertices;
    inputs.landmarks = landmarks;
    inputs.mask = mask;

    for (std::size_t level_index = 0; level_index < graph.levels.size();
         ++level_index) {
        const GraphLevel& level = graph.levels[level_index];
        if (level_index > 0) {
            deformation = transfer_to_level(graph.levels[level_index - 1],
                                            deformation, level);
        }
        const LevelSettings level_settings =
            level_settings_for(settings, level_index);
        inputs.bindings = level.bindings;
        inputs.node_positions = level.node_rest_positions;
        inputs.smoothness_edges = level.edges;

        LevelView view;
        view.level = static_cast<int>(level_index);
        view.nodes = level.node_count();

        double damping = settings.initial_damping;
        CorrespondenceSet found;
        double previous_level_energy = 0.0;

        const int outer_limit = std::max(1, level_settings.max_outer_iterations);
        const int inner_limit = std::max(1, level_settings.max_inner_iterations);
        const int attempt_limit = std::max(1, settings.max_damping_attempts);

        for (int outer = 0; outer < outer_limit; ++outer) {
            const std::vector<Vec3> deformed =
                evaluate(normalized_template.vertices, level.bindings, deformation);
            Mesh deformed_mesh{deformed, normalized_template.faces};
            found = find_correspondences(
                target, deformed, vertex_normals(deformed_mesh),
                settings.correspondence, mask);
            inputs.correspondences =
                build_correspondences(deformed, found, level_settings);

            const double outer_start =
                accumulate_energy(inputs, deformation, level_settings.weights).total();
            if (outer == 0) {
                view.energy_start = outer_start;
            }

            for (int inner = 0; inner < inner_limit; ++inner) {
                // One fused pass gives the system and the energy together, so the
                // inner loop no longer materializes the residual blocks at all.
                const FusedSystem fused =
                    assemble_and_energy(inputs, deformation, level_settings.weights);
                const double current_energy = fused.energies.total();
                const NormalEquations& system = fused.system;

                bool accepted = false;
                double candidate_energy = current_energy;
                Deformation candidate = deformation;
                for (int attempt = 0; attempt < attempt_limit; ++attempt) {
                    const DirectSolveResult solved = solve_direct(
                        damped(system, damping, settings.diagonal_damping));
                    if (!solved.ok) {
                        damping *= settings.damping_increase;
                        continue;
                    }
                    candidate = deformation;
                    retract(candidate, solved.delta);
                    candidate_energy =
                        accumulate_energy(inputs, candidate, level_settings.weights)
                            .total();
                    if (candidate_energy < current_energy) {
                        accepted = true;
                        break;
                    }
                    damping *= settings.damping_increase;
                }
                if (!accepted) {
                    break;  // no damping level improves the energy any further
                }
                deformation = candidate;
                damping = std::max(damping * settings.damping_decrease, 1e-12);
                view.inner_iterations += 1;
                if (relative_change(current_energy, candidate_energy) <=
                    level_settings.inner_tolerance) {
                    break;
                }
            }

            const double level_energy =
                accumulate_energy(inputs, deformation, level_settings.weights).total();
            view.energy_end = level_energy;
            view.correspondence_mean = found.counts.distance_mean;
            view.outer_iterations += 1;
            if (outer > 0 &&
                relative_change(previous_level_energy, level_energy) <=
                    level_settings.outer_tolerance) {
                view.converged = true;
                break;
            }
            previous_level_energy = level_energy;
        }

        result.correspondence_per_level.push_back(found.counts);
        result.report.levels.push_back(view);
        record_level_settings(result.report, level_index, level_settings.weights,
                              level_settings.robust_threshold);
    }

    const GraphLevel& finest = graph.levels.back();
    result.vertices =
        evaluate(normalized_template.vertices, finest.bindings, deformation);
    result.deformation = deformation;

    fill_final_statistics(result, normalized_template, finest, deformation, target,
                          settings, mask);
    const BlockEnergies energies =
        block_energies(build_blocks(inputs, deformation,
                                    level_settings_for(settings, graph.levels.size() - 1)
                                        .weights));
    result.report.energy.point_to_point = energies.point_to_point;
    result.report.energy.point_to_plane = energies.point_to_plane;
    result.report.energy.smoothness = energies.smoothness;
    result.report.energy.anchor = energies.anchor;
    result.report.energy.landmark = energies.landmark;
    return result;
}

}  // namespace nricp
