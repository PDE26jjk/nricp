#pragma once

#include <string>
#include <utility>
#include <vector>

namespace nricp {

// What a run reports. The report is plain data on purpose: it is what makes an
// acceptance run automatable, and it replaces looking at the result, which is
// how the interactive tool is used.
struct BlockEnergyView {
    double point_to_point = 0.0;
    double point_to_plane = 0.0;
    double smoothness = 0.0;
    double anchor = 0.0;
    double landmark = 0.0;

    double total() const {
        return point_to_point + point_to_plane + smoothness + anchor + landmark;
    }
};

struct CorrespondenceView {
    int total = 0;
    int accepted = 0;
    int rejected_distance = 0;
    int rejected_normal = 0;
    int rejected_slope = 0;
    int rejected_masked = 0;
    double distance_mean = 0.0;
    double distance_p95 = 0.0;
    double distance_max = 0.0;
};

// One level of the coarse-to-fine schedule.
struct LevelView {
    int level = 0;
    int nodes = 0;
    int outer_iterations = 0;
    int inner_iterations = 0;
    double energy_start = 0.0;
    double energy_end = 0.0;
    double correspondence_mean = 0.0;
    bool converged = false;
};

struct RunReport {
    std::string route = "reference";
    std::string precision = "double";
    std::string version;
    double setup_seconds = 0.0;
    double solve_seconds = 0.0;
    BlockEnergyView energy;
    CorrespondenceView correspondence;
    std::vector<LevelView> levels;
    // Effective settings, in the order the caller supplied them.
    std::vector<std::pair<std::string, double>> settings;
};

// Serializes the report as JSON. Numbers keep full double precision so a stored
// result can be reproduced; unknown/NaN values are written as `null`.
std::string to_json(const RunReport& report);

}  // namespace nricp
