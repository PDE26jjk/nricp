#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>

#include "core/report.h"
#include "energy/energy.h"
#include "solver/route_a.h"

namespace nricp {

// Bookkeeping every route shares, so the same fields are recorded under the
// same names whatever solved the level.
inline LevelSettings level_settings_for(
    const RouteSettings& settings, std::size_t level) {
    if (settings.levels.empty()) {
        return LevelSettings{};
    }
    return settings.levels[std::min(level, settings.levels.size() - 1)];
}

inline double relative_change(double before, double after) {
    return (before - after) / std::max(std::abs(before), 1e-12);
}

inline void record_level_settings(
    RunReport& report, std::size_t level, const EnergyWeights& weights,
    double robust_threshold) {
    const std::string prefix = "level" + std::to_string(level) + "_";
    report.settings.emplace_back(prefix + "point_to_point", weights.point_to_point);
    report.settings.emplace_back(prefix + "point_to_plane", weights.point_to_plane);
    report.settings.emplace_back(prefix + "smoothness", weights.smoothness);
    report.settings.emplace_back(prefix + "anchor", weights.anchor);
    report.settings.emplace_back(prefix + "landmark", weights.landmark);
    report.settings.emplace_back(prefix + "robust_threshold", robust_threshold);
}

}  // namespace nricp
