#include "core/report.h"

#include <cmath>
#include <iomanip>
#include <sstream>

namespace nricp {
namespace {

std::string escape(const std::string& text) {
    std::string out;
    out.reserve(text.size());
    for (const char character : text) {
        switch (character) {
            case '"':
                out += "\\\"";
                break;
            case '\\':
                out += "\\\\";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\t':
                out += "\\t";
                break;
            default:
                out += character;
                break;
        }
    }
    return out;
}

std::string number(double value) {
    if (!std::isfinite(value)) {
        return "null";
    }
    std::ostringstream out;
    out << std::setprecision(17) << value;
    return out.str();
}

}  // namespace

std::string to_json(const RunReport& report) {
    std::ostringstream out;
    out << std::setprecision(17);
    out << "{\n";
    out << "  \"version\": \"" << escape(report.version) << "\",\n";
    out << "  \"route\": \"" << escape(report.route) << "\",\n";
    out << "  \"precision\": \"" << escape(report.precision) << "\",\n";
    out << "  \"setup_seconds\": " << number(report.setup_seconds) << ",\n";
    out << "  \"solve_seconds\": " << number(report.solve_seconds) << ",\n";

    out << "  \"energy\": {\n";
    out << "    \"point_to_point\": " << number(report.energy.point_to_point) << ",\n";
    out << "    \"point_to_plane\": " << number(report.energy.point_to_plane) << ",\n";
    out << "    \"smoothness\": " << number(report.energy.smoothness) << ",\n";
    out << "    \"anchor\": " << number(report.energy.anchor) << ",\n";
    out << "    \"landmark\": " << number(report.energy.landmark) << ",\n";
    out << "    \"total\": " << number(report.energy.total()) << "\n";
    out << "  },\n";

    out << "  \"correspondence\": {\n";
    out << "    \"total\": " << report.correspondence.total << ",\n";
    out << "    \"accepted\": " << report.correspondence.accepted << ",\n";
    out << "    \"rejected_distance\": " << report.correspondence.rejected_distance
        << ",\n";
    out << "    \"rejected_normal\": " << report.correspondence.rejected_normal
        << ",\n";
    out << "    \"rejected_slope\": " << report.correspondence.rejected_slope << ",\n";
    out << "    \"rejected_masked\": " << report.correspondence.rejected_masked << ",\n";
    out << "    \"distance_mean\": " << number(report.correspondence.distance_mean)
        << ",\n";
    out << "    \"distance_p95\": " << number(report.correspondence.distance_p95)
        << ",\n";
    out << "    \"distance_max\": " << number(report.correspondence.distance_max)
        << "\n";
    out << "  },\n";

    out << "  \"levels\": [";
    for (std::size_t i = 0; i < report.levels.size(); ++i) {
        const LevelView& level = report.levels[i];
        out << (i == 0 ? "\n" : ",\n");
        out << "    {\"level\": " << level.level << ", \"nodes\": " << level.nodes
            << ", \"outer_iterations\": " << level.outer_iterations
            << ", \"inner_iterations\": " << level.inner_iterations
            << ", \"energy_start\": " << number(level.energy_start)
            << ", \"energy_end\": " << number(level.energy_end)
            << ", \"correspondence_mean\": " << number(level.correspondence_mean)
            << ", \"converged\": " << (level.converged ? "true" : "false") << "}";
    }
    out << (report.levels.empty() ? "],\n" : "\n  ],\n");

    out << "  \"settings\": {";
    for (std::size_t i = 0; i < report.settings.size(); ++i) {
        out << (i == 0 ? "\n" : ",\n");
        out << "    \"" << escape(report.settings[i].first)
            << "\": " << number(report.settings[i].second);
    }
    out << (report.settings.empty() ? "}\n" : "\n  }\n");
    out << "}\n";
    return out.str();
}

}  // namespace nricp
