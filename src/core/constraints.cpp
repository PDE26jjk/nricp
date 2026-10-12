#include "core/constraints.h"

#include <string>
#include <unordered_set>

namespace nricp {

ValidationIssues validate_landmarks(
    const std::vector<Landmark>& landmarks, int template_vertex_count) {
    ValidationIssues issues;
    std::unordered_set<int> seen;
    for (std::size_t i = 0; i < landmarks.size(); ++i) {
        const Landmark& landmark = landmarks[i];
        if (landmark.template_vertex < 0 ||
            landmark.template_vertex >= template_vertex_count) {
            issues.add("landmark " + std::to_string(i) + " points at template vertex " +
                       std::to_string(landmark.template_vertex) +
                       " but the template has " + std::to_string(template_vertex_count) +
                       " vertices");
            continue;
        }
        if (!is_finite(landmark.target)) {
            issues.add("landmark " + std::to_string(i) + " has a non-finite target");
        }
        if (!seen.insert(landmark.template_vertex).second) {
            issues.add("landmark " + std::to_string(i) + " marks template vertex " +
                       std::to_string(landmark.template_vertex) + " twice");
        }
    }
    return issues;
}

ValidationIssues validate_mask(const VertexMask& mask, int template_vertex_count) {
    ValidationIssues issues;
    if (mask.active.empty()) {
        // An empty mask means "no mask": everything participates.
        return issues;
    }
    if (static_cast<int>(mask.active.size()) != template_vertex_count) {
        issues.add("mask has " + std::to_string(mask.active.size()) +
                   " entries but the template has " +
                   std::to_string(template_vertex_count) + " vertices");
        return issues;
    }
    int active = 0;
    for (std::size_t i = 0; i < mask.active.size(); ++i) {
        const unsigned char value = mask.active[i];
        if (value != 0 && value != 1) {
            issues.add("mask entry " + std::to_string(i) + " is " +
                       std::to_string(static_cast<int>(value)) + ", expected 0 or 1");
        }
        active += value ? 1 : 0;
    }
    if (active == 0) {
        issues.add("mask excludes every vertex; the fit would be unconstrained");
    }
    return issues;
}

int active_vertex_count(const VertexMask& mask) {
    if (mask.active.empty()) {
        return 0;
    }
    int active = 0;
    for (const unsigned char value : mask.active) {
        active += value ? 1 : 0;
    }
    return active;
}

}  // namespace nricp
