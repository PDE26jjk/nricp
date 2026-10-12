#pragma once

#include <vector>

#include "core/mesh.h"
#include "core/vec3.h"

namespace nricp {

// A marked template vertex and the target position it must reach.
struct Landmark {
    int template_vertex = -1;
    Vec3 target;
};

// Landmark problems the caller can fix: index out of range, duplicate marking,
// non-finite target.
ValidationIssues validate_landmarks(
    const std::vector<Landmark>& landmarks, int template_vertex_count);

// Per-vertex participation mask for the data term: 1 = participates,
// 0 = excluded (hair, eyes, mouth interior, the far side of a partial scan).
struct VertexMask {
    std::vector<unsigned char> active;

    bool empty() const { return active.empty(); }
};

// Mask problems: wrong length, unknown values, or an entirely empty mask.
ValidationIssues validate_mask(const VertexMask& mask, int template_vertex_count);

int active_vertex_count(const VertexMask& mask);

}  // namespace nricp
