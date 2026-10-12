#pragma once

#include <limits>
#include <vector>

#include "core/constraints.h"
#include "core/vec3.h"
#include "geometry/closest_point.h"

namespace nricp {

// Why a correspondence was thrown away. Everything rejected here is a latent
// variable of the fit, not a force: it simply stops contributing this iteration
// and is re-estimated next time.
enum class Rejection : int {
    Accepted = 0,
    Distance = 1,
    Normal = 2,
    Slope = 3,
    Masked = 4,
};

struct CorrespondenceSettings {
    // Distance beyond which the closest point is considered unrelated.
    double distance_threshold = std::numeric_limits<double>::infinity();
    // Reject when the template normal and the target normal disagree by more
    // than the angle whose cosine this is (grazing or opposite surfaces).
    double min_normal_cosine = -1.0;
    // Slope filter. The ratio |n . (v' - y)| / |v' - y| is 1 when the connection
    // runs along the surface normal (the vertex faces its closest point) and 0
    // when it runs along the surface (the closest point is off to the side, so
    // the vertex is sliding sideways and a small motion would move its
    // correspondence elsewhere). Sliding connections are the useless ones, so a
    // ratio *below* this threshold is rejected. Zero disables the filter.
    double min_slope = 0.0;
};

struct Correspondence {
    Vec3 point;   // closest point on the target surface
    Vec3 normal;  // target normal there
    double distance = 0.0;
    int triangle = -1;
    Rejection rejection = Rejection::Accepted;

    bool accepted() const { return rejection == Rejection::Accepted; }
};

struct CorrespondenceCounts {
    int total = 0;
    int accepted = 0;
    int rejected_distance = 0;
    int rejected_normal = 0;
    int rejected_slope = 0;
    int rejected_masked = 0;
    double distance_mean = 0.0;
    double distance_p95 = 0.0;
    double distance_max = 0.0;

    double rejection_fraction() const {
        return total > 0 ? 1.0 - static_cast<double>(accepted) / total : 0.0;
    }
};

struct CorrespondenceSet {
    std::vector<Correspondence> items;  // one per query point, in input order
    CorrespondenceCounts counts;
};

// Finds the closest target point for every query point and applies the filters.
// `query_normals` are the deformed template's vertex normals (the spec's n'_i),
// which the normal-angle filter compares against the target normal.
CorrespondenceSet find_correspondences(
    const TargetQueries& target,
    const std::vector<Vec3>& query_points,
    const std::vector<Vec3>& query_normals,
    const CorrespondenceSettings& settings,
    const VertexMask* mask = nullptr);

}  // namespace nricp
