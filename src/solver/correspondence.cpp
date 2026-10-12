#include "solver/correspondence.h"

#include <algorithm>
#include <cmath>

namespace nricp {
namespace {

bool participates(const VertexMask* mask, std::size_t index) {
    if (mask == nullptr || mask->active.empty()) {
        return true;
    }
    return index < mask->active.size() && mask->active[index] != 0;
}

}  // namespace

CorrespondenceSet find_correspondences(
    const TargetQueries& target,
    const std::vector<Vec3>& query_points,
    const std::vector<Vec3>& query_normals,
    const CorrespondenceSettings& settings,
    const VertexMask* mask) {
    CorrespondenceSet set;
    set.items.resize(query_points.size());

    std::vector<double> accepted_distances;
    accepted_distances.reserve(query_points.size());

    for (std::size_t index = 0; index < query_points.size(); ++index) {
        Correspondence& item = set.items[index];
        item.rejection = Rejection::Accepted;
        if (!participates(mask, index)) {
            item.rejection = Rejection::Masked;
            ++set.counts.rejected_masked;
            set.counts.total += 1;
            continue;
        }

        const ClosestPointResult closest = target.query(query_points[index]);
        item.point = closest.point;
        item.normal = closest.normal;
        item.distance = closest.distance;
        item.triangle = closest.triangle;

        if (closest.distance > settings.distance_threshold) {
            item.rejection = Rejection::Distance;
        } else if (index < query_normals.size() &&
                   dot(query_normals[index], closest.normal) <
                       settings.min_normal_cosine) {
            item.rejection = Rejection::Normal;
        } else if (closest.distance > 0.0) {
            const Vec3 offset = query_points[index] - closest.point;
            const double slope =
                std::abs(dot(closest.normal, offset)) / closest.distance;
            if (slope < settings.min_slope) {
                item.rejection = Rejection::Slope;
            }
        }

        set.counts.total += 1;
        switch (item.rejection) {
            case Rejection::Accepted:
                ++set.counts.accepted;
                accepted_distances.push_back(item.distance);
                break;
            case Rejection::Distance:
                ++set.counts.rejected_distance;
                break;
            case Rejection::Normal:
                ++set.counts.rejected_normal;
                break;
            case Rejection::Slope:
                ++set.counts.rejected_slope;
                break;
            case Rejection::Masked:
                ++set.counts.rejected_masked;
                break;
        }
    }

    if (!accepted_distances.empty()) {
        std::sort(accepted_distances.begin(), accepted_distances.end());
        double sum = 0.0;
        for (const double distance : accepted_distances) {
            sum += distance;
        }
        set.counts.distance_mean = sum / accepted_distances.size();
        const std::size_t p95_index = static_cast<std::size_t>(
            std::min<double>(accepted_distances.size() - 1,
                             0.95 * (accepted_distances.size() - 1)));
        set.counts.distance_p95 = accepted_distances[p95_index];
        set.counts.distance_max = accepted_distances.back();
    }
    return set;
}

}  // namespace nricp
