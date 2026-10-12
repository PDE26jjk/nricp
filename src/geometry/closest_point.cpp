#include "geometry/closest_point.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace nricp {
namespace {

constexpr int kLeafSize = 8;

double distance_squared_to_aabb(const Vec3& point, const Vec3& min, const Vec3& max) {
    const Vec3 closest{
        std::clamp(point.x, min.x, max.x),
        std::clamp(point.y, min.y, max.y),
        std::clamp(point.z, min.z, max.z),
    };
    return squared_norm(point - closest);
}

}  // namespace

Vec3 closest_point_on_triangle(
    const Vec3& point,
    const Vec3& a,
    const Vec3& b,
    const Vec3& c,
    double* barycentric) {
    const Vec3 ab = b - a;
    const Vec3 ac = c - a;
    const Vec3 ap = point - a;
    const double d1 = dot(ab, ap);
    const double d2 = dot(ac, ap);
    if (d1 <= 0.0 && d2 <= 0.0) {
        if (barycentric) {
            barycentric[0] = 1.0;
            barycentric[1] = 0.0;
            barycentric[2] = 0.0;
        }
        return a;
    }

    const Vec3 bp = point - b;
    const double d3 = dot(ab, bp);
    const double d4 = dot(ac, bp);
    if (d3 >= 0.0 && d4 <= d3) {
        if (barycentric) {
            barycentric[0] = 0.0;
            barycentric[1] = 1.0;
            barycentric[2] = 0.0;
        }
        return b;
    }

    const double vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
        const double denominator = d1 - d3;
        const double v = denominator != 0.0 ? d1 / denominator : 0.0;
        if (barycentric) {
            barycentric[0] = 1.0 - v;
            barycentric[1] = v;
            barycentric[2] = 0.0;
        }
        return a + ab * v;
    }

    const Vec3 cp = point - c;
    const double d5 = dot(ab, cp);
    const double d6 = dot(ac, cp);
    if (d6 >= 0.0 && d5 <= d6) {
        if (barycentric) {
            barycentric[0] = 0.0;
            barycentric[1] = 0.0;
            barycentric[2] = 1.0;
        }
        return c;
    }

    const double vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
        const double denominator = d2 - d6;
        const double w = denominator != 0.0 ? d2 / denominator : 0.0;
        if (barycentric) {
            barycentric[0] = 1.0 - w;
            barycentric[1] = 0.0;
            barycentric[2] = w;
        }
        return a + ac * w;
    }

    const double va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
        const double denominator = (d4 - d3) + (d5 - d6);
        const double w = denominator != 0.0 ? (d4 - d3) / denominator : 0.0;
        if (barycentric) {
            barycentric[0] = 0.0;
            barycentric[1] = 1.0 - w;
            barycentric[2] = w;
        }
        return b + (c - b) * w;
    }

    const double denominator = va + vb + vc;
    if (std::abs(denominator) < 1e-300) {
        if (barycentric) {
            barycentric[0] = 1.0;
            barycentric[1] = 0.0;
            barycentric[2] = 0.0;
        }
        return a;
    }
    const double v = vb / denominator;
    const double w = vc / denominator;
    if (barycentric) {
        barycentric[0] = 1.0 - v - w;
        barycentric[1] = v;
        barycentric[2] = w;
    }
    return a + ab * v + ac * w;
}

TargetQueries::TargetQueries(const Mesh& mesh) : mesh_(mesh) {
    const int triangle_count = static_cast<int>(mesh_.faces.size());
    vertex_normals_.assign(mesh_.vertices.size(), Vec3{});

    corner_a_.resize(triangle_count);
    corner_b_.resize(triangle_count);
    corner_c_.resize(triangle_count);
    centroids_.resize(triangle_count);
    primitive_order_.resize(triangle_count);
    std::iota(primitive_order_.begin(), primitive_order_.end(), 0);

    for (int triangle = 0; triangle < triangle_count; ++triangle) {
        const Int3& face = mesh_.faces[triangle];
        const Vec3 a = mesh_.vertices[face.v0];
        const Vec3 b = mesh_.vertices[face.v1];
        const Vec3 c = mesh_.vertices[face.v2];
        corner_a_[triangle] = a;
        corner_b_[triangle] = b;
        corner_c_[triangle] = c;
        centroids_[triangle] = (a + b + c) / 3.0;

        // Area-weighted vertex normals: the cross product's length is twice the
        // triangle area, so accumulating it unnormalized already weights by area.
        const Vec3 weighted = cross(b - a, c - a);
        vertex_normals_[face.v0] += weighted;
        vertex_normals_[face.v1] += weighted;
        vertex_normals_[face.v2] += weighted;
    }
    for (Vec3& normal : vertex_normals_) {
        normal = normalized(normal);
    }

    if (triangle_count > 0) {
        nodes_.reserve(2 * triangle_count);
        root_ = build(0, triangle_count);
    }
}

int TargetQueries::build(int begin, int end) {
    Node node;
    node.min = corner_a_[primitive_order_[begin]];
    node.max = node.min;
    for (int i = begin; i < end; ++i) {
        const int triangle = primitive_order_[i];
        for (const Vec3& corner :
             {corner_a_[triangle], corner_b_[triangle], corner_c_[triangle]}) {
            node.min.x = std::min(node.min.x, corner.x);
            node.min.y = std::min(node.min.y, corner.y);
            node.min.z = std::min(node.min.z, corner.z);
            node.max.x = std::max(node.max.x, corner.x);
            node.max.y = std::max(node.max.y, corner.y);
            node.max.z = std::max(node.max.z, corner.z);
        }
    }

    const int index = static_cast<int>(nodes_.size());
    nodes_.push_back(node);
    const int count = end - begin;
    if (count <= kLeafSize) {
        nodes_[index].begin = begin;
        nodes_[index].count = count;
        return index;
    }

    // Median split along the widest axis of the centroid bounds.
    Vec3 centroid_min = centroids_[primitive_order_[begin]];
    Vec3 centroid_max = centroid_min;
    for (int i = begin; i < end; ++i) {
        const Vec3& centroid = centroids_[primitive_order_[i]];
        centroid_min.x = std::min(centroid_min.x, centroid.x);
        centroid_min.y = std::min(centroid_min.y, centroid.y);
        centroid_min.z = std::min(centroid_min.z, centroid.z);
        centroid_max.x = std::max(centroid_max.x, centroid.x);
        centroid_max.y = std::max(centroid_max.y, centroid.y);
        centroid_max.z = std::max(centroid_max.z, centroid.z);
    }
    const Vec3 extent = centroid_max - centroid_min;
    int axis = 0;
    if (extent.y > extent.x && extent.y >= extent.z) {
        axis = 1;
    } else if (extent.z > extent.x && extent.z > extent.y) {
        axis = 2;
    }
    const int middle = begin + count / 2;
    auto coordinate = [this, axis](int triangle) {
        const Vec3& centroid = centroids_[triangle];
        return axis == 0 ? centroid.x : (axis == 1 ? centroid.y : centroid.z);
    };
    std::nth_element(primitive_order_.begin() + begin,
                     primitive_order_.begin() + middle,
                     primitive_order_.begin() + end,
                     [&coordinate](int lhs, int rhs) {
                         const double a = coordinate(lhs);
                         const double b = coordinate(rhs);
                         return a < b || (a == b && lhs < rhs);
                     });

    const int left = build(begin, middle);
    const int right = build(middle, end);
    nodes_[index].left = left;
    nodes_[index].right = right;
    return index;
}

void TargetQueries::query_node(
    int node_index,
    const Vec3& point,
    double* best_distance_squared,
    ClosestPointResult* best) const {
    const Node& node = nodes_[node_index];
    if (distance_squared_to_aabb(point, node.min, node.max) > *best_distance_squared) {
        return;
    }
    if (node.left < 0) {
        for (int i = node.begin; i < node.begin + node.count; ++i) {
            const int triangle = primitive_order_[i];
            double barycentric[3] = {1.0, 0.0, 0.0};
            const Vec3 closest = closest_point_on_triangle(
                point, corner_a_[triangle], corner_b_[triangle], corner_c_[triangle],
                barycentric);
            const double distance_squared = squared_norm(closest - point);
            if (distance_squared < *best_distance_squared) {
                *best_distance_squared = distance_squared;
                best->point = closest;
                best->distance = std::sqrt(distance_squared);
                best->triangle = triangle;
                const Int3& face = mesh_.faces[triangle];
                best->normal = normalized(
                    vertex_normals_[face.v0] * barycentric[0] +
                    vertex_normals_[face.v1] * barycentric[1] +
                    vertex_normals_[face.v2] * barycentric[2]);
            }
        }
        return;
    }

    // Visit the nearer child first so the far one is pruned more often.
    const double left_distance =
        distance_squared_to_aabb(point, nodes_[node.left].min, nodes_[node.left].max);
    const double right_distance =
        distance_squared_to_aabb(point, nodes_[node.right].min, nodes_[node.right].max);
    const int first = left_distance <= right_distance ? node.left : node.right;
    const int second = first == node.left ? node.right : node.left;
    query_node(first, point, best_distance_squared, best);
    query_node(second, point, best_distance_squared, best);
}

ClosestPointResult TargetQueries::query(const Vec3& point) const {
    ClosestPointResult best;
    best.normal = Vec3{0.0, 0.0, 0.0};
    double best_distance_squared = std::numeric_limits<double>::infinity();
    if (root_ < 0) {
        return best;
    }
    query_node(root_, point, &best_distance_squared, &best);
    return best;
}

}  // namespace nricp
