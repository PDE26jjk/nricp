#pragma once

#include <vector>

#include "core/mesh.h"

namespace nricp {

struct ClosestPointResult {
    Vec3 point;
    Vec3 normal;  // interpolated vertex normal at the closest point
    double distance = 0.0;
    int triangle = -1;
};

// Closest point on a triangle, plus its barycentric coordinates (u, v, w) with
// respect to (a, b, c). Degenerate triangles fall back to the nearest corner.
Vec3 closest_point_on_triangle(
    const Vec3& point,
    const Vec3& a,
    const Vec3& b,
    const Vec3& c,
    double* barycentric);

// Closest-point structure over the target surface. It is built once per wrap
// call because the target does not move; only the query points do.
class TargetQueries {
public:
    // BVH layout. Public so the accelerated path can upload the structure the
    // host built instead of rebuilding it on the device: building is O(F log F)
    // and happens once per call, while the queries happen millions of times.
    struct Node {
        Vec3 min;
        Vec3 max;
        int left = -1;
        int right = -1;
        int begin = 0;  // leaf: range in primitive_order_
        int count = 0;
    };

    explicit TargetQueries(const Mesh& mesh);

    ClosestPointResult query(const Vec3& point) const;

    const Mesh& mesh() const { return mesh_; }
    int triangle_count() const { return static_cast<int>(mesh_.faces.size()); }
    int node_count() const { return static_cast<int>(nodes_.size()); }
    const std::vector<Vec3>& vertex_normals() const { return vertex_normals_; }
    const std::vector<Node>& nodes() const { return nodes_; }
    const std::vector<int>& primitive_order() const { return primitive_order_; }
    const std::vector<Vec3>& corner_a() const { return corner_a_; }
    const std::vector<Vec3>& corner_b() const { return corner_b_; }
    const std::vector<Vec3>& corner_c() const { return corner_c_; }
    int root() const { return root_; }

private:
    int build(int begin, int end);
    void query_node(int node_index, const Vec3& point, double* best_distance_squared,
                    ClosestPointResult* best) const;

    Mesh mesh_;
    std::vector<Vec3> vertex_normals_;
    std::vector<Vec3> corner_a_;
    std::vector<Vec3> corner_b_;
    std::vector<Vec3> corner_c_;
    std::vector<Vec3> centroids_;
    std::vector<int> primitive_order_;
    std::vector<Node> nodes_;
    int root_ = -1;
};

}  // namespace nricp
