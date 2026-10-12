#pragma once

#include <cstddef>

#include <thrust/device_vector.h>

#include "core/mesh.h"
#include "cuda/device_math.cuh"

namespace nricp {
namespace gpu {

// Bounding volume of one hierarchy node.
struct AABB3D {
    Vec3f min;
    Vec3f max;
};

// One node of the linear hierarchy. A leaf stores its primitive index plus one
// in `left` and leaves `right` at zero; an internal node stores both child
// indices plus one. The "+1" leaves zero free to mean "not a child", so the
// traversal is a plain index walk with a single test per node -- the encoding
// qmyidp's lbvh uses.
struct LbvhNode {
    int left = 0;
    int right = 0;
};

// Why a query was dropped. The codes match nricp::Rejection in
// solver/correspondence.h so the solver can tally the same categories the CPU
// path reports; zero means the correspondence was accepted.
enum class QueryRejection : unsigned char {
    Accepted = 0,
    Distance = 1,
    Normal = 2,
    Slope = 3,
    Masked = 4,
};

// A Morton-code LBVH over the target's triangles, built and kept on the device.
//
// Layout and build follow qmyidp's src/simulation/contact/lbvh.cu (itself
// adapted from ZOZO's ppf-contact-solver): 30-bit Morton codes from the triangle
// centroids, a thrust radix sort of the (code, index) pairs, Karras' linear
// build, then one bottom-up kernel that fits every box. The target is static
// over a wrap, so the hierarchy is built once and only queried afterwards.
//
// Everything stays on the device, including the root index, which is why
// building it costs no synchronization: the solver uploads the target mesh once
// before the schedule starts and downloads nothing until it ends.
class Lbvh {
public:
    // Builds the hierarchy over `target_mesh` and keeps its own device copy of
    // the vertices, faces and area-weighted vertex normals, so a query needs
    // nothing but the query points.
    void build(const Mesh& target_mesh, cudaStream_t stream);

    // Nearest point on the target surface for every query, with the same filters
    // the CPU path applies: distance, normal angle, slope, and the vertex mask.
    // `query_normals` are the deformed template's vertex normals.
    // `out_rejection` receives one QueryRejection code per query.
    void query(const float* queries, const float* query_normals, int query_count,
               const unsigned char* mask, float distance_threshold,
               float min_normal_cosine, float min_slope, float* out_points,
               float* out_normals, float* out_distances, int* out_triangles,
               unsigned char* out_rejection, cudaStream_t stream) const;

    void clear();

    bool empty() const { return triangle_count_ == 0; }
    int triangle_count() const { return triangle_count_; }
    int vertex_count() const { return vertex_count_; }
    int node_count() const { return static_cast<int>(nodes_.size()); }

    // Device bytes held, for the report.
    std::size_t device_bytes() const;

private:
    int triangle_count_ = 0;
    int vertex_count_ = 0;
    thrust::device_vector<float> points_;
    thrust::device_vector<int> faces_;
    thrust::device_vector<float> normals_;
    thrust::device_vector<float> centroids_;
    thrust::device_vector<unsigned int> codes_;
    thrust::device_vector<unsigned int> order_;
    thrust::device_vector<LbvhNode> nodes_;
    thrust::device_vector<AABB3D> aabbs_;
    thrust::device_vector<unsigned int> parent_;
    thrust::device_vector<unsigned int> refit_count_;
    thrust::device_vector<unsigned int> root_;
};

}  // namespace gpu
}  // namespace nricp
