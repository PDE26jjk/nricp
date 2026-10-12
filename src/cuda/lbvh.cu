#include "cuda/lbvh.cuh"

#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/sort.h>
#include <thrust/system/cuda/execution_policy.h>

#include <cfloat>
#include <cstddef>
#include <vector>

#include "cuda/cuda_error.h"

namespace nricp {
namespace gpu {
namespace {

constexpr int kThreads = 256;
constexpr int kStackSize = 64;
constexpr unsigned int kNoParent = 0xffffffffu;
constexpr float kInfinity = 3.4e38f;

int blocks_for(int count) { return (count + kThreads - 1) / kThreads; }

__device__ inline Vec3f load_vec3(const float* data, int index) {
    return {data[index * 3 + 0], data[index * 3 + 1], data[index * 3 + 2]};
}

// ---------------------------------------------------------------------------
// Target precomputation: centroids and area-weighted vertex normals
// ---------------------------------------------------------------------------

static __global__ void centroid_kernel(
    const float* __restrict__ points, const int* __restrict__ faces,
    int triangle_count, float* __restrict__ centroids) {
    const int triangle = blockIdx.x * blockDim.x + threadIdx.x;
    if (triangle >= triangle_count) {
        return;
    }
    const int* corners = faces + triangle * 3;
    const Vec3f a = load_vec3(points, corners[0]);
    const Vec3f b = load_vec3(points, corners[1]);
    const Vec3f c = load_vec3(points, corners[2]);
    const Vec3f centroid{(a.x + b.x + c.x) / 3.0f, (a.y + b.y + c.y) / 3.0f,
                         (a.z + b.z + c.z) / 3.0f};
    centroids[triangle * 3 + 0] = centroid.x;
    centroids[triangle * 3 + 1] = centroid.y;
    centroids[triangle * 3 + 2] = centroid.z;
}

// Area-weighted vertex normals: the cross product's length is twice the
// triangle's area, so accumulating it unnormalized already weights each face by
// area. The accumulation is atomic, so the sum order varies between runs; the
// result is used as a filter direction, not as a measurement.
static __global__ void accumulate_face_normals_kernel(
    const float* __restrict__ points, const int* __restrict__ faces,
    int triangle_count, float* __restrict__ normals) {
    const int triangle = blockIdx.x * blockDim.x + threadIdx.x;
    if (triangle >= triangle_count) {
        return;
    }
    const int* corners = faces + triangle * 3;
    const Vec3f a = load_vec3(points, corners[0]);
    const Vec3f b = load_vec3(points, corners[1]);
    const Vec3f c = load_vec3(points, corners[2]);
    const Vec3f weighted = cross(b - a, c - a);
    for (int corner = 0; corner < 3; ++corner) {
        float* slot = normals + corners[corner] * 3;
        atomicAdd(&slot[0], weighted.x);
        atomicAdd(&slot[1], weighted.y);
        atomicAdd(&slot[2], weighted.z);
    }
}

static __global__ void normalize_vectors_kernel(float* __restrict__ data, int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    const Vec3f value = normalized(load_vec3(data, index));
    data[index * 3 + 0] = value.x;
    data[index * 3 + 1] = value.y;
    data[index * 3 + 2] = value.z;
}

// Scene bounds over the centroids, in one block and so without a second kernel
// launch and without a host round trip: the bounds are only used to quantize the
// Morton codes, so a shared-memory reduction is precise enough.
static __global__ void scene_bounds_kernel(
    const float* __restrict__ centroids, int count, float* __restrict__ bounds) {
    __shared__ float shared_min[3][kThreads];
    __shared__ float shared_max[3][kThreads];
    float local_min[3] = {kInfinity, kInfinity, kInfinity};
    float local_max[3] = {-kInfinity, -kInfinity, -kInfinity};
    for (int index = threadIdx.x; index < count; index += blockDim.x) {
        for (int axis = 0; axis < 3; ++axis) {
            const float value = centroids[index * 3 + axis];
            local_min[axis] = fminf(local_min[axis], value);
            local_max[axis] = fmaxf(local_max[axis], value);
        }
    }
    for (int axis = 0; axis < 3; ++axis) {
        shared_min[axis][threadIdx.x] = local_min[axis];
        shared_max[axis][threadIdx.x] = local_max[axis];
    }
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            for (int axis = 0; axis < 3; ++axis) {
                shared_min[axis][threadIdx.x] = fminf(
                    shared_min[axis][threadIdx.x], shared_min[axis][threadIdx.x + stride]);
                shared_max[axis][threadIdx.x] = fmaxf(
                    shared_max[axis][threadIdx.x], shared_max[axis][threadIdx.x + stride]);
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        for (int axis = 0; axis < 3; ++axis) {
            bounds[axis] = shared_min[axis][0];
            bounds[3 + axis] = shared_max[axis][0];
        }
    }
}

// ---------------------------------------------------------------------------
// Morton codes
// ---------------------------------------------------------------------------

// Spreads the low ten bits of `value` so that a 3-axis interleave fits in 30
// bits. Same masks as qmyidp's expand_bits_3d.
__device__ __host__ inline unsigned int expand_bits_3d(unsigned int value) {
    value &= 0x000003ffu;
    value = (value | (value << 16)) & 0x030000ffu;
    value = (value | (value << 8)) & 0x0300f00fu;
    value = (value | (value << 4)) & 0x030c30c3u;
    value = (value | (value << 2)) & 0x09249249u;
    return value;
}

__device__ __host__ inline unsigned int morton_code_3d(
    unsigned int x, unsigned int y, unsigned int z) {
    return (expand_bits_3d(z) << 2) | (expand_bits_3d(y) << 1) | expand_bits_3d(x);
}

static __global__ void morton_codes_kernel(
    const float* __restrict__ centroids, int count, const float* __restrict__ bounds,
    unsigned int* __restrict__ codes, unsigned int* __restrict__ order) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    const float min_x = bounds[0], min_y = bounds[1], min_z = bounds[2];
    const float max_x = bounds[3], max_y = bounds[4], max_z = bounds[5];
    const float scale_x = (max_x - min_x) > 1e-10f ? 1023.0f / (max_x - min_x) : 0.0f;
    const float scale_y = (max_y - min_y) > 1e-10f ? 1023.0f / (max_y - min_y) : 0.0f;
    const float scale_z = (max_z - min_z) > 1e-10f ? 1023.0f / (max_z - min_z) : 0.0f;
    const Vec3f centroid = load_vec3(centroids, index);
    const unsigned int x = static_cast<unsigned int>(
        fminf(1023.0f, fmaxf(0.0f, (centroid.x - min_x) * scale_x)));
    const unsigned int y = static_cast<unsigned int>(
        fminf(1023.0f, fmaxf(0.0f, (centroid.y - min_y) * scale_y)));
    const unsigned int z = static_cast<unsigned int>(
        fminf(1023.0f, fmaxf(0.0f, (centroid.z - min_z) * scale_z)));
    codes[index] = morton_code_3d(x, y, z);
    order[index] = static_cast<unsigned int>(index);
}

// ---------------------------------------------------------------------------
// Linear build (Karras)
// ---------------------------------------------------------------------------

__device__ inline int leading_zeros(unsigned int value) {
    return value == 0u ? 32 : __clz(value);
}

__device__ inline int longest_common_prefix(
    const unsigned int* codes, int count, int i, int j) {
    if (j < 0 || j >= count) {
        return -1;
    }
    const unsigned int key_i = codes[i];
    const unsigned int key_j = codes[j];
    if (key_i == key_j) {
        return 32 + leading_zeros(static_cast<unsigned int>(i ^ j));
    }
    return leading_zeros(key_i ^ key_j);
}

// Finds the range [left, right] the internal node `i` covers and where it splits.
__device__ inline void find_split(
    const unsigned int* codes, int count, int i, int* left, int* right, int* split) {
    const int prefix_previous = longest_common_prefix(codes, count, i, i - 1);
    const int prefix_next = longest_common_prefix(codes, count, i, i + 1);
    const int direction = prefix_next > prefix_previous ? 1 : -1;
    const int prefix_min =
        prefix_previous < prefix_next ? prefix_previous : prefix_next;
    int span = 2;
    while (longest_common_prefix(codes, count, i, i + span * direction) > prefix_min) {
        span *= 2;
    }
    int last = 0;
    for (int step = span / 2; step >= 1; step /= 2) {
        if (longest_common_prefix(codes, count, i, i + (last + step) * direction) >
            prefix_min) {
            last += step;
        }
    }
    const int j = i + last * direction;
    const int range_left = i < j ? i : j;
    const int range_right = i < j ? j : i;
    const int node_prefix = longest_common_prefix(codes, count, range_left, range_right);
    int offset = 0;
    int step = range_right - range_left;
    do {
        step = (step + 1) / 2;
        if (longest_common_prefix(codes, count, range_left, range_left + offset + step) >
            node_prefix) {
            offset += step;
        }
    } while (step > 1);
    *left = range_left;
    *right = range_right;
    *split = range_left + offset;
}

// Leaves occupy [0, count), internal nodes [count, 2 * count - 1); both store a
// child index plus one so that zero can mean "leaf".
static __global__ void build_tree_kernel(
    const unsigned int* __restrict__ codes, const unsigned int* __restrict__ order,
    int count, LbvhNode* __restrict__ nodes) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        nodes[index].left = static_cast<int>(order[index]) + 1;
        nodes[index].right = 0;
    }
    if (index >= count - 1) {
        return;
    }
    int left = 0, right = 0, split = 0;
    find_split(codes, count, index, &left, &right, &split);
    const int left_child = left == split ? left : count + split;
    const int right_child = split + 1 == right ? right : count + split + 1;
    nodes[count + index].left = left_child + 1;
    nodes[count + index].right = right_child + 1;
}

static __global__ void init_parent_kernel(unsigned int* __restrict__ parent, int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        parent[index] = kNoParent;
    }
}

static __global__ void set_parent_kernel(
    const LbvhNode* __restrict__ nodes, unsigned int* __restrict__ parent, int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    const LbvhNode node = nodes[index];
    if (node.right != 0) {
        parent[node.left - 1] = static_cast<unsigned int>(index);
        parent[node.right - 1] = static_cast<unsigned int>(index);
    }
}

// The root is the only internal node without a parent. It is resolved on the
// device rather than with thrust::find so that no result is read back here.
static __global__ void find_root_kernel(
    const unsigned int* __restrict__ parent, int leaf_count, int node_count,
    unsigned int* __restrict__ root) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= leaf_count && index < node_count &&
        parent[index] == kNoParent) {
        atomicMin(root, static_cast<unsigned int>(index));
    }
}

static __global__ void close_root_kernel(
    const unsigned int* __restrict__ root, unsigned int* __restrict__ parent) {
    // A self-parent marks the root, which is what stops the bottom-up refit.
    parent[*root] = *root;
}

// ---------------------------------------------------------------------------
// Bounding volumes
// ---------------------------------------------------------------------------

__device__ inline AABB3D merge_boxes(const AABB3D& a, const AABB3D& b) {
    AABB3D merged;
    merged.min = {fminf(a.min.x, b.min.x), fminf(a.min.y, b.min.y),
                  fminf(a.min.z, b.min.z)};
    merged.max = {fmaxf(a.max.x, b.max.x), fmaxf(a.max.y, b.max.y),
                  fmaxf(a.max.z, b.max.z)};
    return merged;
}

// Walks from a finished leaf to the root, merging a parent's two child boxes as
// soon as both children have reported. The fence makes a child's box visible
// before its completion counter is incremented, so the second thread to arrive
// always reads two finished boxes. Same protocol as qmyidp's bottom_up_refit.
__device__ inline void refit_upwards(
    unsigned int start, const LbvhNode* __restrict__ nodes,
    const unsigned int* __restrict__ parent, unsigned int* __restrict__ refit_count,
    AABB3D* __restrict__ aabbs) {
    unsigned int index = start;
    for (;;) {
        const unsigned int parent_index = parent[index];
        if (parent_index == index || parent_index == kNoParent) {
            return;
        }
        __threadfence();
        if (atomicAdd(&refit_count[parent_index], 1u) != 1u) {
            // The sibling has not arrived yet; when it does it merges and
            // carries on towards the root.
            return;
        }
        const LbvhNode node = nodes[parent_index];
        aabbs[parent_index] =
            merge_boxes(aabbs[node.left - 1], aabbs[node.right - 1]);
        index = parent_index;
    }
}

// Computes the leaf boxes and fits every internal box in the same launch.
static __global__ void refit_kernel(
    const float* __restrict__ points, const int* __restrict__ faces,
    int triangle_count, const LbvhNode* __restrict__ nodes,
    const unsigned int* __restrict__ parent, unsigned int* __restrict__ refit_count,
    AABB3D* __restrict__ aabbs) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= triangle_count) {
        return;
    }
    // The leaf at `index` holds primitive `nodes[index].left - 1`, which the
    // Morton sort has permuted away from `index`; taking the primitive from the
    // node is what keeps the box with its own triangle.
    const int triangle = nodes[index].left - 1;
    const int* corners = faces + triangle * 3;
    const Vec3f a = load_vec3(points, corners[0]);
    const Vec3f b = load_vec3(points, corners[1]);
    const Vec3f c = load_vec3(points, corners[2]);
    AABB3D box;
    box.min = {fminf(a.x, fminf(b.x, c.x)), fminf(a.y, fminf(b.y, c.y)),
               fminf(a.z, fminf(b.z, c.z))};
    box.max = {fmaxf(a.x, fmaxf(b.x, c.x)), fmaxf(a.y, fmaxf(b.y, c.y)),
               fmaxf(a.z, fmaxf(b.z, c.z))};
    aabbs[index] = box;
    refit_upwards(static_cast<unsigned int>(index), nodes, parent, refit_count,
                  aabbs);
}

// ---------------------------------------------------------------------------
// Nearest-point query
// ---------------------------------------------------------------------------

__device__ inline float squared_distance_to_box(const Vec3f& point, const AABB3D& box) {
    const float dx = fmaxf(fmaxf(box.min.x - point.x, point.x - box.max.x), 0.0f);
    const float dy = fmaxf(fmaxf(box.min.y - point.y, point.y - box.max.y), 0.0f);
    const float dz = fmaxf(fmaxf(box.min.z - point.z, point.z - box.max.z), 0.0f);
    return dx * dx + dy * dy + dz * dz;
}

// Closest point of a triangle by the standard Voronoi-region test; the
// barycentric coordinates come back for the normal interpolation.
__device__ inline Vec3f closest_point_on_triangle(
    const Vec3f& point, const Vec3f& a, const Vec3f& b, const Vec3f& c,
    float* barycentric) {
    const Vec3f ab = b - a;
    const Vec3f ac = c - a;
    const Vec3f ap = point - a;
    const float d1 = dot(ab, ap);
    const float d2 = dot(ac, ap);
    if (d1 <= 0.0f && d2 <= 0.0f) {
        barycentric[0] = 1.0f;
        barycentric[1] = 0.0f;
        barycentric[2] = 0.0f;
        return a;
    }
    const Vec3f bp = point - b;
    const float d3 = dot(ab, bp);
    const float d4 = dot(ac, bp);
    if (d3 >= 0.0f && d4 <= d3) {
        barycentric[0] = 0.0f;
        barycentric[1] = 1.0f;
        barycentric[2] = 0.0f;
        return b;
    }
    const float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
        const float v = d1 - d3 != 0.0f ? d1 / (d1 - d3) : 0.0f;
        barycentric[0] = 1.0f - v;
        barycentric[1] = v;
        barycentric[2] = 0.0f;
        return a + ab * v;
    }
    const Vec3f cp = point - c;
    const float d5 = dot(ab, cp);
    const float d6 = dot(ac, cp);
    if (d6 >= 0.0f && d5 <= d6) {
        barycentric[0] = 0.0f;
        barycentric[1] = 0.0f;
        barycentric[2] = 1.0f;
        return c;
    }
    const float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
        const float w = d2 - d6 != 0.0f ? d2 / (d2 - d6) : 0.0f;
        barycentric[0] = 1.0f - w;
        barycentric[1] = 0.0f;
        barycentric[2] = w;
        return a + ac * w;
    }
    const float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
        const float denominator = (d4 - d3) + (d5 - d6);
        const float w = denominator != 0.0f ? (d4 - d3) / denominator : 0.0f;
        barycentric[0] = 0.0f;
        barycentric[1] = 1.0f - w;
        barycentric[2] = w;
        return b + (c - b) * w;
    }
    const float denominator = va + vb + vc;
    if (fabsf(denominator) < 1e-20f) {
        barycentric[0] = 1.0f;
        barycentric[1] = 0.0f;
        barycentric[2] = 0.0f;
        return a;
    }
    const float v = vb / denominator;
    const float w = vc / denominator;
    barycentric[0] = 1.0f - v - w;
    barycentric[1] = v;
    barycentric[2] = w;
    return a + ab * v + ac * w;
}

static __global__ void query_kernel(
    const LbvhNode* __restrict__ nodes, const AABB3D* __restrict__ aabbs,
    const unsigned int* __restrict__ root, const float* __restrict__ points,
    const int* __restrict__ faces, const float* __restrict__ vertex_normals,
    const float* __restrict__ queries, const float* __restrict__ query_normals,
    float distance_threshold, float min_normal_cosine, float min_slope,
    const unsigned char* __restrict__ mask, int query_count,
    float* __restrict__ out_points, float* __restrict__ out_normals,
    float* __restrict__ out_distances, int* __restrict__ out_triangles,
    unsigned char* __restrict__ out_rejection) {
    const int query = blockIdx.x * blockDim.x + threadIdx.x;
    if (query >= query_count) {
        return;
    }
    const Vec3f point = load_vec3(queries, query);
    Vec3f best_point;
    Vec3f best_normal;
    float best_distance_squared = kInfinity;
    int best_triangle = -1;

    int stack[kStackSize];
    int stack_size = 0;
    stack[stack_size++] = static_cast<int>(*root);
    while (stack_size > 0) {
        const int node_index = stack[--stack_size];
        if (squared_distance_to_box(point, aabbs[node_index]) >= best_distance_squared) {
            continue;
        }
        const LbvhNode node = nodes[node_index];
        if (node.right == 0) {
            const int triangle = node.left - 1;
            const int* corners = faces + triangle * 3;
            float barycentric[3];
            const Vec3f closest = closest_point_on_triangle(
                point, load_vec3(points, corners[0]), load_vec3(points, corners[1]),
                load_vec3(points, corners[2]), barycentric);
            const float distance_squared = squared_norm(closest - point);
            if (distance_squared < best_distance_squared) {
                best_distance_squared = distance_squared;
                best_point = closest;
                best_triangle = triangle;
                best_normal = normalized(
                    load_vec3(vertex_normals, corners[0]) * barycentric[0] +
                    load_vec3(vertex_normals, corners[1]) * barycentric[1] +
                    load_vec3(vertex_normals, corners[2]) * barycentric[2]);
                if (best_distance_squared <= 0.0f) {
                    break;
                }
            }
            continue;
        }
        if (stack_size >= kStackSize - 2) {
            continue;
        }
        const int left = node.left - 1;
        const int right = node.right - 1;
        // Push the farther child first so the closer one is popped first.
        const float left_distance = squared_distance_to_box(point, aabbs[left]);
        const float right_distance = squared_distance_to_box(point, aabbs[right]);
        if (left_distance < right_distance) {
            stack[stack_size++] = right;
            stack[stack_size++] = left;
        } else {
            stack[stack_size++] = left;
            stack[stack_size++] = right;
        }
    }

    const float distance = sqrtf(best_distance_squared);
    out_points[query * 3 + 0] = best_point.x;
    out_points[query * 3 + 1] = best_point.y;
    out_points[query * 3 + 2] = best_point.z;
    out_normals[query * 3 + 0] = best_normal.x;
    out_normals[query * 3 + 1] = best_normal.y;
    out_normals[query * 3 + 2] = best_normal.z;
    out_distances[query] = distance;
    out_triangles[query] = best_triangle;

    // The filters in the order the CPU path applies them, so the rejection the
    // report counts is the same one.
    unsigned char code = static_cast<unsigned char>(QueryRejection::Accepted);
    if (mask != nullptr && mask[query] == 0) {
        code = static_cast<unsigned char>(QueryRejection::Masked);
    } else if (distance > distance_threshold) {
        code = static_cast<unsigned char>(QueryRejection::Distance);
    } else if (dot(load_vec3(query_normals, query), best_normal) < min_normal_cosine) {
        code = static_cast<unsigned char>(QueryRejection::Normal);
    } else if (distance > 0.0f && min_slope > 0.0f) {
        const Vec3f offset = point - best_point;
        if (fabsf(dot(best_normal, offset)) / distance < min_slope) {
            code = static_cast<unsigned char>(QueryRejection::Slope);
        }
    }
    out_rejection[query] = code;
}

}  // namespace

void Lbvh::clear() {
    triangle_count_ = 0;
    vertex_count_ = 0;
    points_.clear();
    faces_.clear();
    normals_.clear();
    centroids_.clear();
    codes_.clear();
    order_.clear();
    nodes_.clear();
    aabbs_.clear();
    parent_.clear();
    refit_count_.clear();
    root_.clear();
}

void Lbvh::build(const Mesh& target_mesh, cudaStream_t stream) {
    clear();
    triangle_count_ = static_cast<int>(target_mesh.faces.size());
    vertex_count_ = static_cast<int>(target_mesh.vertices.size());
    if (triangle_count_ <= 0 || vertex_count_ <= 0) {
        triangle_count_ = 0;
        vertex_count_ = 0;
        return;
    }

    std::vector<float> host_points(static_cast<std::size_t>(vertex_count_) * 3);
    for (int vertex = 0; vertex < vertex_count_; ++vertex) {
        const Vec3& value = target_mesh.vertices[static_cast<std::size_t>(vertex)];
        host_points[vertex * 3 + 0] = static_cast<float>(value.x);
        host_points[vertex * 3 + 1] = static_cast<float>(value.y);
        host_points[vertex * 3 + 2] = static_cast<float>(value.z);
    }
    std::vector<int> host_faces(static_cast<std::size_t>(triangle_count_) * 3);
    for (int triangle = 0; triangle < triangle_count_; ++triangle) {
        const Int3& face = target_mesh.faces[static_cast<std::size_t>(triangle)];
        host_faces[triangle * 3 + 0] = face.v0;
        host_faces[triangle * 3 + 1] = face.v1;
        host_faces[triangle * 3 + 2] = face.v2;
    }
    points_.assign(host_points.begin(), host_points.end());
    faces_.assign(host_faces.begin(), host_faces.end());

    const float* points = thrust::raw_pointer_cast(points_.data());
    const int* faces = thrust::raw_pointer_cast(faces_.data());

    normals_.resize(static_cast<std::size_t>(vertex_count_) * 3);
    thrust::fill(thrust::cuda::par.on(stream), normals_.begin(), normals_.end(), 0.0f);
    accumulate_face_normals_kernel<<<blocks_for(triangle_count_), kThreads, 0, stream>>>(
        points, faces, triangle_count_, thrust::raw_pointer_cast(normals_.data()));
    throw_on_cuda_error(cudaGetLastError());
    normalize_vectors_kernel<<<blocks_for(vertex_count_), kThreads, 0, stream>>>(
        thrust::raw_pointer_cast(normals_.data()), vertex_count_);
    throw_on_cuda_error(cudaGetLastError());

    centroids_.resize(static_cast<std::size_t>(triangle_count_) * 3);
    centroid_kernel<<<blocks_for(triangle_count_), kThreads, 0, stream>>>(
        points, faces, triangle_count_, thrust::raw_pointer_cast(centroids_.data()));
    throw_on_cuda_error(cudaGetLastError());

    thrust::device_vector<float> bounds(6);
    scene_bounds_kernel<<<1, kThreads, 0, stream>>>(
        thrust::raw_pointer_cast(centroids_.data()), triangle_count_,
        thrust::raw_pointer_cast(bounds.data()));
    throw_on_cuda_error(cudaGetLastError());

    codes_.resize(static_cast<std::size_t>(triangle_count_));
    order_.resize(static_cast<std::size_t>(triangle_count_));
    morton_codes_kernel<<<blocks_for(triangle_count_), kThreads, 0, stream>>>(
        thrust::raw_pointer_cast(centroids_.data()), triangle_count_,
        thrust::raw_pointer_cast(bounds.data()),
        thrust::raw_pointer_cast(codes_.data()),
        thrust::raw_pointer_cast(order_.data()));
    throw_on_cuda_error(cudaGetLastError());

    thrust::sort_by_key(thrust::cuda::par.on(stream), codes_.begin(), codes_.end(),
                        order_.begin());

    const int node_count = 2 * triangle_count_ - 1;
    nodes_.resize(static_cast<std::size_t>(node_count));
    parent_.resize(static_cast<std::size_t>(node_count));
    aabbs_.resize(static_cast<std::size_t>(node_count));
    refit_count_.resize(static_cast<std::size_t>(node_count));
    root_.resize(1);

    build_tree_kernel<<<blocks_for(node_count > triangle_count_ ? node_count
                                                               : triangle_count_),
                        kThreads, 0, stream>>>(
        thrust::raw_pointer_cast(codes_.data()),
        thrust::raw_pointer_cast(order_.data()), triangle_count_,
        thrust::raw_pointer_cast(nodes_.data()));
    throw_on_cuda_error(cudaGetLastError());

    init_parent_kernel<<<blocks_for(node_count), kThreads, 0, stream>>>(
        thrust::raw_pointer_cast(parent_.data()), node_count);
    throw_on_cuda_error(cudaGetLastError());
    set_parent_kernel<<<blocks_for(node_count), kThreads, 0, stream>>>(
        thrust::raw_pointer_cast(nodes_.data()),
        thrust::raw_pointer_cast(parent_.data()), node_count);
    throw_on_cuda_error(cudaGetLastError());

    if (triangle_count_ == 1) {
        // A single leaf is its own root and has no parent to merge into.
        throw_on_cuda_error(cudaMemsetAsync(
            thrust::raw_pointer_cast(root_.data()), 0, sizeof(unsigned int), stream));
        const unsigned int zero = 0;
        throw_on_cuda_error(cudaMemcpyAsync(
            thrust::raw_pointer_cast(parent_.data()), &zero, sizeof(unsigned int),
            cudaMemcpyHostToDevice, stream));
    } else {
        throw_on_cuda_error(cudaMemsetAsync(
            thrust::raw_pointer_cast(root_.data()), 0xff, sizeof(unsigned int), stream));
        find_root_kernel<<<blocks_for(node_count), kThreads, 0, stream>>>(
            thrust::raw_pointer_cast(parent_.data()), triangle_count_, node_count,
            thrust::raw_pointer_cast(root_.data()));
        throw_on_cuda_error(cudaGetLastError());
        close_root_kernel<<<1, 1, 0, stream>>>(
            thrust::raw_pointer_cast(root_.data()),
            thrust::raw_pointer_cast(parent_.data()));
        throw_on_cuda_error(cudaGetLastError());
    }

    throw_on_cuda_error(cudaMemsetAsync(
        thrust::raw_pointer_cast(refit_count_.data()), 0,
        static_cast<std::size_t>(node_count) * sizeof(unsigned int), stream));
    refit_kernel<<<blocks_for(triangle_count_), kThreads, 0, stream>>>(
        points, faces, triangle_count_, thrust::raw_pointer_cast(nodes_.data()),
        thrust::raw_pointer_cast(parent_.data()),
        thrust::raw_pointer_cast(refit_count_.data()),
        thrust::raw_pointer_cast(aabbs_.data()));
    throw_on_cuda_error(cudaGetLastError());
}

void Lbvh::query(const float* queries, const float* query_normals, int query_count,
                 const unsigned char* mask, float distance_threshold,
                 float min_normal_cosine, float min_slope, float* out_points,
                 float* out_normals, float* out_distances, int* out_triangles,
                 unsigned char* out_rejection, cudaStream_t stream) const {
    if (query_count <= 0 || triangle_count_ == 0) {
        return;
    }
    query_kernel<<<blocks_for(query_count), kThreads, 0, stream>>>(
        thrust::raw_pointer_cast(nodes_.data()),
        thrust::raw_pointer_cast(aabbs_.data()),
        thrust::raw_pointer_cast(root_.data()),
        thrust::raw_pointer_cast(points_.data()),
        thrust::raw_pointer_cast(faces_.data()),
        thrust::raw_pointer_cast(normals_.data()), queries, query_normals,
        distance_threshold, min_normal_cosine, min_slope, mask, query_count,
        out_points, out_normals, out_distances, out_triangles, out_rejection);
    throw_on_cuda_error(cudaGetLastError());
}

std::size_t Lbvh::device_bytes() const {
    const auto count = [](std::size_t elements, std::size_t size) {
        return elements * size;
    };
    return count(points_.size(), sizeof(float)) + count(faces_.size(), sizeof(int)) +
           count(normals_.size(), sizeof(float)) +
           count(centroids_.size(), sizeof(float)) +
           count(codes_.size(), sizeof(unsigned int)) +
           count(order_.size(), sizeof(unsigned int)) +
           count(nodes_.size(), sizeof(LbvhNode)) +
           count(aabbs_.size(), sizeof(AABB3D)) +
           count(parent_.size(), sizeof(unsigned int)) +
           count(refit_count_.size(), sizeof(unsigned int)) +
           count(root_.size(), sizeof(unsigned int));
}

}  // namespace gpu
}  // namespace nricp
