#include "cuda/gpu_solver.cuh"

#include <cooperative_groups.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/system/cuda/execution_policy.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/rotation.h"
#include "core/version.h"
#include "cuda/cuda_error.h"
#include "cuda/device_math.cuh"
#include "cuda/lbvh.cuh"
#include "solver/route_shared.h"

// Route B. The shape of the schedule is route A's -- coarse to fine,
// correspondence updates outside, damped Gauss-Newton steps inside -- but the
// state lives on the device for the whole of a level and the host only sees the
// numbers it needs to make a decision:
//
//   * the node transforms are not uploaded or downloaded inside a level. An
//     accepted step is applied by a kernel and a rejected one is undone by
//     copying the accepted state back on the device; the host learns the new
//     transforms at the level boundary, where the next level's warm start needs
//     them anyway.
//   * the deformed template's normals and the target's hierarchy come out of
//     kernels, so an outer iteration has no host-side geometry work in it.
//   * the damped linear solve is a single cooperative kernel. This is what the
//     timeline said to do: a kernel launch costs about 10.5 microseconds on the
//     reference machine (measured with an empty kernel), so the
//     host-orchestrated PCG's roughly 230 launches per solve dominated a run
//     whose own kernels add up to a few hundred milliseconds. The iteration now
//     stays on the device, its inner products are grid-wide reductions, and it
//     decides convergence itself, so a solve costs one launch instead of
//     hundreds.
//
// What stays host-side is the schedule: which energy is lower, whether a level
// has converged, which damping to try next.
namespace nricp {
namespace {

namespace cg = cooperative_groups;

using gpu::Lbvh;
using gpu::Mat3f;
using gpu::Mat6f;
using gpu::Vec3f;
using gpu::throw_on_cuda_error;

constexpr int kThreads = 256;
constexpr int kEnergyKinds = 5;  // point-to-point, plane, smoothness, anchor, landmark
constexpr int kRejectionKinds = 5;
constexpr int kReduceBlocks = 32;  // blocks that contribute a partial sum

// One block owns one node's block in the gathered accumulation, so this is the
// block's own thread count rather than the project-wide one.
constexpr int kGatherThreads = 128;

// Six values per node, so the gathered kernel reduces this many numbers per
// thread at the end of its sweep.
constexpr int kBlockValues = 36;

int blocks_for(int count) { return (count + kThreads - 1) / kThreads; }

int gather_blocks_for(int count) { return (count + kGatherThreads - 1) / kGatherThreads; }

__device__ inline Vec3f load_vec3(const float* data, int index) {
    return {data[index * 3 + 0], data[index * 3 + 1], data[index * 3 + 2]};
}

__device__ inline void store_vec3(float* data, int index, const Vec3f& value) {
    data[index * 3 + 0] = value.x;
    data[index * 3 + 1] = value.y;
    data[index * 3 + 2] = value.z;
}

// Sums `value` across the block; every thread returns the total.
//
// The trailing barrier is not redundant: without it a thread that is still
// reading `shared[0]` could race the next call's first write to it.
__device__ inline float block_sum(float value) {
    __shared__ float shared[kThreads];
    shared[threadIdx.x] = value;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            shared[threadIdx.x] += shared[threadIdx.x + offset];
        }
        __syncthreads();
    }
    const float total = shared[0];
    __syncthreads();
    return total;
}

// ---------------------------------------------------------------------------
// State on the device
// ---------------------------------------------------------------------------

// What the linear solve reports back. Nothing else about it reaches the host
// before the solve is over: the recurrence lives in device memory.
struct PcgState {
    float residual = 0.0f;
    float converged = 0.0f;
    float stop = 0.0f;
    int iteration_count = 0;
};

// The solve's working vectors, sliced out of one allocation. Four vectors of six
// values per node plus the per-iteration reduction slots, which are indexed by
// iteration so that no slot has to be cleared between iterations.
struct PcgWorkspace {
    float* residual = nullptr;
    float* direction = nullptr;
    float* image = nullptr;
    float* preconditioned = nullptr;
    float* reduce = nullptr;  // three slots per iteration: p.Ap, r.r, r.z

    __host__ __device__ static PcgWorkspace of(float* base, int node_count) {
        const std::size_t stride = static_cast<std::size_t>(node_count) * 6;
        PcgWorkspace workspace;
        workspace.residual = base;
        workspace.direction = base + stride;
        workspace.image = base + stride * 2;
        workspace.preconditioned = base + stride * 3;
        workspace.reduce = base + stride * 4;
        return workspace;
    }

    __host__ __device__ static std::size_t floats_for(int node_count,
                                                      int max_iterations) {
        return static_cast<std::size_t>(node_count) * 24 +
               static_cast<std::size_t>(max_iterations) * 3 + 4;
    }
};

static __global__ void evaluate_kernel(
    const float* points, const int* binding_nodes, const float* binding_weights,
    const int* binding_counts, const Mat3f* rotations, const float* translations,
    const float* node_rest, int vertex_count, float* out) {
    const int vertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex >= vertex_count) {
        return;
    }
    const Vec3f point = load_vec3(points, vertex);
    const int count = binding_counts[vertex];
    Vec3f sum;
    for (int slot = 0; slot < count; ++slot) {
        const int node = binding_nodes[vertex * 4 + slot];
        const float weight = binding_weights[vertex * 4 + slot];
        if (weight == 0.0f || node < 0) {
            continue;
        }
        const Vec3f rest = load_vec3(node_rest, node);
        const Vec3f local = point - rest;
        const Vec3f translation = load_vec3(translations, node);
        sum += (rotations[node] * local + rest + translation) * weight;
    }
    store_vec3(out, vertex, sum);
}

// Backs the accepted state up before a step is attempted, and puts it back when
// the step is rejected. One kernel for both directions: the caller swaps the
// buffers.
static __global__ void copy_state_kernel(
    const Mat3f* __restrict__ rotations_in, Mat3f* __restrict__ rotations_out,
    const float* __restrict__ translations_in, float* __restrict__ translations_out,
    int node_count) {
    const int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= node_count) {
        return;
    }
    rotations_out[node] = rotations_in[node];
    for (int axis = 0; axis < 3; ++axis) {
        translations_out[node * 3 + axis] = translations_in[node * 3 + axis];
    }
}

// The tangent-space increment, applied in place: R <- exp([w]) R and t <- t + u.
// The same update core/deformation.cpp's retract() applies on the host, so the
// device walks the trajectory the specification describes.
static __global__ void retract_kernel(
    Mat3f* __restrict__ rotations, float* __restrict__ translations,
    const float* __restrict__ delta, int node_count) {
    const int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= node_count) {
        return;
    }
    const float* values = delta + node * 6;
    const Vec3f omega{values[0], values[1], values[2]};
    rotations[node] = gpu::exp_so3(omega) * rotations[node];
    float* translation = translations + node * 3;
    translation[0] += values[3];
    translation[1] += values[4];
    translation[2] += values[5];
}

// ---------------------------------------------------------------------------
// Deformed template normals
// ---------------------------------------------------------------------------

// Area-weighted vertex normals of the deformed template, the spec's n'_i. The
// host used to evaluate the deformed mesh and compute these every outer
// iteration; doing it here is what takes that download out of the loop.
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
    store_vec3(data, index, normalized(load_vec3(data, index)));
}

// ---------------------------------------------------------------------------
// Correspondence filtering and reweighting
// ---------------------------------------------------------------------------

static __global__ void huber_weights_kernel(
    const float* deformed, const float* targets, const unsigned char* rejection,
    int count, float threshold, float* weights) {
    const int vertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex >= count) {
        return;
    }
    if (rejection[vertex] != 0 || threshold <= 0.0f) {
        weights[vertex] = 1.0f;
        return;
    }
    const float squared = squared_norm(load_vec3(deformed, vertex) -
                                       load_vec3(targets, vertex));
    const float squared_threshold = threshold * threshold;
    weights[vertex] = squared <= squared_threshold ? 1.0f : threshold / sqrtf(squared);
}

// Tallies why correspondences were dropped and sums the accepted distances. The
// report's percentiles need the distances ordered, which is a device sort at the
// end of the level; everything here is a plain atomic tally.
static __global__ void correspondence_counts_kernel(
    const unsigned char* __restrict__ rejection, const float* __restrict__ distances,
    int vertex_count, int* __restrict__ counts, float* __restrict__ distance_sum) {
    const int vertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex >= vertex_count) {
        return;
    }
    const unsigned char code = rejection[vertex];
    if (code == 0) {
        atomicAdd(&counts[0], 1);
        atomicAdd(distance_sum, distances[vertex]);
    } else if (code < kRejectionKinds) {
        atomicAdd(&counts[code], 1);
    }
}

// Rejected vertices are parked at infinity, so one sort of the whole array puts
// the accepted distances first, in order.
static __global__ void distances_with_infinity_kernel(
    const unsigned char* __restrict__ rejection, const float* __restrict__ distances,
    int vertex_count, float* __restrict__ out) {
    const int vertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex >= vertex_count) {
        return;
    }
    out[vertex] = rejection[vertex] == 0 ? distances[vertex] : 3.4e38f;
}

// Mean, p95 and max of the accepted distances, from the sorted array and the
// accepted count the tally produced. The percentile picks the same element the
// host implementation does.
static __global__ void percentile_kernel(
    const float* __restrict__ sorted, const int* __restrict__ counts,
    const float* __restrict__ distance_sum, float* __restrict__ stats) {
    const int accepted = counts[0];
    float mean = 0.0f;
    float p95 = 0.0f;
    float maximum = 0.0f;
    if (accepted > 0) {
        mean = distance_sum[0] / static_cast<float>(accepted);
        const int index = min(accepted - 1, static_cast<int>(0.95 * (accepted - 1)));
        p95 = sorted[index];
        maximum = sorted[accepted - 1];
    }
    stats[0] = mean;
    stats[1] = p95;
    stats[2] = maximum;
}

// ---------------------------------------------------------------------------
// Fused accumulation of the reduced system and the block energies
// ---------------------------------------------------------------------------

// Fills the six-vector Jacobian of one linear functional of a node transform.
__device__ inline void fill_linear_row(const Vec3f& lambda, const Vec3f& rotated,
                                       float weight, float row[6]) {
    const Vec3f rotation_part = cross(lambda, rotated);
    row[0] = -weight * rotation_part.x;
    row[1] = -weight * rotation_part.y;
    row[2] = -weight * rotation_part.z;
    row[3] = weight * lambda.x;
    row[4] = weight * lambda.y;
    row[5] = weight * lambda.z;
}

__device__ inline Vec3f node_rotated_offset(const Mat3f* rotations,
                                            const float* node_rest, int node,
                                            const Vec3f& point) {
    return rotations[node] * (point - load_vec3(node_rest, node));
}

// ---------------------------------------------------------------------------
// The gathered accumulation
//
// The data term used to run the other way round: every vertex scattered its
// 6x6 block into whichever nodes it touched, so a node's block was written by
// hundreds of threads at once and every write was an atomic into L2. ncu caught
// exactly that -- the kernel sat at 1.6% of the machine with `lg_throttle`
// dominating the stalls, because a hundred thousand vertices were funnelling
// their blocks through a few hundred nodes' worth of addresses.
//
// It now runs the way the data lies: one block owns one node, walks that node's
// bound vertices, and keeps its 6x6 block and 6-vector right-hand side in
// registers. The only write is the block's own result, so there is no atomic in
// the pass at all, and the block's contribution to a node is computed by the
// thread that will store it.
//
// The cost is that a vertex is visited once per node it binds to -- three times,
// typically, since the binding formula zeroes the farthest of its four slots --
// and its deformation and Jacobians are recomputed each time. That is vector
// arithmetic; the atomics it replaces were not.
// ---------------------------------------------------------------------------

// One thread's share of a node's block plus one row of the vertex's residuals.
// `jacobians` is compacted to the slots that belong to this node, which is
// normally one of them; the pair loop is what makes a vertex bound to the same
// node twice come out right.
__device__ inline void gather_row(float* block, float* right, int slots,
                                  const float* jacobians, float value) {
    for (int a = 0; a < slots; ++a) {
        const float* ja = jacobians + a * 6;
        for (int column = 0; column < 6; ++column) {
            right[column] -= ja[column] * value;
        }
        for (int b = 0; b < slots; ++b) {
            const float* jb = jacobians + b * 6;
            for (int row = 0; row < 6; ++row) {
                const float lhs = ja[row];
                for (int column = 0; column < 6; ++column) {
                    block[row * 6 + column] += lhs * jb[column];
                }
            }
        }
    }
}

// The node -> bound-vertex lists the gather walks. Built once per level, so the
// counting pass is the only atomic left on this path and it does not run per
// iteration. A vertex lands in the list of every node it binds to with a
// non-zero weight, which is what makes the gather cover the same rows the
// scatter did.
static __global__ void count_bound_kernel(
    const int* __restrict__ binding_nodes, const int* __restrict__ binding_counts,
    const float* __restrict__ binding_weights, int vertex_count,
    int* __restrict__ counts) {
    const int vertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex >= vertex_count) {
        return;
    }
    const int count = binding_counts[vertex];
    for (int slot = 0; slot < count; ++slot) {
        const int node = binding_nodes[vertex * 4 + slot];
        if (node < 0 || binding_weights[vertex * 4 + slot] == 0.0f) {
            continue;
        }
        atomicAdd(&counts[node], 1);
    }
}

static __global__ void scatter_bound_kernel(
    const int* __restrict__ binding_nodes, const int* __restrict__ binding_counts,
    const float* __restrict__ binding_weights, int vertex_count,
    int* __restrict__ cursor, int* __restrict__ out) {
    const int vertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex >= vertex_count) {
        return;
    }
    const int count = binding_counts[vertex];
    for (int slot = 0; slot < count; ++slot) {
        const int node = binding_nodes[vertex * 4 + slot];
        if (node < 0 || binding_weights[vertex * 4 + slot] == 0.0f) {
            continue;
        }
        out[atomicAdd(&cursor[node], 1)] = vertex;
    }
}

// ---------------------------------------------------------------------------
// The node-pair structure
//
// A vertex bound to several nodes contributes a cross block J_a^T J_b between
// every pair of those nodes. Route A has always assembled them (its
// `interactions_` map); the device path wrote only each node's own diagonal, so
// it was minimising a weaker Hessian -- the energies and the gradient were
// right, the curvature was not, and the missing part showed up as four to six
// times the inner iterations.
//
// The pairs are the same for every iteration of a level -- they come from the
// bindings -- so they are derived once: key each (vertex, slot pair) by its
// sorted node pair, sort, and run-length the keys into groups. A group is an
// "edge" with two nodes as far as the solve is concerned, so the groups are
// simply appended to the graph's edge list and the same block solver consumes
// them.
// ---------------------------------------------------------------------------

// One entry per pair of a vertex's slots that land on two different nodes; slot
// pairs that land on the same node belong to that node's own block and are
// handled by the gathered pass.
static __global__ void count_pairs_kernel(
    const int* __restrict__ binding_nodes, const int* __restrict__ binding_counts,
    const float* __restrict__ binding_weights, int vertex_count,
    int* __restrict__ counts) {
    const int vertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex >= vertex_count) {
        return;
    }
    int nodes[4];
    int slots = 0;
    const int count = binding_counts[vertex];
    for (int slot = 0; slot < count; ++slot) {
        const int node = binding_nodes[vertex * 4 + slot];
        if (node < 0 || binding_weights[vertex * 4 + slot] == 0.0f) {
            continue;
        }
        nodes[slots++] = node;
    }
    int pairs = 0;
    for (int a = 0; a < slots; ++a) {
        for (int b = a + 1; b < slots; ++b) {
            if (nodes[a] != nodes[b]) {
                ++pairs;
            }
        }
    }
    counts[vertex] = pairs;
}

static __global__ void emit_pairs_kernel(
    const int* __restrict__ binding_nodes, const int* __restrict__ binding_counts,
    const float* __restrict__ binding_weights, const int* __restrict__ begins,
    int vertex_count, int node_count, unsigned long long* __restrict__ keys,
    unsigned long long* __restrict__ payload) {
    const int vertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex >= vertex_count) {
        return;
    }
    int slot_of[4];
    int nodes[4];
    int slots = 0;
    const int count = binding_counts[vertex];
    for (int slot = 0; slot < count; ++slot) {
        const int node = binding_nodes[vertex * 4 + slot];
        if (node < 0 || binding_weights[vertex * 4 + slot] == 0.0f) {
            continue;
        }
        slot_of[slots] = slot;
        nodes[slots] = node;
        ++slots;
    }
    int at = begins[vertex];
    for (int a = 0; a < slots; ++a) {
        for (int b = a + 1; b < slots; ++b) {
            if (nodes[a] == nodes[b]) {
                continue;
            }
            const int low = nodes[a] < nodes[b] ? nodes[a] : nodes[b];
            const int high = nodes[a] < nodes[b] ? nodes[b] : nodes[a];
            const int low_slot = nodes[a] < nodes[b] ? slot_of[a] : slot_of[b];
            const int high_slot = nodes[a] < nodes[b] ? slot_of[b] : slot_of[a];
            keys[at] = static_cast<unsigned long long>(low) * node_count + high;
            payload[at] = (static_cast<unsigned long long>(vertex) << 4) |
                          (static_cast<unsigned long long>(low_slot) << 2) |
                          static_cast<unsigned long long>(high_slot);
            ++at;
        }
    }
}

// Marks the first entry of each run of equal keys, so an inclusive scan turns
// the keys into group indices.
static __global__ void mark_group_starts_kernel(
    const unsigned long long* __restrict__ keys, int count, int* __restrict__ flags) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    flags[index] = (index == 0 || keys[index] != keys[index - 1]) ? 1 : 0;
}

static __global__ void scatter_group_begin_kernel(
    const int* __restrict__ group_of_entry, int count, int* __restrict__ begins) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    if (index == 0 || group_of_entry[index] != group_of_entry[index - 1]) {
        // One writer per group, so a plain store is enough.
        begins[group_of_entry[index] - 1] = index;
    }
}

// The tail of the edge list: the groups' node pairs, which is all the block
// solver needs to know about them.
static __global__ void fill_pair_edges_kernel(
    const unsigned long long* __restrict__ keys, const int* __restrict__ begins,
    int group_count, int node_count, int* __restrict__ edge_nodes) {
    const int group = blockIdx.x * blockDim.x + threadIdx.x;
    if (group >= group_count) {
        return;
    }
    const unsigned long long key = keys[begins[group]];
    edge_nodes[group * 2 + 0] = static_cast<int>(key / node_count);
    edge_nodes[group * 2 + 1] = static_cast<int>(key % node_count);
}

// The cross blocks themselves, laid out like the gathered node pass: one block
// per group, its threads striding over the group's entries with a 6x6 in
// registers each, one reduction, one write. A single thread per group was the
// first attempt and it is the wrong shape -- there are far fewer groups than
// entries (a few thousand groups against a few hundred thousand entries at a
// hundred thousand vertices), so it left the device nearly idle while a handful
// of threads walked dozens of entries each.
//
// Only the block is written: a row's contribution to the gradient goes to its
// own node's right-hand side, which the gathered pass already did.
static __global__ void assemble_pair_kernel(
    const float* points, const float* node_rest, const int* binding_nodes,
    const float* binding_weights, const int* binding_counts, const Mat3f* rotations,
    const float* translations, const float* correspondence_points,
    const float* correspondence_normals, const unsigned char* rejection,
    const float* robust_weights, const unsigned char* mask,
    const unsigned long long* payload, const int* group_begin, int group_count,
    int entry_count, const float* landmark_flag, const float* landmark_targets,
    float point_weight, float plane_weight, float landmark_weight,
    int graph_edge_count, Mat6f* pair_blocks) {
    const int group = blockIdx.x;
    if (group >= group_count) {
        return;
    }
    const int begin = group_begin[group];
    const int end = group + 1 < group_count ? group_begin[group + 1] : entry_count;
    float block[kBlockValues];
    for (int i = 0; i < kBlockValues; ++i) {
        block[i] = 0.0f;
    }
    for (int index = begin + threadIdx.x; index < end; index += blockDim.x) {
        const unsigned long long packed = payload[index];
        const int vertex = static_cast<int>(packed >> 4);
        const int slot_a = static_cast<int>((packed >> 2) & 0x3);
        const int slot_b = static_cast<int>(packed & 0x3);
        const bool carry_data =
            rejection[vertex] == 0 && (mask == nullptr || mask[vertex] != 0);
        const Vec3f point = load_vec3(points, vertex);
        const int count = binding_counts[vertex];
        const float robust = robust_weights[vertex];
        Vec3f deformed;
        for (int slot = 0; slot < count; ++slot) {
            const int node = binding_nodes[vertex * 4 + slot];
            const float weight = binding_weights[vertex * 4 + slot];
            if (weight == 0.0f || node < 0) {
                continue;
            }
            const Vec3f rest = load_vec3(node_rest, node);
            deformed += (rotations[node] * (point - rest) + rest +
                         load_vec3(translations, node)) *
                        weight;
        }
        const int node_a = binding_nodes[vertex * 4 + slot_a];
        const int node_b = binding_nodes[vertex * 4 + slot_b];
        const float weight_a = binding_weights[vertex * 4 + slot_a];
        const float weight_b = binding_weights[vertex * 4 + slot_b];
        const Vec3f rotated_a =
            node_rotated_offset(rotations, node_rest, node_a, point);
        const Vec3f rotated_b =
            node_rotated_offset(rotations, node_rest, node_b, point);
        const Vec3f offset = deformed - load_vec3(correspondence_points, vertex);
        const Vec3f normal = load_vec3(correspondence_normals, vertex);

        // One row of the pair's outer product: A[a][b] += J_a^T J_b, which is
        // not symmetric on its own, so both halves of the group's two nodes are
        // produced by the same row rather than by transposing.
        const auto add_pair = [&](const Vec3f& lambda, float scale) {
            float jacobian_a[6];
            float jacobian_b[6];
            fill_linear_row(lambda, rotated_a, scale * weight_a, jacobian_a);
            fill_linear_row(lambda, rotated_b, scale * weight_b, jacobian_b);
            for (int row = 0; row < 6; ++row) {
                const float lhs = jacobian_a[row];
                for (int column = 0; column < 6; ++column) {
                    block[row * 6 + column] += lhs * jacobian_b[column];
                }
            }
        };
        if (carry_data) {
            if (point_weight > 0.0f) {
                const float scale = sqrtf(point_weight * robust);
                for (int row = 0; row < 3; ++row) {
                    const Vec3f axis = row == 0   ? Vec3f{1.0f, 0.0f, 0.0f}
                                       : row == 1 ? Vec3f{0.0f, 1.0f, 0.0f}
                                                  : Vec3f{0.0f, 0.0f, 1.0f};
                    add_pair(axis, scale);
                }
            }
            if (plane_weight > 0.0f) {
                add_pair(normal, sqrtf(plane_weight * robust));
            }
        }
        const float landmark =
            (landmark_flag == nullptr || landmark_flag[vertex] == 0.0f)
                ? 0.0f
                : landmark_weight;
        if (landmark > 0.0f) {
            const float scale = sqrtf(landmark);
            for (int row = 0; row < 3; ++row) {
                const Vec3f axis = row == 0   ? Vec3f{1.0f, 0.0f, 0.0f}
                                   : row == 1 ? Vec3f{0.0f, 1.0f, 0.0f}
                                              : Vec3f{0.0f, 0.0f, 1.0f};
                add_pair(axis, scale);
            }
        }
    }
    Mat6f& target = pair_blocks[graph_edge_count + group];
    __shared__ float staged[kGatherThreads][kBlockValues + 1];
    for (int i = 0; i < kBlockValues; ++i) {
        staged[threadIdx.x][i] = block[i];
    }
    __syncthreads();
    for (int span = blockDim.x / 2; span > 0; span >>= 1) {
        if (threadIdx.x < span) {
            for (int i = 0; i < kBlockValues; ++i) {
                staged[threadIdx.x][i] += staged[threadIdx.x + span][i];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        for (int i = 0; i < kBlockValues; ++i) {
            target.m[i] = staged[0][i];
        }
    }
}

static __global__ void assemble_node_kernel(
    const float* points, const float* node_rest, const int* binding_nodes,
    const float* binding_weights, const int* binding_counts, const Mat3f* rotations,
    const float* translations, const float* correspondence_points,
    const float* correspondence_normals, const unsigned char* rejection,
    const float* robust_weights, const unsigned char* mask, const int* gather_begin,
    const int* gather_vertex, const float* landmark_flag,
    const float* landmark_targets, float point_weight, float plane_weight,
    float landmark_weight, int node_count, Mat6f* diag_blocks, float* rhs) {
    const int node = blockIdx.x;
    if (node >= node_count) {
        return;
    }
    float block[kBlockValues];
    float right[6];
    for (int i = 0; i < kBlockValues; ++i) {
        block[i] = 0.0f;
    }
    for (int i = 0; i < 6; ++i) {
        right[i] = 0.0f;
    }

    const int begin = gather_begin[node];
    const int end = gather_begin[node + 1];
    for (int index = begin + threadIdx.x; index < end; index += blockDim.x) {
        const int vertex = gather_vertex[index];
        // The mask and the correspondence filters silence the *data* rows of a
        // vertex; a landmark marker on the same vertex is a separate constraint
        // and keeps its rows, which is what the separate landmark block did.
        const bool carry_data =
            rejection[vertex] == 0 && (mask == nullptr || mask[vertex] != 0);
        const Vec3f point = load_vec3(points, vertex);
        const int count = binding_counts[vertex];
        const float robust = robust_weights[vertex];

        // The vertex's slots, and which of them land on this node.
        Vec3f rotated[4];
        float weights[4];
        int own[4];
        int slots = 0;
        int own_count = 0;
        Vec3f deformed;
        for (int slot = 0; slot < count; ++slot) {
            const int other = binding_nodes[vertex * 4 + slot];
            const float weight = binding_weights[vertex * 4 + slot];
            if (other < 0 || weight == 0.0f) {
                continue;
            }
            const Vec3f rest = load_vec3(node_rest, other);
            deformed += (rotations[other] * (point - rest) + rest +
                         load_vec3(translations, other)) *
                        weight;
            rotated[slots] = node_rotated_offset(rotations, node_rest, other, point);
            weights[slots] = weight;
            if (other == node) {
                own[own_count++] = slots;
            }
            ++slots;
        }
        if (own_count == 0) {
            continue;
        }
        float jacobians[4][6];
        const Vec3f offset = deformed - load_vec3(correspondence_points, vertex);
        const Vec3f normal = load_vec3(correspondence_normals, vertex);

        if (carry_data && point_weight > 0.0f) {
            const float scale = sqrtf(point_weight * robust);
            for (int row = 0; row < 3; ++row) {
                const Vec3f axis = row == 0   ? Vec3f{1.0f, 0.0f, 0.0f}
                                   : row == 1 ? Vec3f{0.0f, 1.0f, 0.0f}
                                              : Vec3f{0.0f, 0.0f, 1.0f};
                for (int s = 0; s < own_count; ++s) {
                    fill_linear_row(axis, rotated[own[s]],
                                    scale * weights[own[s]], jacobians[s]);
                }
                const float value =
                    row == 0 ? offset.x : (row == 1 ? offset.y : offset.z);
                gather_row(block, right, own_count, &jacobians[0][0], scale * value);
            }
        }
        if (carry_data && plane_weight > 0.0f) {
            const float scale = sqrtf(plane_weight * robust);
            for (int s = 0; s < own_count; ++s) {
                fill_linear_row(normal, rotated[own[s]], scale * weights[own[s]],
                                jacobians[s]);
            }
            gather_row(block, right, own_count, &jacobians[0][0],
                       scale * dot(normal, offset));
        }

        // A landmark vertex carries an extra set of rows onto its own target,
        // with the level's landmark weight rather than the data weight.
        const float landmark =
            (landmark_flag == nullptr || landmark_flag[vertex] == 0.0f)
                ? 0.0f
                : landmark_weight;
        if (landmark > 0.0f) {
            const float scale = sqrtf(landmark);
            const Vec3f to_marker =
                deformed - load_vec3(landmark_targets, vertex);
            for (int row = 0; row < 3; ++row) {
                const Vec3f axis = row == 0   ? Vec3f{1.0f, 0.0f, 0.0f}
                                   : row == 1 ? Vec3f{0.0f, 1.0f, 0.0f}
                                              : Vec3f{0.0f, 0.0f, 1.0f};
                for (int s = 0; s < own_count; ++s) {
                    fill_linear_row(axis, rotated[own[s]],
                                    scale * weights[own[s]], jacobians[s]);
                }
                const float value = row == 0   ? to_marker.x
                                    : row == 1 ? to_marker.y
                                               : to_marker.z;
                gather_row(block, right, own_count, &jacobians[0][0], scale * value);
            }
        }
    }

    // One pad word per thread so the reduction's stride is odd, which keeps its
    // shared accesses off the same bank.
    __shared__ float staged[kGatherThreads][kBlockValues + 1];
    __shared__ float staged_right[kGatherThreads][7];
    for (int i = 0; i < kBlockValues; ++i) {
        staged[threadIdx.x][i] = block[i];
    }
    for (int i = 0; i < 6; ++i) {
        staged_right[threadIdx.x][i] = right[i];
    }
    __syncthreads();
    for (int span = blockDim.x / 2; span > 0; span >>= 1) {
        if (threadIdx.x < span) {
            for (int i = 0; i < kBlockValues; ++i) {
                staged[threadIdx.x][i] += staged[threadIdx.x + span][i];
            }
            for (int i = 0; i < 6; ++i) {
                staged_right[threadIdx.x][i] += staged_right[threadIdx.x + span][i];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        for (int i = 0; i < kBlockValues; ++i) {
            diag_blocks[node].m[i] = staged[0][i];
        }
        for (int i = 0; i < 6; ++i) {
            rhs[node * 6 + i] = staged_right[0][i];
        }
    }
}


// The data term's energies, one pass over the vertices. The right-hand side and
// the blocks are not built here any more: the gathered pass owns them, and this
// kernel's job is only the two numbers the accept/reject test and the level's
// convergence test look at.
//
// Those numbers used to be two `atomicAdd`s per vertex into two single
// addresses, which is a hundred thousand serialized updates per pass and showed
// up as one of the two most expensive kernels at 100k. They are reduced across
// the block first and added once per block instead, which is where the 250x
// drop in those atomics comes from. The early exits the earlier version had are
// gone with it, because every thread has to reach the block reduction.
static __global__ void energy_data_kernel(
    const float* points, const float* node_rest, const int* binding_nodes,
    const float* binding_weights, const int* binding_counts, const Mat3f* rotations,
    const float* translations, const float* correspondence_points,
    const float* correspondence_normals, const unsigned char* rejection,
    const float* robust_weights, const unsigned char* mask, int vertex_count,
    float plane_weight, float point_weight, float* energies) {
    const int vertex = blockIdx.x * blockDim.x + threadIdx.x;
    float point_energy = 0.0f;
    float plane_energy = 0.0f;
    const bool active = vertex < vertex_count && rejection[vertex] == 0 &&
                        (mask == nullptr || mask[vertex] != 0);
    if (active) {
        const Vec3f point = load_vec3(points, vertex);
        const int count = binding_counts[vertex];
        Vec3f deformed;
        for (int slot = 0; slot < count; ++slot) {
            const int node = binding_nodes[vertex * 4 + slot];
            const float weight = binding_weights[vertex * 4 + slot];
            if (weight == 0.0f || node < 0) {
                continue;
            }
            const Vec3f rest = load_vec3(node_rest, node);
            deformed += (rotations[node] * (point - rest) + rest +
                         load_vec3(translations, node)) *
                        weight;
        }
        const float robust = robust_weights[vertex];
        const Vec3f offset = deformed - load_vec3(correspondence_points, vertex);
        if (point_weight > 0.0f) {
            const Vec3f scaled = offset * sqrtf(point_weight * robust);
            point_energy = squared_norm(scaled);
        }
        if (plane_weight > 0.0f) {
            const Vec3f normal = load_vec3(correspondence_normals, vertex);
            const float value = sqrtf(plane_weight * robust) * dot(normal, offset);
            plane_energy = value * value;
        }
    }
    __shared__ float shared[2][kThreads];
    shared[0][threadIdx.x] = point_energy;
    shared[1][threadIdx.x] = plane_energy;
    __syncthreads();
    for (int span = blockDim.x / 2; span > 0; span >>= 1) {
        if (threadIdx.x < span) {
            shared[0][threadIdx.x] += shared[0][threadIdx.x + span];
            shared[1][threadIdx.x] += shared[1][threadIdx.x + span];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(&energies[0], shared[0][0]);
        atomicAdd(&energies[1], shared[1][0]);
    }
}

static __global__ void accumulate_smoothness_kernel(
    const float* node_rest, const int* edge_nodes, int edge_count,
    const Mat3f* rotations, const float* translations, float weight,
    bool with_jacobians, Mat6f* diag_blocks, float* rhs, Mat6f* edge_blocks,
    float* energies) {
    const int edge = blockIdx.x * blockDim.x + threadIdx.x;
    if (edge >= edge_count) {
        return;
    }
    const int node_a = edge_nodes[edge * 2 + 0];
    const int node_b = edge_nodes[edge * 2 + 1];
    const Vec3f rest_a = load_vec3(node_rest, node_a);
    const Vec3f rest_b = load_vec3(node_rest, node_b);
    const Vec3f translation_a = load_vec3(translations, node_a);
    const Vec3f translation_b = load_vec3(translations, node_b);
    const Vec3f field_a =
        rotations[node_a] * (rest_b - rest_a) + rest_a + translation_a;
    const Vec3f field_b = rest_b + translation_b;
    const float scale = sqrtf(weight);
    const Vec3f difference = (field_a - field_b) * scale;
    atomicAdd(&energies[2], squared_norm(difference));
    if (!with_jacobians) {
        return;
    }
    const Vec3f rotated = rotations[node_a] * (rest_b - rest_a);
    const Vec3f axes[3] = {
        {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}};
    float* block_aa = &diag_blocks[node_a].m[0];
    float* block_bb = &diag_blocks[node_b].m[0];
    float* block_ab = &edge_blocks[edge].m[0];
    for (int row = 0; row < 3; ++row) {
        float jacobian_a[6];
        float jacobian_b[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        fill_linear_row(axes[row], rotated, scale, jacobian_a);
        jacobian_b[row + 3] = -scale;
        const float value =
            row == 0 ? difference.x : (row == 1 ? difference.y : difference.z);
        for (int column = 0; column < 6; ++column) {
            atomicAdd(&rhs[node_a * 6 + column], -jacobian_a[column] * value);
            atomicAdd(&rhs[node_b * 6 + column], -jacobian_b[column] * value);
        }
        for (int a = 0; a < 6; ++a) {
            for (int b = 0; b < 6; ++b) {
                atomicAdd(&block_aa[a * 6 + b], jacobian_a[a] * jacobian_a[b]);
                atomicAdd(&block_bb[a * 6 + b], jacobian_b[a] * jacobian_b[b]);
                // The pair block is written by this edge alone, so it needs no
                // atomics: one edge owns exactly one block.
                block_ab[a * 6 + b] += jacobian_a[a] * jacobian_b[b];
            }
        }
    }
}

static __global__ void accumulate_anchor_kernel(
    int node_count, const Mat3f* rotations, const float* translations, float weight,
    bool with_jacobians, Mat6f* diag_blocks, float* rhs, float* energies) {
    const int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= node_count) {
        return;
    }
    const Vec3f omega = gpu::log_so3(rotations[node]);
    const Vec3f translation = load_vec3(translations, node);
    const float scale = sqrtf(weight);
    float values[6] = {scale * omega.x, scale * omega.y, scale * omega.z,
                       scale * translation.x, scale * translation.y,
                       scale * translation.z};
    float sum = 0.0f;
    for (int i = 0; i < 6; ++i) {
        sum += values[i] * values[i];
    }
    atomicAdd(&energies[3], sum);
    if (!with_jacobians) {
        return;
    }
    const Mat3f rotation_jacobian = gpu::left_jacobian_inverse(omega);
    float jacobian[6][6] = {};
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            jacobian[row][column] = scale * rotation_jacobian.m[row * 3 + column];
        }
    }
    for (int row = 0; row < 6; ++row) {
        jacobian[row + 3][row + 3] = scale;
    }
    float* block = &diag_blocks[node].m[0];
    for (int column = 0; column < 6; ++column) {
        float gradient = 0.0f;
        for (int k = 0; k < 6; ++k) {
            gradient += jacobian[k][column] * values[k];
        }
        atomicAdd(&rhs[node * 6 + column], -gradient);
    }
    for (int row = 0; row < 6; ++row) {
        for (int column = 0; column < 6; ++column) {
            float outer = 0.0f;
            for (int k = 0; k < 6; ++k) {
                outer += jacobian[k][row] * jacobian[k][column];
            }
            atomicAdd(&block[row * 6 + column], outer);
        }
    }
}

static __global__ void accumulate_landmark_kernel(
    const float* points, const float* node_rest, const int* binding_nodes,
    const float* binding_weights, const int* binding_counts, const Mat3f* rotations,
    const float* translations, const int* landmark_vertices,
    const float* landmark_targets, int landmark_count, float weight,
    bool with_jacobians, Mat6f* diag_blocks, float* rhs, float* energies) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= landmark_count) {
        return;
    }
    const int vertex = landmark_vertices[index];
    if (vertex < 0) {
        return;
    }
    const Vec3f point = load_vec3(points, vertex);
    const Vec3f target = load_vec3(landmark_targets, index);
    const int count = binding_counts[vertex];
    Vec3f deformed;
    for (int slot = 0; slot < count; ++slot) {
        const int node = binding_nodes[vertex * 4 + slot];
        const float slot_weight = binding_weights[vertex * 4 + slot];
        if (slot_weight == 0.0f || node < 0) {
            continue;
        }
        const Vec3f rest = load_vec3(node_rest, node);
        deformed += (rotations[node] * (point - rest) + rest +
                     load_vec3(translations, node)) *
                    slot_weight;
    }
    const float scale = sqrtf(weight);
    const Vec3f scaled = (deformed - target) * scale;
    atomicAdd(&energies[4], squared_norm(scaled));
    if (!with_jacobians) {
        return;
    }
    for (int row = 0; row < 3; ++row) {
        const Vec3f axis = row == 0   ? Vec3f{1.0f, 0.0f, 0.0f}
                           : row == 1 ? Vec3f{0.0f, 1.0f, 0.0f}
                                      : Vec3f{0.0f, 0.0f, 1.0f};
        const float value = row == 0 ? scaled.x : (row == 1 ? scaled.y : scaled.z);
        for (int slot = 0; slot < count; ++slot) {
            const int node = binding_nodes[vertex * 4 + slot];
            const float slot_weight = binding_weights[vertex * 4 + slot];
            if (slot_weight == 0.0f || node < 0) {
                continue;
            }
            float jacobian[6];
            fill_linear_row(
                axis, node_rotated_offset(rotations, node_rest, node, point),
                scale * slot_weight, jacobian);
            for (int column = 0; column < 6; ++column) {
                atomicAdd(&rhs[node * 6 + column], -jacobian[column] * value);
            }
            float* block = &diag_blocks[node].m[0];
            for (int a = 0; a < 6; ++a) {
                for (int b = 0; b < 6; ++b) {
                    atomicAdd(&block[a * 6 + b], jacobian[a] * jacobian[b]);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The linear solve: block-diagonal preconditioner and one cooperative PCG kernel
// ---------------------------------------------------------------------------

static __global__ void factor_kernel(
    const Mat6f* diagonal, float damping, bool diagonal_scaling, Mat6f* factor,
    int node_count) {
    const int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= node_count) {
        return;
    }
    Mat6f block = diagonal[node];
    float jitter = 0.0f;
    for (int attempt = 0; attempt < 8; ++attempt) {
        Mat6f candidate = block;
        for (int row = 0; row < 6; ++row) {
            const float scale =
                diagonal_scaling ? fmaxf(fabsf(block.at(row, row)), 1e-6f) : 1.0f;
            candidate.at(row, row) += damping * scale + jitter;
        }
        if (gpu::cholesky6(candidate)) {
            factor[node] = candidate;
            return;
        }
        jitter = jitter == 0.0f ? 1e-6f : jitter * 10.0f;
    }
    // Last resort: identity, which keeps the iteration finite.
    factor[node] = Mat6f{};
}

// The whole damped linear solve as one kernel.
//
// The grid is cooperative: every block stays resident and the iteration is
// separated by grid-wide barriers, so the recurrence never returns to the host
// and a solve costs a single launch. Each iteration runs four barriers, in this
// order:
//
//   A  after the diagonal product, before the edge contributions are added to it
//   B  after the edge contributions, which is also where p.Ap is complete
//   C  after the two inner products, which is where beta and the residual exist
//   D  after the direction update, before the next iteration reads it
//
// Every elementwise pass is spread over the grid with a fixed stride, so a node
// belongs to exactly one thread for the whole solve; the edge pass is the only
// part that needs atomics, and it owns the image vector's off-diagonal half.
static __global__ void pcg_kernel(
    const Mat6f* __restrict__ diagonal, const Mat6f* __restrict__ factor,
    const Mat6f* __restrict__ edge_blocks, const int* __restrict__ edge_nodes,
    int edge_count, const float* __restrict__ rhs, float damping,
    bool diagonal_scaling, int node_count, float tolerance, int max_iterations,
    float* __restrict__ solution, float* __restrict__ workspace_base,
    PcgState* state) {
    const cg::grid_group grid = cg::this_grid();
    const PcgWorkspace workspace = PcgWorkspace::of(workspace_base, node_count);
    const int stride = gridDim.x * blockDim.x;
    const int thread = blockIdx.x * blockDim.x + threadIdx.x;
    // The two scalars the setup pass produces live past the per-iteration slots,
    // so that the first iteration's slots do not overwrite them.
    float* const seed = workspace.reduce + 3 * max_iterations;

    // ---- r = b, x = 0, z = M^-1 r, p = z, plus ||b|| and the starting r.z ----
    float rhs_squared = 0.0f;
    float initial_rz = 0.0f;
    for (int node = thread; node < node_count; node += stride) {
        float residual[6];
        float preconditioned[6];
        for (int i = 0; i < 6; ++i) {
            residual[i] = rhs[node * 6 + i];
            solution[node * 6 + i] = 0.0f;
        }
        gpu::solve_cholesky6(factor[node], residual, preconditioned);
        for (int i = 0; i < 6; ++i) {
            workspace.residual[node * 6 + i] = residual[i];
            workspace.direction[node * 6 + i] = preconditioned[i];
            workspace.preconditioned[node * 6 + i] = preconditioned[i];
            rhs_squared += residual[i] * residual[i];
            initial_rz += residual[i] * preconditioned[i];
        }
    }
    const float rhs_total = block_sum(rhs_squared);
    const float rz_total = block_sum(initial_rz);
    if (threadIdx.x == 0) {
        atomicAdd(&seed[0], rhs_total);
        atomicAdd(&seed[1], rz_total);
    }
    grid.sync();

    const float rhs_norm = sqrtf(fmaxf(seed[0], 0.0f));
    if (rhs_norm <= 0.0f) {
        // An already-solved system: report success without dividing by zero.
        if (thread == 0) {
            state->converged = 1.0f;
            state->stop = 1.0f;
        }
        return;
    }
    float rz = seed[1];
    int completed = 0;
    float residual_norm = 1.0f;

    for (int iteration = 0; iteration < max_iterations; ++iteration) {
        float* const slot = workspace.reduce + 3 * iteration;

        // ---- image = (D + damping) * direction, with the diagonal part only ----
        float local_dot = 0.0f;
        for (int node = thread; node < node_count; node += stride) {
            float direction[6];
            float image[6];
            for (int i = 0; i < 6; ++i) {
                direction[i] = workspace.direction[node * 6 + i];
            }
            for (int row = 0; row < 6; ++row) {
                float sum = 0.0f;
                for (int column = 0; column < 6; ++column) {
                    float value = diagonal[node].at(row, column);
                    if (row == column) {
                        const float scale =
                            diagonal_scaling
                                ? fmaxf(fabsf(diagonal[node].at(row, row)), 1e-6f)
                                : 1.0f;
                        value += damping * scale;
                    }
                    sum += value * direction[column];
                }
                image[row] = sum;
                local_dot += direction[row] * sum;
            }
            for (int i = 0; i < 6; ++i) {
                workspace.image[node * 6 + i] = image[i];
            }
        }
        grid.sync();  // A: the diagonal product is in place

        // ---- add the off-diagonal blocks, and finish p.Ap on the way ----
        for (int edge = thread; edge < edge_count; edge += stride) {
            const int node_a = edge_nodes[edge * 2 + 0];
            const int node_b = edge_nodes[edge * 2 + 1];
            const Mat6f block = edge_blocks[edge];
            float direction_a[6];
            float direction_b[6];
            for (int i = 0; i < 6; ++i) {
                direction_a[i] = workspace.direction[node_a * 6 + i];
                direction_b[i] = workspace.direction[node_b * 6 + i];
            }
            for (int row = 0; row < 6; ++row) {
                float sum_a = 0.0f;
                float sum_b = 0.0f;
                for (int column = 0; column < 6; ++column) {
                    sum_a += block.at(row, column) * direction_b[column];
                    sum_b += block.at(column, row) * direction_a[column];
                }
                // The coupling is symmetric, so summing the image's own quadratic
                // form over the edges counts each pair exactly once.
                local_dot += direction_a[row] * sum_a + direction_b[row] * sum_b;
                atomicAdd(&workspace.image[node_a * 6 + row], sum_a);
                atomicAdd(&workspace.image[node_b * 6 + row], sum_b);
            }
        }
        const float dot_total = block_sum(local_dot);
        if (threadIdx.x == 0) {
            atomicAdd(slot, dot_total);
        }
        grid.sync();  // B: the image and p.Ap are complete

        const float p_ap = slot[0];
        if (fabsf(p_ap) < 1e-30f) {
            // The direction is numerically orthogonal to its image: the step is
            // meaningless, so stop without taking one.
            break;
        }
        const float alpha = rz / p_ap;

        // ---- x += alpha p, r -= alpha image, z = M^-1 r, and r.r, r.z ----
        float local_rr = 0.0f;
        float local_rz = 0.0f;
        for (int node = thread; node < node_count; node += stride) {
            float residual[6];
            float preconditioned[6];
            for (int i = 0; i < 6; ++i) {
                residual[i] = workspace.residual[node * 6 + i] -
                              alpha * workspace.image[node * 6 + i];
                solution[node * 6 + i] += alpha * workspace.direction[node * 6 + i];
            }
            gpu::solve_cholesky6(factor[node], residual, preconditioned);
            for (int i = 0; i < 6; ++i) {
                workspace.residual[node * 6 + i] = residual[i];
                workspace.preconditioned[node * 6 + i] = preconditioned[i];
                local_rr += residual[i] * residual[i];
                local_rz += residual[i] * preconditioned[i];
            }
        }
        const float rr_total = block_sum(local_rr);
        const float rz_total_next = block_sum(local_rz);
        if (threadIdx.x == 0) {
            atomicAdd(&slot[1], rr_total);
            atomicAdd(&slot[2], rz_total_next);
        }
        grid.sync();  // C: the inner products are complete

        const float rz_next = slot[2];
        residual_norm = sqrtf(fmaxf(slot[1], 0.0f)) / rhs_norm;
        completed = iteration + 1;
        if (residual_norm <= tolerance || completed >= max_iterations) {
            break;
        }
        const float beta = fabsf(rz) > 1e-30f ? rz_next / rz : 0.0f;
        rz = rz_next;
        for (int node = thread; node < node_count; node += stride) {
            for (int i = 0; i < 6; ++i) {
                workspace.direction[node * 6 + i] =
                    workspace.preconditioned[node * 6 + i] +
                    beta * workspace.direction[node * 6 + i];
            }
        }
        grid.sync();  // D: the direction is complete before the next product
    }

    if (thread == 0) {
        state->iteration_count = completed;
        state->residual = residual_norm;
        state->converged = residual_norm <= tolerance ? 1.0f : 0.0f;
        state->stop = 1.0f;
    }
}

// ---------------------------------------------------------------------------
// Host-side plumbing
// ---------------------------------------------------------------------------

thrust::device_vector<float> upload_vectors(const std::vector<Vec3>& values) {
    thrust::device_vector<float> out(values.size() * 3);
    std::vector<float> flat(values.size() * 3);
    for (std::size_t i = 0; i < values.size(); ++i) {
        flat[i * 3 + 0] = static_cast<float>(values[i].x);
        flat[i * 3 + 1] = static_cast<float>(values[i].y);
        flat[i * 3 + 2] = static_cast<float>(values[i].z);
    }
    if (!flat.empty()) {
        out.assign(flat.begin(), flat.end());
    }
    return out;
}

std::vector<Vec3> download_vectors(const thrust::device_vector<float>& values) {
    std::vector<float> flat(values.size());
    if (!flat.empty()) {
        thrust::copy(values.begin(), values.end(), flat.begin());
    }
    std::vector<Vec3> out(flat.size() / 3);
    for (std::size_t i = 0; i < out.size(); ++i) {
        out[i] = {flat[i * 3 + 0], flat[i * 3 + 1], flat[i * 3 + 2]};
    }
    return out;
}

thrust::device_vector<int> upload_ints(const std::vector<int>& values) {
    thrust::device_vector<int> out(values.size());
    if (!values.empty()) {
        out.assign(values.begin(), values.end());
    }
    return out;
}

thrust::device_vector<float> upload_floats(const std::vector<float>& values) {
    thrust::device_vector<float> out(values.size());
    if (!values.empty()) {
        out.assign(values.begin(), values.end());
    }
    return out;
}

std::vector<int> flatten_binding_nodes(const std::vector<VertexBinding>& bindings) {
    std::vector<int> flat(bindings.size() * 4);
    for (std::size_t i = 0; i < bindings.size(); ++i) {
        for (int slot = 0; slot < 4; ++slot) {
            flat[i * 4 + slot] = bindings[i].nodes[slot];
        }
    }
    return flat;
}

std::vector<float> flatten_binding_weights(
    const std::vector<VertexBinding>& bindings) {
    std::vector<float> flat(bindings.size() * 4);
    for (std::size_t i = 0; i < bindings.size(); ++i) {
        for (int slot = 0; slot < 4; ++slot) {
            flat[i * 4 + slot] = static_cast<float>(bindings[i].weights[slot]);
        }
    }
    return flat;
}

// Everything one level needs on the device. The transforms live here for the
// whole level: an accepted step is a kernel, a rejected one is a device-to-device
// copy of the backup, and the host only sees the energies that decide between
// them.
struct GpuLevel {
    int node_count = 0;
    int edge_count = 0;
    thrust::device_vector<float> node_rest;
    thrust::device_vector<int> binding_nodes;
    thrust::device_vector<float> binding_weights;
    thrust::device_vector<int> binding_counts;
    thrust::device_vector<int> edges;
    // The node -> bound-vertex lists the gathered accumulation walks, plus the
    // scratch the counting pass and the scatter need. Sized for every vertex
    // binding all four of its slots, which is the most any level can produce.
    thrust::device_vector<int> gather_begin;
    thrust::device_vector<int> gather_cursor;
    thrust::device_vector<int> gather_counts;
    thrust::device_vector<int> gather_vertex;
    // The node pairs that share a bound vertex. A vertex bound to several nodes
    // puts a cross block between every pair of those nodes into the system, and
    // one entry per (vertex, slot pair) with the same node pair is one group.
    // The groups are appended to the edge list, because a group *is* an edge with
    // two nodes as far as the solve is concerned.
    thrust::device_vector<unsigned long long> pair_keys;
    thrust::device_vector<unsigned long long> pair_payload;
    thrust::device_vector<int> pair_vertex_begin;
    thrust::device_vector<int> pair_flags;
    thrust::device_vector<int> pair_group_of_entry;
    thrust::device_vector<int> pair_group_begin;
    int pair_count = 0;        // entries emitted by this level
    int pair_group_count = 0;  // distinct node pairs
    int total_edge_count = 0;  // graph edges followed by the pair groups
    thrust::device_vector<Mat3f> rotations;
    thrust::device_vector<float> translations;
    thrust::device_vector<Mat3f> rotations_backup;
    thrust::device_vector<float> translations_backup;
    thrust::device_vector<float> deformed;
    thrust::device_vector<float> query_normals;
    thrust::device_vector<float> correspondence_points;
    thrust::device_vector<float> correspondence_normals;
    thrust::device_vector<float> correspondence_distances;
    thrust::device_vector<int> correspondence_triangles;
    thrust::device_vector<unsigned char> rejection;
    thrust::device_vector<float> robust_weights;
    thrust::device_vector<Mat6f> diagonal;
    thrust::device_vector<Mat6f> edge_blocks;
    thrust::device_vector<Mat6f> factor;
    thrust::device_vector<float> rhs;
    thrust::device_vector<float> delta;
    thrust::device_vector<float> pcg_workspace;
    thrust::device_vector<PcgState> pcg;
    thrust::device_vector<float> energies;
    thrust::device_vector<int> rejection_counts;
    thrust::device_vector<float> distance_sum;
    thrust::device_vector<float> sorted_distances;
    thrust::device_vector<float> statistics;
};

// Sizes every buffer for the largest level of the schedule. A session allocates
// once and reuses these for all its levels and all its steps: rebuilding them
// per level would free and reallocate each time, and every free synchronizes the
// device.
void reserve_level(GpuLevel& state, int max_nodes, int max_edges, int vertex_count,
                   int pcg_max_iterations) {
    max_nodes = std::max(max_nodes, 1);
    max_edges = std::max(max_edges, 1);
    state.node_rest = thrust::device_vector<float>(max_nodes * 3);
    state.binding_nodes = thrust::device_vector<int>(vertex_count * 4);
    state.binding_weights = thrust::device_vector<float>(vertex_count * 4);
    state.binding_counts = thrust::device_vector<int>(vertex_count);
    state.edges = thrust::device_vector<int>(max_edges * 2);
    state.gather_begin = thrust::device_vector<int>(max_nodes + 1);
    state.gather_cursor = thrust::device_vector<int>(max_nodes);
    state.gather_counts = thrust::device_vector<int>(max_nodes);
    state.gather_vertex = thrust::device_vector<int>(vertex_count * 4);
    // A vertex has at most six slot pairs, so this is the most entries any level
    // can emit and the most groups it can produce.
    const std::size_t pair_capacity = static_cast<std::size_t>(vertex_count) * 6;
    state.pair_keys = thrust::device_vector<unsigned long long>(pair_capacity);
    state.pair_payload = thrust::device_vector<unsigned long long>(pair_capacity);
    state.pair_vertex_begin = thrust::device_vector<int>(vertex_count + 1);
    state.pair_flags = thrust::device_vector<int>(pair_capacity);
    state.pair_group_of_entry = thrust::device_vector<int>(pair_capacity);
    state.pair_group_begin = thrust::device_vector<int>(pair_capacity + 1);
    state.rotations = thrust::device_vector<Mat3f>(max_nodes);
    state.translations = thrust::device_vector<float>(max_nodes * 3);
    state.rotations_backup = thrust::device_vector<Mat3f>(max_nodes);
    state.translations_backup = thrust::device_vector<float>(max_nodes * 3);
    state.deformed = thrust::device_vector<float>(vertex_count * 3);
    state.query_normals = thrust::device_vector<float>(vertex_count * 3);
    state.correspondence_points = thrust::device_vector<float>(vertex_count * 3);
    state.correspondence_normals = thrust::device_vector<float>(vertex_count * 3);
    state.correspondence_distances = thrust::device_vector<float>(vertex_count);
    state.correspondence_triangles = thrust::device_vector<int>(vertex_count);
    state.rejection = thrust::device_vector<unsigned char>(vertex_count, 0);
    state.robust_weights = thrust::device_vector<float>(vertex_count, 1.0f);
    state.diagonal = thrust::device_vector<Mat6f>(max_nodes);
    state.edge_blocks = thrust::device_vector<Mat6f>(max_edges);
    state.factor = thrust::device_vector<Mat6f>(max_nodes);
    state.rhs = thrust::device_vector<float>(static_cast<std::size_t>(max_nodes) * 6);
    state.delta = thrust::device_vector<float>(static_cast<std::size_t>(max_nodes) * 6);
    state.pcg_workspace = thrust::device_vector<float>(
        PcgWorkspace::floats_for(max_nodes, pcg_max_iterations));
    state.pcg = thrust::device_vector<PcgState>(1);
    state.energies = thrust::device_vector<float>(kEnergyKinds);
    state.rejection_counts = thrust::device_vector<int>(kRejectionKinds);
    state.distance_sum = thrust::device_vector<float>(1);
    state.sorted_distances = thrust::device_vector<float>(vertex_count);
    state.statistics = thrust::device_vector<float>(3);
}

// Loads one level's own data into the reserved buffers and rebuilds the
// node -> bound-vertex lists. Assigning a smaller count than the buffer holds
// keeps the storage, so this never reallocates; the gather is per level because
// the bindings are, and rebuilding it costs two passes over the vertices rather
// than one per iteration.
void load_level(GpuLevel& state, const GraphLevel& level, int vertex_count,
                cudaStream_t stream) {
    state.node_count = level.node_count();
    state.edge_count = static_cast<int>(level.edges.size());
    std::vector<float> node_rest(level.node_rest_positions.size() * 3);
    for (std::size_t i = 0; i < level.node_rest_positions.size(); ++i) {
        node_rest[i * 3 + 0] = static_cast<float>(level.node_rest_positions[i].x);
        node_rest[i * 3 + 1] = static_cast<float>(level.node_rest_positions[i].y);
        node_rest[i * 3 + 2] = static_cast<float>(level.node_rest_positions[i].z);
    }
    state.node_rest.assign(node_rest.begin(), node_rest.end());
    const std::vector<int> binding_nodes = flatten_binding_nodes(level.bindings);
    const std::vector<float> binding_weights = flatten_binding_weights(level.bindings);
    std::vector<int> counts(level.bindings.size());
    for (std::size_t i = 0; i < level.bindings.size(); ++i) {
        counts[i] = level.bindings[i].count;
    }
    state.binding_nodes.assign(binding_nodes.begin(), binding_nodes.end());
    state.binding_weights.assign(binding_weights.begin(), binding_weights.end());
    state.binding_counts.assign(counts.begin(), counts.end());
    std::vector<int> edge_list(static_cast<std::size_t>(state.edge_count) * 2);
    for (int edge = 0; edge < state.edge_count; ++edge) {
        edge_list[edge * 2 + 0] = level.edges[edge].a;
        edge_list[edge * 2 + 1] = level.edges[edge].b;
    }
    state.edges.assign(edge_list.begin(), edge_list.end());

    const auto par = thrust::cuda::par.on(stream);
    thrust::fill(par, state.gather_counts.begin(),
                 state.gather_counts.begin() + state.node_count, 0);
    count_bound_kernel<<<blocks_for(vertex_count), kThreads, 0, stream>>>(
        thrust::raw_pointer_cast(state.binding_nodes.data()),
        thrust::raw_pointer_cast(state.binding_counts.data()),
        thrust::raw_pointer_cast(state.binding_weights.data()), vertex_count,
        thrust::raw_pointer_cast(state.gather_counts.data()));
    throw_on_cuda_error(cudaGetLastError());
    thrust::fill(par, state.gather_begin.begin(), state.gather_begin.begin() + 1, 0);
    thrust::inclusive_scan(par, state.gather_counts.begin(),
                           state.gather_counts.begin() + state.node_count,
                           state.gather_begin.begin() + 1);
    thrust::copy(par, state.gather_begin.begin(),
                 state.gather_begin.begin() + state.node_count,
                 state.gather_cursor.begin());
    scatter_bound_kernel<<<blocks_for(vertex_count), kThreads, 0, stream>>>(
        thrust::raw_pointer_cast(state.binding_nodes.data()),
        thrust::raw_pointer_cast(state.binding_counts.data()),
        thrust::raw_pointer_cast(state.binding_weights.data()), vertex_count,
        thrust::raw_pointer_cast(state.gather_cursor.data()),
        thrust::raw_pointer_cast(state.gather_vertex.data()));
    throw_on_cuda_error(cudaGetLastError());

    // ---- the node-pair groups ----
    const int* bound_nodes = thrust::raw_pointer_cast(state.binding_nodes.data());
    const int* bound_counts = thrust::raw_pointer_cast(state.binding_counts.data());
    const float* bound_weights =
        thrust::raw_pointer_cast(state.binding_weights.data());
    int* vertex_begin = thrust::raw_pointer_cast(state.pair_vertex_begin.data());
    unsigned long long* keys = thrust::raw_pointer_cast(state.pair_keys.data());
    unsigned long long* payload = thrust::raw_pointer_cast(state.pair_payload.data());
    int* flags = thrust::raw_pointer_cast(state.pair_flags.data());
    int* group_of_entry = thrust::raw_pointer_cast(state.pair_group_of_entry.data());
    int* group_begin = thrust::raw_pointer_cast(state.pair_group_begin.data());

    thrust::fill(par, state.pair_flags.begin(),
                 state.pair_flags.begin() + vertex_count, 0);
    count_pairs_kernel<<<blocks_for(vertex_count), kThreads, 0, stream>>>(
        bound_nodes, bound_counts, bound_weights, vertex_count, flags);
    throw_on_cuda_error(cudaGetLastError());
    thrust::fill(par, state.pair_vertex_begin.begin(),
                 state.pair_vertex_begin.begin() + 1, 0);
    thrust::inclusive_scan(par, state.pair_flags.begin(),
                           state.pair_flags.begin() + vertex_count,
                           state.pair_vertex_begin.begin() + 1);
    // The one number the host needs here, because it sizes the arrays: how many
    // entries the level produced.
    throw_on_cuda_error(cudaMemcpyAsync(&state.pair_count, vertex_begin + vertex_count,
                                        sizeof(int), cudaMemcpyDeviceToHost, stream));
    throw_on_cuda_error(cudaStreamSynchronize(stream));
    state.pair_group_count = 0;
    if (state.pair_count > 0) {
        emit_pairs_kernel<<<blocks_for(vertex_count), kThreads, 0, stream>>>(
            bound_nodes, bound_counts, bound_weights, vertex_begin, vertex_count,
            state.node_count, keys, payload);
        throw_on_cuda_error(cudaGetLastError());
        thrust::sort_by_key(par, state.pair_keys.begin(),
                            state.pair_keys.begin() + state.pair_count,
                            state.pair_payload.begin());
        mark_group_starts_kernel<<<blocks_for(state.pair_count), kThreads, 0, stream>>>(
            keys, state.pair_count, flags);
        throw_on_cuda_error(cudaGetLastError());
        thrust::inclusive_scan(par, state.pair_flags.begin(),
                               state.pair_flags.begin() + state.pair_count,
                               state.pair_group_of_entry.begin());
        scatter_group_begin_kernel<<<blocks_for(state.pair_count), kThreads, 0,
                                     stream>>>(group_of_entry, state.pair_count,
                                               group_begin);
        throw_on_cuda_error(cudaGetLastError());
        int last_group = 0;
        throw_on_cuda_error(cudaMemcpyAsync(&last_group,
                                            group_of_entry + state.pair_count - 1,
                                            sizeof(int), cudaMemcpyDeviceToHost, stream));
        throw_on_cuda_error(cudaStreamSynchronize(stream));
        state.pair_group_count = last_group;
    }
    state.total_edge_count = state.edge_count + state.pair_group_count;

    // The groups are appended to the edge list: to the block solver they are
    // edges like any other, which is what lets it consume them unchanged.
    state.edges.resize(static_cast<std::size_t>(state.total_edge_count) * 2);
    state.edge_blocks.resize(std::max(state.total_edge_count, 1));
    if (state.pair_group_count > 0) {
        fill_pair_edges_kernel<<<blocks_for(state.pair_group_count), kThreads, 0,
                                 stream>>>(
            keys, group_begin, state.pair_group_count, state.node_count,
            thrust::raw_pointer_cast(state.edges.data()) + state.edge_count * 2);
        throw_on_cuda_error(cudaGetLastError());
    }
}

// A pinned staging buffer for the handful of scalars the schedule has to see.
// Control flow -- is this step lowering the energy, has the solve converged --
// cannot be decided without them, but staging through pinned memory keeps each
// one a single transfer instead of a pageable copy that also drains the stream.
class PinnedFloats {
public:
    explicit PinnedFloats(int capacity) {
        throw_on_cuda_error(
            cudaMallocHost(reinterpret_cast<void**>(&data_), sizeof(float) * capacity));
    }
    ~PinnedFloats() {
        if (data_ != nullptr) {
            cudaFreeHost(data_);
        }
    }
    PinnedFloats(const PinnedFloats&) = delete;
    PinnedFloats& operator=(const PinnedFloats&) = delete;
    float* data() { return data_; }
    float at(int index) const { return data_[index]; }

private:
    float* data_ = nullptr;
};

// The only place the schedule waits for the device. Everything a decision needs
// has been reduced to a handful of scalars by then.
void read_back(
    const thrust::device_vector<float>& source, int count, PinnedFloats& destination,
    cudaStream_t stream) {
    if (count <= 0 || source.size() == 0) {
        return;
    }
    throw_on_cuda_error(cudaMemcpyAsync(
        destination.data(), thrust::raw_pointer_cast(source.data()),
        sizeof(float) * static_cast<std::size_t>(count), cudaMemcpyDeviceToHost,
        stream));
    throw_on_cuda_error(cudaStreamSynchronize(stream));
}

// The stream the whole route runs on. One stream means the device work is one
// ordered pipeline, so the only synchronization the schedule pays for is the one
// it asks for.
//
// It is a *blocking* stream rather than a non-blocking one, and that is
// deliberate: the thrust upload helpers (device_vector::assign, the
// device-to-host copies that end a level) run on the legacy default stream,
// which only orders itself against blocking streams. With a non-blocking stream
// those copies would be free to overtake the kernels that produce their input.
class StreamGuard {
public:
    StreamGuard() {
        throw_on_cuda_error(cudaStreamCreate(&stream_));
    }
    ~StreamGuard() {
        if (stream_ != nullptr) {
            cudaStreamDestroy(stream_);
        }
    }
    StreamGuard(const StreamGuard&) = delete;
    StreamGuard& operator=(const StreamGuard&) = delete;
    cudaStream_t get() const { return stream_; }

private:
    cudaStream_t stream_ = nullptr;
};

void upload_state(
    const Deformation& deformation, thrust::device_vector<Mat3f>& rotations,
    thrust::device_vector<float>& translations) {
    std::vector<Mat3f> host_rotations(deformation.nodes.size());
    std::vector<float> host_translations(deformation.nodes.size() * 3);
    for (std::size_t i = 0; i < deformation.nodes.size(); ++i) {
        for (int k = 0; k < 9; ++k) {
            host_rotations[i].m[k] =
                static_cast<float>(deformation.nodes[i].rotation.m[k]);
        }
        host_translations[i * 3 + 0] =
            static_cast<float>(deformation.nodes[i].translation.x);
        host_translations[i * 3 + 1] =
            static_cast<float>(deformation.nodes[i].translation.y);
        host_translations[i * 3 + 2] =
            static_cast<float>(deformation.nodes[i].translation.z);
    }
    if (!host_rotations.empty()) {
        rotations.assign(host_rotations.begin(), host_rotations.end());
        translations.assign(host_translations.begin(), host_translations.end());
    }
}

// Rebuilds the host-side deformation from the device state. This is the only
// download of the transforms in a level, and it happens at the level boundary,
// where the next level's warm start needs them.
Deformation download_state(
    const thrust::device_vector<Mat3f>& rotations,
    const thrust::device_vector<float>& translations,
    const std::vector<Vec3>& node_rest_positions) {
    std::vector<Mat3f> host_rotations(rotations.size());
    std::vector<float> host_translations(translations.size());
    if (!host_rotations.empty()) {
        thrust::copy(rotations.begin(), rotations.end(), host_rotations.begin());
    }
    if (!host_translations.empty()) {
        thrust::copy(translations.begin(), translations.end(), host_translations.begin());
    }
    Deformation deformation = identity_deformation(node_rest_positions);
    const std::size_t count = std::min(deformation.nodes.size(), host_rotations.size());
    for (std::size_t i = 0; i < count; ++i) {
        for (int k = 0; k < 9; ++k) {
            deformation.nodes[i].rotation.m[k] = host_rotations[i].m[k];
        }
        deformation.nodes[i].translation = {host_translations[i * 3 + 0],
                                            host_translations[i * 3 + 1],
                                            host_translations[i * 3 + 2]};
    }
    return deformation;
}

struct EnergyReadout {
    float total = 0.0f;
    float point_to_point = 0.0f;
    float point_to_plane = 0.0f;
    float smoothness = 0.0f;
    float anchor = 0.0f;
    float landmark = 0.0f;
};

EnergyReadout read_energies(
    const thrust::device_vector<float>& energies, PinnedFloats& staging,
    cudaStream_t stream) {
    read_back(energies, kEnergyKinds, staging, stream);
    EnergyReadout out;
    out.point_to_point = staging.at(0);
    out.point_to_plane = staging.at(1);
    out.smoothness = staging.at(2);
    out.anchor = staging.at(3);
    out.landmark = staging.at(4);
    out.total = out.point_to_point + out.point_to_plane + out.smoothness +
                out.anchor + out.landmark;
    return out;
}

struct PcgOutcome {
    bool ok = false;
    int iterations = 0;
    float residual = 0.0f;
};

bool cooperative_launch_supported() {
    int device = 0;
    int supported = 0;
    return cudaGetDevice(&device) == cudaSuccess &&
           cudaDeviceGetAttribute(&supported, cudaDevAttrCooperativeLaunch, device) ==
               cudaSuccess &&
           supported != 0;
}

// Blocks that fit on the device at once, which is what a cooperative launch
// requires: the whole grid has to be resident, so it is sized from the kernel's
// own occupancy rather than guessed.
int cooperative_block_count() {
    static int cached = -1;
    if (cached >= 0) {
        return cached;
    }
    cached = 0;
    int device = 0;
    cudaDeviceProp properties{};
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaGetDeviceProperties(&properties, device) != cudaSuccess) {
        return cached;
    }
    int blocks_per_sm = 0;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, pcg_kernel,
                                                      kThreads, 0) != cudaSuccess) {
        return cached;
    }
    cached = std::max(1, blocks_per_sm) * std::max(1, properties.multiProcessorCount);
    return cached;
}

}  // namespace

// The session's device state and its position in the schedule. Keeping it behind
// a pointer is what lets the header stay free of the CUDA and thrust types, so a
// host translation unit that only calls the API does not pull them in.
struct RouteBSession::State {
    // Copies of the inputs, so the caller's objects do not have to outlive the
    // session.
    Mesh template_mesh;
    TemplateGraph graph;
    RouteSettings settings;
    std::vector<Landmark> landmarks;

    Lbvh hierarchy;
    StreamGuard stream_guard;
    PinnedFloats staging{std::max(kEnergyKinds, 3)};
    thrust::device_vector<float> points;
    thrust::device_vector<int> template_faces;
    thrust::device_vector<unsigned char> mask;
    const unsigned char* mask_pointer = nullptr;
    thrust::device_vector<int> landmark_vertices;
    thrust::device_vector<float> landmark_targets;
    // The landmark rows live inside the gathered pass, which walks vertices, so
    // the markers are also indexed per vertex: a flag saying which vertices are
    // markers at all, and the position each is asked to sit on. Both are fixed
    // for the run; the weight itself comes from the level's schedule, since the
    // per-level schedule anneals it.
    thrust::device_vector<float> landmark_flag;
    thrust::device_vector<float> landmark_target;
    GpuLevel level;  // reserved at the finest level's size and reused

    cudaStream_t stream = nullptr;
    int vertex_count = 0;
    int triangle_count = 0;
    int landmark_count = 0;
    int pcg_max_iterations = 30;
    int grid_blocks = 1;
    float distance_threshold = 0.0f;
    float min_normal_cosine = -1.0f;
    float min_slope = 0.0f;
    PcgState host_pcg;

    // Where the schedule got to. A step runs exactly one phase, so the caller
    // regains control between an outer iteration's correspondences, its linear
    // solves, and the level boundary.
    enum class Phase { BeginLevel, Refresh, InnerStep, CloseOuter, EndLevel, Done };
    Phase phase = Phase::BeginLevel;
    bool finished = false;
    std::size_t level_index = 0;
    int outer = 0;
    int inner = 0;
    double damping = 0.0;
    double current_energy = 0.0;
    double previous_level_energy = 0.0;
    LevelView view;
    RunReport report;
    std::vector<CorrespondenceCounts> counts_per_level;
    CorrespondenceCounts counts;
    Deformation deformation;
    EnergyReadout level_readout;
    bool evaluated = false;
    std::vector<Vec3> cached_vertices;

    auto par() const { return thrust::cuda::par.on(stream); }
    LevelSettings level_settings() const {
        return level_settings_for(settings, level_index);
    }

    EnergyReadout accumulate(const LevelSettings& settings_for_level,
                             bool with_jacobians);
    CorrespondenceCounts summarize();
    PcgOutcome solve_system(float damping_value);
    void begin_level();
    void refresh();
    bool inner_step();
    bool close_outer();
    bool end_level();
    // The fitted template in the graph's normalized frame -- what a RouteResult
    // carries, since the caller converts it back to its own units.
    std::vector<Vec3> fitted_vertices();
};

EnergyReadout RouteBSession::State::accumulate(
    const LevelSettings& settings_for_level, bool with_jacobians) {
    thrust::fill(par(), level.energies.begin(), level.energies.end(), 0.0f);
    if (with_jacobians) {
        // The gathered pass assigns every node's block and right-hand side
        // outright, so neither needs clearing first; only the per-edge blocks do,
        // because the smoothness kernel accumulates into the edge that owns them.
        thrust::fill(par(), level.edge_blocks.begin(), level.edge_blocks.end(), Mat6f{});
        assemble_node_kernel<<<dim3(level.node_count), kGatherThreads, 0, stream>>>(
            thrust::raw_pointer_cast(points.data()),
            thrust::raw_pointer_cast(level.node_rest.data()),
            thrust::raw_pointer_cast(level.binding_nodes.data()),
            thrust::raw_pointer_cast(level.binding_weights.data()),
            thrust::raw_pointer_cast(level.binding_counts.data()),
            thrust::raw_pointer_cast(level.rotations.data()),
            thrust::raw_pointer_cast(level.translations.data()),
            thrust::raw_pointer_cast(level.correspondence_points.data()),
            thrust::raw_pointer_cast(level.correspondence_normals.data()),
            thrust::raw_pointer_cast(level.rejection.data()),
            thrust::raw_pointer_cast(level.robust_weights.data()), mask_pointer,
            thrust::raw_pointer_cast(level.gather_begin.data()),
            thrust::raw_pointer_cast(level.gather_vertex.data()),
            thrust::raw_pointer_cast(landmark_flag.data()),
            thrust::raw_pointer_cast(landmark_target.data()),
            static_cast<float>(settings_for_level.weights.point_to_point),
            static_cast<float>(settings_for_level.weights.point_to_plane),
            static_cast<float>(settings_for_level.weights.landmark),
            level.node_count, thrust::raw_pointer_cast(level.diagonal.data()),
            thrust::raw_pointer_cast(level.rhs.data()));
        throw_on_cuda_error(cudaGetLastError());
        if (level.pair_group_count > 0) {
            // One block per group, like the gathered node pass.
            assemble_pair_kernel<<<dim3(level.pair_group_count), kGatherThreads, 0,
                                   stream>>>(
                thrust::raw_pointer_cast(points.data()),
                thrust::raw_pointer_cast(level.node_rest.data()),
                thrust::raw_pointer_cast(level.binding_nodes.data()),
                thrust::raw_pointer_cast(level.binding_weights.data()),
                thrust::raw_pointer_cast(level.binding_counts.data()),
                thrust::raw_pointer_cast(level.rotations.data()),
                thrust::raw_pointer_cast(level.translations.data()),
                thrust::raw_pointer_cast(level.correspondence_points.data()),
                thrust::raw_pointer_cast(level.correspondence_normals.data()),
                thrust::raw_pointer_cast(level.rejection.data()),
                thrust::raw_pointer_cast(level.robust_weights.data()), mask_pointer,
                thrust::raw_pointer_cast(level.pair_payload.data()),
                thrust::raw_pointer_cast(level.pair_group_begin.data()),
                level.pair_group_count, level.pair_count,
                thrust::raw_pointer_cast(landmark_flag.data()),
                thrust::raw_pointer_cast(landmark_target.data()),
                static_cast<float>(settings_for_level.weights.point_to_point),
                static_cast<float>(settings_for_level.weights.point_to_plane),
                static_cast<float>(settings_for_level.weights.landmark),
                level.edge_count,
                thrust::raw_pointer_cast(level.edge_blocks.data()));
            throw_on_cuda_error(cudaGetLastError());
        }
    }
    accumulate_anchor_kernel<<<blocks_for(level.node_count), kThreads, 0, stream>>>(
        level.node_count, thrust::raw_pointer_cast(level.rotations.data()),
        thrust::raw_pointer_cast(level.translations.data()),
        static_cast<float>(settings_for_level.weights.anchor), with_jacobians,
        thrust::raw_pointer_cast(level.diagonal.data()),
        thrust::raw_pointer_cast(level.rhs.data()),
        thrust::raw_pointer_cast(level.energies.data()));
    throw_on_cuda_error(cudaGetLastError());
    if (settings_for_level.weights.smoothness > 0.0 && level.edge_count > 0) {
        accumulate_smoothness_kernel<<<blocks_for(level.edge_count), kThreads, 0,
                                       stream>>>(
            thrust::raw_pointer_cast(level.node_rest.data()),
            thrust::raw_pointer_cast(level.edges.data()), level.edge_count,
            thrust::raw_pointer_cast(level.rotations.data()),
            thrust::raw_pointer_cast(level.translations.data()),
            static_cast<float>(settings_for_level.weights.smoothness), with_jacobians,
            thrust::raw_pointer_cast(level.diagonal.data()),
            thrust::raw_pointer_cast(level.rhs.data()),
            thrust::raw_pointer_cast(level.edge_blocks.data()),
            thrust::raw_pointer_cast(level.energies.data()));
        throw_on_cuda_error(cudaGetLastError());
    }
    if (settings_for_level.weights.point_to_point > 0.0 ||
        settings_for_level.weights.point_to_plane > 0.0) {
        energy_data_kernel<<<blocks_for(vertex_count), kThreads, 0, stream>>>(
            thrust::raw_pointer_cast(points.data()),
            thrust::raw_pointer_cast(level.node_rest.data()),
            thrust::raw_pointer_cast(level.binding_nodes.data()),
            thrust::raw_pointer_cast(level.binding_weights.data()),
            thrust::raw_pointer_cast(level.binding_counts.data()),
            thrust::raw_pointer_cast(level.rotations.data()),
            thrust::raw_pointer_cast(level.translations.data()),
            thrust::raw_pointer_cast(level.correspondence_points.data()),
            thrust::raw_pointer_cast(level.correspondence_normals.data()),
            thrust::raw_pointer_cast(level.rejection.data()),
            thrust::raw_pointer_cast(level.robust_weights.data()), mask_pointer,
            vertex_count,
            static_cast<float>(settings_for_level.weights.point_to_plane),
            static_cast<float>(settings_for_level.weights.point_to_point),
            thrust::raw_pointer_cast(level.energies.data()));
        throw_on_cuda_error(cudaGetLastError());
    }
    if (settings_for_level.weights.landmark > 0.0 && landmark_count > 0) {
        accumulate_landmark_kernel<<<blocks_for(landmark_count), kThreads, 0, stream>>>(
            thrust::raw_pointer_cast(points.data()),
            thrust::raw_pointer_cast(level.node_rest.data()),
            thrust::raw_pointer_cast(level.binding_nodes.data()),
            thrust::raw_pointer_cast(level.binding_weights.data()),
            thrust::raw_pointer_cast(level.binding_counts.data()),
            thrust::raw_pointer_cast(level.rotations.data()),
            thrust::raw_pointer_cast(level.translations.data()),
            thrust::raw_pointer_cast(landmark_vertices.data()),
            thrust::raw_pointer_cast(landmark_targets.data()), landmark_count,
                static_cast<float>(settings_for_level.weights.landmark), false,
            thrust::raw_pointer_cast(level.diagonal.data()),
            thrust::raw_pointer_cast(level.rhs.data()),
            thrust::raw_pointer_cast(level.energies.data()));
        throw_on_cuda_error(cudaGetLastError());
    }
    return read_energies(level.energies, staging, stream);
}

// The correspondence statistics: an atomic tally of the filters and one device
// sort for the percentiles, so the host reads five integers and three floats
// rather than a slice of the correspondence arrays.
CorrespondenceCounts RouteBSession::State::summarize() {
    CorrespondenceCounts out;
    out.total = vertex_count;
    throw_on_cuda_error(cudaMemsetAsync(
        thrust::raw_pointer_cast(level.rejection_counts.data()), 0,
        sizeof(int) * kRejectionKinds, stream));
    throw_on_cuda_error(cudaMemsetAsync(
        thrust::raw_pointer_cast(level.distance_sum.data()), 0, sizeof(float),
        stream));
    correspondence_counts_kernel<<<blocks_for(vertex_count), kThreads, 0, stream>>>(
        thrust::raw_pointer_cast(level.rejection.data()),
        thrust::raw_pointer_cast(level.correspondence_distances.data()), vertex_count,
        thrust::raw_pointer_cast(level.rejection_counts.data()),
        thrust::raw_pointer_cast(level.distance_sum.data()));
    throw_on_cuda_error(cudaGetLastError());
    distances_with_infinity_kernel<<<blocks_for(vertex_count), kThreads, 0, stream>>>(
        thrust::raw_pointer_cast(level.rejection.data()),
        thrust::raw_pointer_cast(level.correspondence_distances.data()), vertex_count,
        thrust::raw_pointer_cast(level.sorted_distances.data()));
    throw_on_cuda_error(cudaGetLastError());
    thrust::sort(par(), level.sorted_distances.begin(), level.sorted_distances.end());
    percentile_kernel<<<1, 1, 0, stream>>>(
        thrust::raw_pointer_cast(level.sorted_distances.data()),
        thrust::raw_pointer_cast(level.rejection_counts.data()),
        thrust::raw_pointer_cast(level.distance_sum.data()),
        thrust::raw_pointer_cast(level.statistics.data()));
    throw_on_cuda_error(cudaGetLastError());
    std::vector<int> host_counts(kRejectionKinds);
    std::vector<float> host_statistics(3);
    throw_on_cuda_error(cudaMemcpyAsync(
        host_counts.data(), thrust::raw_pointer_cast(level.rejection_counts.data()),
        sizeof(int) * kRejectionKinds, cudaMemcpyDeviceToHost, stream));
    throw_on_cuda_error(cudaMemcpyAsync(
        host_statistics.data(), thrust::raw_pointer_cast(level.statistics.data()),
        sizeof(float) * 3, cudaMemcpyDeviceToHost, stream));
    throw_on_cuda_error(cudaStreamSynchronize(stream));
    out.accepted = host_counts[0];
    out.rejected_distance = host_counts[1];
    out.rejected_normal = host_counts[2];
    out.rejected_slope = host_counts[3];
    out.rejected_masked = host_counts[4];
    out.distance_mean = host_statistics[0];
    out.distance_p95 = host_statistics[1];
    out.distance_max = host_statistics[2];
    return out;
}

// One damped linear solve. The preconditioner is rebuilt for the current
// damping, the recurrence runs inside a single cooperative kernel, and the only
// traffic afterwards is the four scalars the outcome consists of.
PcgOutcome RouteBSession::State::solve_system(float damping_value) {
    PcgOutcome outcome;
    if (level.node_count <= 0) {
        return outcome;
    }
    factor_kernel<<<blocks_for(level.node_count), kThreads, 0, stream>>>(
        thrust::raw_pointer_cast(level.diagonal.data()), damping_value,
        settings.diagonal_damping, thrust::raw_pointer_cast(level.factor.data()),
        level.node_count);
    throw_on_cuda_error(cudaGetLastError());
    throw_on_cuda_error(cudaMemsetAsync(
        thrust::raw_pointer_cast(level.pcg_workspace.data()), 0,
        sizeof(float) *
            PcgWorkspace::floats_for(level.node_count, pcg_max_iterations),
        stream));
    throw_on_cuda_error(cudaMemsetAsync(thrust::raw_pointer_cast(level.pcg.data()), 0,
                                        sizeof(PcgState), stream));

    // The cooperative launch takes its arguments by address, so every one of them
    // has to be a mutable local of exactly the kernel's parameter type.
    Mat6f* diagonal = thrust::raw_pointer_cast(level.diagonal.data());
    Mat6f* factor = thrust::raw_pointer_cast(level.factor.data());
    Mat6f* edge_blocks = thrust::raw_pointer_cast(level.edge_blocks.data());
    int* edges = thrust::raw_pointer_cast(level.edges.data());
    // The block solver sees the graph's edges and the node-pair groups as one
    // list: a group is an edge with two nodes.
    int edge_count = level.total_edge_count;
    float* rhs = thrust::raw_pointer_cast(level.rhs.data());
    bool diagonal_scaling = settings.diagonal_damping;
    int node_count = level.node_count;
    float tolerance = settings.pcg_tolerance;
    int iteration_limit = pcg_max_iterations;
    float* delta = thrust::raw_pointer_cast(level.delta.data());
    float* workspace = thrust::raw_pointer_cast(level.pcg_workspace.data());
    PcgState* pcg = thrust::raw_pointer_cast(level.pcg.data());
    void* arguments[] = {&diagonal,      &factor,    &edge_blocks,
                         &edges,         &edge_count, &rhs,
                         &damping_value, &diagonal_scaling,
                         &node_count,    &tolerance,  &iteration_limit,
                         &delta,         &workspace,  &pcg};
    throw_on_cuda_error(cudaLaunchCooperativeKernel(
        reinterpret_cast<const void*>(pcg_kernel), dim3(grid_blocks), dim3(kThreads),
        arguments, 0, stream));

    throw_on_cuda_error(cudaMemcpyAsync(&host_pcg, pcg, sizeof(PcgState),
                                        cudaMemcpyDeviceToHost, stream));
    throw_on_cuda_error(cudaStreamSynchronize(stream));
    outcome.ok = host_pcg.converged != 0.0f;
    outcome.iterations = host_pcg.iteration_count;
    outcome.residual = host_pcg.residual;
    return outcome;
}

void RouteBSession::State::begin_level() {
    const GraphLevel& graph_level = graph.levels[level_index];
    if (level_index > 0) {
        deformation = transfer_to_level(graph.levels[level_index - 1], deformation,
                                        graph_level);
    }
    load_level(level, graph_level, vertex_count, stream);
    upload_state(deformation, level.rotations, level.translations);
    view = LevelView{};
    view.level = static_cast<int>(level_index);
    view.nodes = level.node_count;
    damping = settings.initial_damping;
    previous_level_energy = 0.0;
    outer = 0;
    inner = 0;
    current_energy = 0.0;
}

// One correspondence update: the deformed template, its normals, the closest
// point on the target for every vertex, the robust weights, and the system the
// linear solves will consume.
void RouteBSession::State::refresh() {
    const LevelSettings settings_for_level = level_settings();
    evaluate_kernel<<<blocks_for(vertex_count), kThreads, 0, stream>>>(
        thrust::raw_pointer_cast(points.data()),
        thrust::raw_pointer_cast(level.binding_nodes.data()),
        thrust::raw_pointer_cast(level.binding_weights.data()),
        thrust::raw_pointer_cast(level.binding_counts.data()),
        thrust::raw_pointer_cast(level.rotations.data()),
        thrust::raw_pointer_cast(level.translations.data()),
        thrust::raw_pointer_cast(level.node_rest.data()), vertex_count,
        thrust::raw_pointer_cast(level.deformed.data()));
    throw_on_cuda_error(cudaGetLastError());
    thrust::fill(par(), level.query_normals.begin(), level.query_normals.end(), 0.0f);
    accumulate_face_normals_kernel<<<blocks_for(triangle_count), kThreads, 0, stream>>>(
        thrust::raw_pointer_cast(level.deformed.data()),
        thrust::raw_pointer_cast(template_faces.data()), triangle_count,
        thrust::raw_pointer_cast(level.query_normals.data()));
    throw_on_cuda_error(cudaGetLastError());
    normalize_vectors_kernel<<<blocks_for(vertex_count), kThreads, 0, stream>>>(
        thrust::raw_pointer_cast(level.query_normals.data()), vertex_count);
    throw_on_cuda_error(cudaGetLastError());
    hierarchy.query(
        thrust::raw_pointer_cast(level.deformed.data()),
        thrust::raw_pointer_cast(level.query_normals.data()), vertex_count,
        mask_pointer, distance_threshold, min_normal_cosine, min_slope,
        thrust::raw_pointer_cast(level.correspondence_points.data()),
        thrust::raw_pointer_cast(level.correspondence_normals.data()),
        thrust::raw_pointer_cast(level.correspondence_distances.data()),
        thrust::raw_pointer_cast(level.correspondence_triangles.data()),
        thrust::raw_pointer_cast(level.rejection.data()), stream);
    huber_weights_kernel<<<blocks_for(vertex_count), kThreads, 0, stream>>>(
        thrust::raw_pointer_cast(level.deformed.data()),
        thrust::raw_pointer_cast(level.correspondence_points.data()),
        thrust::raw_pointer_cast(level.rejection.data()), vertex_count,
        static_cast<float>(settings_for_level.robust_threshold),
        thrust::raw_pointer_cast(level.robust_weights.data()));
    throw_on_cuda_error(cudaGetLastError());
    current_energy = accumulate(settings_for_level, true).total;
    inner = 0;
    if (outer == 0) {
        view.energy_start = current_energy;
    }
}

// One damped Newton step: the solve, the step, the energy, and the decision.
// Returns false when the inner loop for this correspondence set is over.
bool RouteBSession::State::inner_step() {
    const LevelSettings settings_for_level = level_settings();
    const int attempt_limit = std::max(1, settings.max_damping_attempts);
    double candidate_energy = current_energy;
    bool accepted = false;
    for (int attempt = 0; attempt < attempt_limit && !accepted; ++attempt) {
        const PcgOutcome outcome = solve_system(static_cast<float>(damping));
        if (!outcome.ok) {
            damping *= settings.damping_increase;
            continue;
        }
        // Keep the accepted state, step, measure, and only then decide: the whole
        // sequence runs on the device.
        copy_state_kernel<<<blocks_for(level.node_count), kThreads, 0, stream>>>(
            thrust::raw_pointer_cast(level.rotations.data()),
            thrust::raw_pointer_cast(level.rotations_backup.data()),
            thrust::raw_pointer_cast(level.translations.data()),
            thrust::raw_pointer_cast(level.translations_backup.data()),
            level.node_count);
        throw_on_cuda_error(cudaGetLastError());
        retract_kernel<<<blocks_for(level.node_count), kThreads, 0, stream>>>(
            thrust::raw_pointer_cast(level.rotations.data()),
            thrust::raw_pointer_cast(level.translations.data()),
            thrust::raw_pointer_cast(level.delta.data()), level.node_count);
        throw_on_cuda_error(cudaGetLastError());
        candidate_energy = accumulate(settings_for_level, false).total;
        if (candidate_energy < current_energy) {
            accepted = true;
            break;
        }
        // Rejected: the accepted state goes back and the next attempt uses a
        // stronger damping.
        copy_state_kernel<<<blocks_for(level.node_count), kThreads, 0, stream>>>(
            thrust::raw_pointer_cast(level.rotations_backup.data()),
            thrust::raw_pointer_cast(level.rotations.data()),
            thrust::raw_pointer_cast(level.translations_backup.data()),
            thrust::raw_pointer_cast(level.translations.data()), level.node_count);
        throw_on_cuda_error(cudaGetLastError());
        damping *= settings.damping_increase;
    }
    if (!accepted) {
        return false;
    }
    const double improvement = relative_change(current_energy, candidate_energy);
    damping = std::max(damping * settings.damping_decrease, 1e-12);
    inner += 1;
    view.inner_iterations += 1;
    if (improvement <= settings_for_level.inner_tolerance) {
        return false;
    }
    // The accepted state is on the device; assemble the system for the next inner
    // iteration from it.
    current_energy = accumulate(settings_for_level, true).total;
    return true;
}

// Returns true when the level is over.
bool RouteBSession::State::close_outer() {
    const LevelSettings settings_for_level = level_settings();
    level_readout = accumulate(settings_for_level, false);
    const double level_energy = level_readout.total;
    view.energy_end = level_energy;
    bool converged = false;
    view.outer_iterations += 1;
    if (outer > 0 &&
        relative_change(previous_level_energy, level_energy) <=
            settings_for_level.outer_tolerance) {
        view.converged = true;
        converged = true;
    }
    previous_level_energy = level_energy;
    outer += 1;
    if (converged) {
        return true;
    }
    return outer >= std::max(1, settings_for_level.max_outer_iterations);
}

// Returns true when that was the last level.
bool RouteBSession::State::end_level() {
    const LevelSettings settings_for_level = level_settings();
    counts = summarize();
    view.correspondence_mean = counts.distance_mean;
    counts_per_level.push_back(counts);
    report.levels.push_back(view);
    record_level_settings(report, level_index, settings_for_level.weights,
                          settings_for_level.robust_threshold);
    // The level is over: the transforms come back to the host, for the warm start
    // of the next level and for the caller.
    deformation = download_state(level.rotations, level.translations,
                                 graph.levels[level_index].node_rest_positions);
    if (level_index + 1 == graph.levels.size()) {
        // Evaluate once more so the cached vertices belong to the final accepted
        // state rather than to the start of the last outer pass.
        evaluate_kernel<<<blocks_for(vertex_count), kThreads, 0, stream>>>(
            thrust::raw_pointer_cast(points.data()),
            thrust::raw_pointer_cast(level.binding_nodes.data()),
            thrust::raw_pointer_cast(level.binding_weights.data()),
            thrust::raw_pointer_cast(level.binding_counts.data()),
            thrust::raw_pointer_cast(level.rotations.data()),
            thrust::raw_pointer_cast(level.translations.data()),
            thrust::raw_pointer_cast(level.node_rest.data()), vertex_count,
            thrust::raw_pointer_cast(level.deformed.data()));
        throw_on_cuda_error(cudaGetLastError());
        cached_vertices = download_vectors(level.deformed);
        evaluated = true;
        report.energy.point_to_point = level_readout.point_to_point;
        report.energy.point_to_plane = level_readout.point_to_plane;
        report.energy.smoothness = level_readout.smoothness;
        report.energy.anchor = level_readout.anchor;
        report.energy.landmark = level_readout.landmark;
        report.correspondence.total = counts.total;
        report.correspondence.accepted = counts.accepted;
        report.correspondence.rejected_distance = counts.rejected_distance;
        report.correspondence.distance_mean = counts.distance_mean;
        report.correspondence.distance_p95 = counts.distance_p95;
        report.correspondence.distance_max = counts.distance_max;
        return true;
    }
    ++level_index;
    return false;
}

RouteBSession::RouteBSession(
    const Mesh& normalized_template,
    const TargetQueries& target,
    const TemplateGraph& graph,
    const RouteSettings& settings,
    const VertexMask* mask,
    const std::vector<Landmark>& landmarks)
    : state_(std::make_unique<State>()) {
    State& session = *state_;
    session.report.route = "route_b";
    session.report.precision = "float";
    session.report.version = NRICP_VERSION_STRING;
    session.template_mesh = normalized_template;
    session.graph = graph;
    session.settings = settings;
    session.landmarks = landmarks;
    session.vertex_count = static_cast<int>(normalized_template.vertices.size());
    session.triangle_count = static_cast<int>(normalized_template.faces.size());
    if (session.graph.levels.empty() || session.vertex_count <= 0 ||
        session.triangle_count <= 0) {
        session.finished = true;
        session.phase = State::Phase::Done;
        return;
    }
    if (!cooperative_launch_supported()) {
        throw std::runtime_error(
            "route B needs a device that supports cooperative launches (the linear "
            "solve is a persistent kernel); this device does not report the "
            "attribute");
    }
    session.stream = session.stream_guard.get();
    session.pcg_max_iterations = std::max(1, settings.pcg_max_iterations);
    session.grid_blocks = cooperative_block_count();
    session.distance_threshold =
        static_cast<float>(settings.correspondence.distance_threshold);
    session.min_normal_cosine =
        static_cast<float>(settings.correspondence.min_normal_cosine);
    session.min_slope = static_cast<float>(settings.correspondence.min_slope);
    session.deformation =
        identity_deformation(session.graph.levels.front().node_rest_positions);

    // The target's hierarchy, built on the device and never read back.
    session.hierarchy.build(target.mesh(), session.stream);
    if (session.hierarchy.empty()) {
        session.finished = true;
        session.phase = State::Phase::Done;
        return;
    }

    // The template's fixed inputs, uploaded once.
    session.points = upload_vectors(normalized_template.vertices);
    std::vector<int> template_faces(
        static_cast<std::size_t>(session.triangle_count) * 3);
    for (int triangle = 0; triangle < session.triangle_count; ++triangle) {
        const Int3& face =
            normalized_template.faces[static_cast<std::size_t>(triangle)];
        template_faces[triangle * 3 + 0] = face.v0;
        template_faces[triangle * 3 + 1] = face.v1;
        template_faces[triangle * 3 + 2] = face.v2;
    }
    session.template_faces = upload_ints(template_faces);
    if (mask != nullptr && !mask->active.empty()) {
        session.mask.assign(mask->active.begin(), mask->active.end());
        session.mask_pointer = thrust::raw_pointer_cast(session.mask.data());
    }
    std::vector<int> landmark_vertices;
    std::vector<Vec3> landmark_targets;
    for (const Landmark& landmark : landmarks) {
        landmark_vertices.push_back(landmark.template_vertex);
        landmark_targets.push_back(landmark.target);
    }
    session.landmark_vertices = upload_ints(landmark_vertices);
    session.landmark_targets = upload_vectors(landmark_targets);
    session.landmark_count = static_cast<int>(landmark_vertices.size());
    std::vector<float> landmark_weight(static_cast<std::size_t>(session.vertex_count),
                                       0.0f);
    std::vector<float> landmark_target(static_cast<std::size_t>(session.vertex_count) * 3,
                                       0.0f);
    for (std::size_t index = 0; index < landmark_vertices.size(); ++index) {
        const int vertex = landmark_vertices[index];
        if (vertex < 0 || vertex >= session.vertex_count) {
            continue;
        }
        landmark_weight[static_cast<std::size_t>(vertex)] = 1.0f;
        landmark_target[static_cast<std::size_t>(vertex) * 3 + 0] =
            static_cast<float>(landmark_targets[index].x);
        landmark_target[static_cast<std::size_t>(vertex) * 3 + 1] =
            static_cast<float>(landmark_targets[index].y);
        landmark_target[static_cast<std::size_t>(vertex) * 3 + 2] =
            static_cast<float>(landmark_targets[index].z);
    }
    session.landmark_flag.assign(landmark_weight.begin(), landmark_weight.end());
    session.landmark_target.assign(landmark_target.begin(), landmark_target.end());

    // Every buffer at the finest level's size: from here on a level only assigns
    // into storage that already exists, so neither a level change nor a step
    // reallocates.
    const GraphLevel& finest = session.graph.levels.back();
    reserve_level(session.level, finest.node_count(),
                  static_cast<int>(finest.edges.size()), session.vertex_count,
                  session.pcg_max_iterations);
}

RouteBSession::~RouteBSession() = default;
RouteBSession::RouteBSession(RouteBSession&&) noexcept = default;
RouteBSession& RouteBSession::operator=(RouteBSession&&) noexcept = default;

WrapProgress RouteBSession::step() {
    State& session = *state_;
    if (!session.finished) {
        switch (session.phase) {
            case State::Phase::BeginLevel:
                session.begin_level();
                session.phase = State::Phase::Refresh;
                break;
            case State::Phase::Refresh:
                session.refresh();
                session.phase = State::Phase::InnerStep;
                break;
            case State::Phase::InnerStep:
                session.phase = session.inner_step() ? State::Phase::InnerStep
                                                     : State::Phase::CloseOuter;
                break;
            case State::Phase::CloseOuter:
                session.phase =
                    session.close_outer() ? State::Phase::EndLevel : State::Phase::Refresh;
                break;
            case State::Phase::EndLevel:
                session.phase =
                    session.end_level() ? State::Phase::Done : State::Phase::BeginLevel;
                break;
            case State::Phase::Done:
                session.finished = true;
                break;
        }
    }
    WrapProgress progress;
    progress.level = static_cast<int>(session.level_index);
    progress.outer = session.outer;
    progress.inner = session.inner;
    progress.finished = session.finished;
    progress.level_converged = session.view.converged;
    progress.energy = session.current_energy;
    progress.correspondence_mean = session.counts.distance_mean;
    return progress;
}

bool RouteBSession::finished() const { return state_->finished; }

std::vector<Vec3> RouteBSession::vertices() const {
    State& session = *state_;
    // The schedule works in the normalized frame the graph was built in; a caller
    // drawing the result wants the units it handed in, which is also what the
    // batch entry point reports.
    Mesh restored;
    restored.vertices = session.fitted_vertices();
    undo_normalization(restored, session.graph.normalization);
    return restored.vertices;
}

// Evaluated on demand: one launch and one copy, so a caller can redraw as often
// as it likes without the session having to keep anything in step.
std::vector<Vec3> RouteBSession::State::fitted_vertices() {
    if (evaluated && !cached_vertices.empty()) {
        return cached_vertices;
    }
    if (vertex_count <= 0) {
        return {};
    }
    evaluate_kernel<<<blocks_for(vertex_count), kThreads, 0, stream>>>(
        thrust::raw_pointer_cast(points.data()),
        thrust::raw_pointer_cast(level.binding_nodes.data()),
        thrust::raw_pointer_cast(level.binding_weights.data()),
        thrust::raw_pointer_cast(level.binding_counts.data()),
        thrust::raw_pointer_cast(level.rotations.data()),
        thrust::raw_pointer_cast(level.translations.data()),
        thrust::raw_pointer_cast(level.node_rest.data()), vertex_count,
        thrust::raw_pointer_cast(level.deformed.data()));
    throw_on_cuda_error(cudaGetLastError());
    return download_vectors(level.deformed);
}

const RunReport& RouteBSession::report() const { return state_->report; }

const std::vector<CorrespondenceCounts>& RouteBSession::correspondence_per_level()
    const {
    return state_->counts_per_level;
}

const TemplateGraph& RouteBSession::graph() const { return state_->graph; }

RouteResult RouteBSession::result() const {
    State& session = *state_;
    RouteResult out;
    out.report = session.report;
    out.correspondence_per_level = session.counts_per_level;
    out.deformation = session.deformation;
    if (session.vertex_count > 0) {
        out.vertices = session.fitted_vertices();
    }
    if (!session.evaluated) {
        // The schedule is still running: report the last energy pass and
        // correspondence set it saw, so a partial result is still a result.
        out.report.energy.point_to_point = session.level_readout.point_to_point;
        out.report.energy.point_to_plane = session.level_readout.point_to_plane;
        out.report.energy.smoothness = session.level_readout.smoothness;
        out.report.energy.anchor = session.level_readout.anchor;
        out.report.energy.landmark = session.level_readout.landmark;
        out.report.correspondence.total = session.counts.total;
        out.report.correspondence.accepted = session.counts.accepted;
        out.report.correspondence.rejected_distance =
            session.counts.rejected_distance;
        out.report.correspondence.distance_mean = session.counts.distance_mean;
        out.report.correspondence.distance_p95 = session.counts.distance_p95;
        out.report.correspondence.distance_max = session.counts.distance_max;
    }
    return out;
}

RouteResult solve_route_b(
    const Mesh& normalized_template,
    const TargetQueries& target,
    const TemplateGraph& graph,
    const RouteSettings& settings,
    const VertexMask* mask,
    const std::vector<Landmark>& landmarks) {
    RouteBSession session(normalized_template, target, graph, settings, mask,
                          landmarks);
    while (!session.finished()) {
        session.step();
    }
    return session.result();
}

bool device_available() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

std::string device_description() {
    if (!device_available()) {
        return "no CUDA device";
    }
    cudaDeviceProp properties{};
    if (cudaGetDeviceProperties(&properties, 0) != cudaSuccess) {
        return "CUDA device present, properties unavailable";
    }
    return std::string(properties.name) + " (cc " +
           std::to_string(properties.major) + "." +
           std::to_string(properties.minor) + ")";
}

}  // namespace nricp
