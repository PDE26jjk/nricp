#include "cuda/assemble.cuh"

#include <algorithm>
#include <cstddef>
#include <vector>

#include <thrust/fill.h>
#include <thrust/system/cuda/execution_policy.h>

#include "cuda/cuda_error.h"
#include "cuda/solver_common.cuh"

namespace nricp {
namespace gpu {

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

// One accumulation pass: the reduced system's blocks and right-hand side, and
// the five block energies. The data term's blocks are assigned outright by the
// gathered passes, so only the parts that are accumulated -- the edge blocks and
// the energies -- have to be cleared first.
EnergyReadout assemble_system(
    GpuLevel& level, const AssembleInputs& inputs, bool with_jacobians,
    PinnedFloats& staging, cudaStream_t stream) {
    const auto par = thrust::cuda::par.on(stream);
    thrust::fill(par, level.energies.begin(), level.energies.end(), 0.0f);
    if (with_jacobians) {
        thrust::fill(par, level.edge_blocks.begin(), level.edge_blocks.end(),
                     Mat6f{});
        assemble_node_kernel<<<dim3(level.node_count), kGatherThreads, 0, stream>>>(
            inputs.template_points,
            thrust::raw_pointer_cast(level.node_rest.data()),
            thrust::raw_pointer_cast(level.binding_nodes.data()),
            thrust::raw_pointer_cast(level.binding_weights.data()),
            thrust::raw_pointer_cast(level.binding_counts.data()),
            thrust::raw_pointer_cast(level.rotations.data()),
            thrust::raw_pointer_cast(level.translations.data()),
            thrust::raw_pointer_cast(level.correspondence_points.data()),
            thrust::raw_pointer_cast(level.correspondence_normals.data()),
            thrust::raw_pointer_cast(level.rejection.data()),
            thrust::raw_pointer_cast(level.robust_weights.data()), inputs.mask,
            thrust::raw_pointer_cast(level.gather_begin.data()),
            thrust::raw_pointer_cast(level.gather_vertex.data()),
            inputs.landmark_flag, inputs.landmark_target, inputs.point_to_point,
            inputs.point_to_plane, inputs.landmark, level.node_count,
            thrust::raw_pointer_cast(level.diagonal.data()),
            thrust::raw_pointer_cast(level.rhs.data()));
        throw_on_cuda_error(cudaGetLastError());
        if (level.pair_group_count > 0) {
            // One block per group, like the gathered node pass.
            assemble_pair_kernel<<<dim3(level.pair_group_count), kGatherThreads, 0,
                                   stream>>>(
                inputs.template_points,
                thrust::raw_pointer_cast(level.node_rest.data()),
                thrust::raw_pointer_cast(level.binding_nodes.data()),
                thrust::raw_pointer_cast(level.binding_weights.data()),
                thrust::raw_pointer_cast(level.binding_counts.data()),
                thrust::raw_pointer_cast(level.rotations.data()),
                thrust::raw_pointer_cast(level.translations.data()),
                thrust::raw_pointer_cast(level.correspondence_points.data()),
                thrust::raw_pointer_cast(level.correspondence_normals.data()),
                thrust::raw_pointer_cast(level.rejection.data()),
                thrust::raw_pointer_cast(level.robust_weights.data()), inputs.mask,
                thrust::raw_pointer_cast(level.pair_payload.data()),
                thrust::raw_pointer_cast(level.pair_group_begin.data()),
                level.pair_group_count, level.pair_count, inputs.landmark_flag,
                inputs.landmark_target, inputs.point_to_point,
                inputs.point_to_plane, inputs.landmark, level.edge_count,
                thrust::raw_pointer_cast(level.edge_blocks.data()));
            throw_on_cuda_error(cudaGetLastError());
        }
    }
    accumulate_anchor_kernel<<<blocks_for(level.node_count), kThreads, 0, stream>>>(
        level.node_count, thrust::raw_pointer_cast(level.rotations.data()),
        thrust::raw_pointer_cast(level.translations.data()), inputs.anchor,
        with_jacobians, thrust::raw_pointer_cast(level.diagonal.data()),
        thrust::raw_pointer_cast(level.rhs.data()),
        thrust::raw_pointer_cast(level.energies.data()));
    throw_on_cuda_error(cudaGetLastError());
    if (inputs.smoothness > 0.0f && level.edge_count > 0) {
        accumulate_smoothness_kernel<<<blocks_for(level.edge_count), kThreads, 0,
                                       stream>>>(
            thrust::raw_pointer_cast(level.node_rest.data()),
            thrust::raw_pointer_cast(level.edges.data()), level.edge_count,
            thrust::raw_pointer_cast(level.rotations.data()),
            thrust::raw_pointer_cast(level.translations.data()),
            inputs.smoothness, with_jacobians,
            thrust::raw_pointer_cast(level.diagonal.data()),
            thrust::raw_pointer_cast(level.rhs.data()),
            thrust::raw_pointer_cast(level.edge_blocks.data()),
            thrust::raw_pointer_cast(level.energies.data()));
        throw_on_cuda_error(cudaGetLastError());
    }
    if (inputs.point_to_point > 0.0f || inputs.point_to_plane > 0.0f) {
        energy_data_kernel<<<blocks_for(inputs.vertex_count), kThreads, 0, stream>>>(
            inputs.template_points,
            thrust::raw_pointer_cast(level.node_rest.data()),
            thrust::raw_pointer_cast(level.binding_nodes.data()),
            thrust::raw_pointer_cast(level.binding_weights.data()),
            thrust::raw_pointer_cast(level.binding_counts.data()),
            thrust::raw_pointer_cast(level.rotations.data()),
            thrust::raw_pointer_cast(level.translations.data()),
            thrust::raw_pointer_cast(level.correspondence_points.data()),
            thrust::raw_pointer_cast(level.correspondence_normals.data()),
            thrust::raw_pointer_cast(level.rejection.data()),
            thrust::raw_pointer_cast(level.robust_weights.data()), inputs.mask,
            inputs.vertex_count, inputs.point_to_plane, inputs.point_to_point,
            thrust::raw_pointer_cast(level.energies.data()));
        throw_on_cuda_error(cudaGetLastError());
    }
    if (inputs.landmark > 0.0f && inputs.landmark_count > 0) {
        accumulate_landmark_kernel<<<blocks_for(inputs.landmark_count), kThreads, 0,
                                     stream>>>(
            inputs.template_points,
            thrust::raw_pointer_cast(level.node_rest.data()),
            thrust::raw_pointer_cast(level.binding_nodes.data()),
            thrust::raw_pointer_cast(level.binding_weights.data()),
            thrust::raw_pointer_cast(level.binding_counts.data()),
            thrust::raw_pointer_cast(level.rotations.data()),
            thrust::raw_pointer_cast(level.translations.data()),
            inputs.landmark_vertices, inputs.landmark_targets,
            inputs.landmark_count, inputs.landmark,
            // The landmark rows are already in the gathered pass above; this
            // call is only after the energy they carry.
            false, thrust::raw_pointer_cast(level.diagonal.data()),
            thrust::raw_pointer_cast(level.rhs.data()),
            thrust::raw_pointer_cast(level.energies.data()));
        throw_on_cuda_error(cudaGetLastError());
    }
    return read_energies(level.energies, staging, stream);
}

}  // namespace gpu
}  // namespace nricp
