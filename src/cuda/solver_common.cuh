#pragma once

#include "cuda/device_math.cuh"

// ---------------------------------------------------------------------------
// The vocabulary the device side of route B is written in
// ---------------------------------------------------------------------------
//
// Everything here is inline, so each translation unit that includes this header
// compiles its own copy and no device symbol has to cross one: the device path
// stays linkable without separable compilation, which is what the CMakeLists of
// this directory relies on.
namespace nricp {
namespace gpu {

constexpr int kThreads = 256;
constexpr int kEnergyKinds = 5;  // point-to-point, plane, smoothness, anchor, landmark
constexpr int kRejectionKinds = 5;

// One block owns one node's block in the gathered accumulation, so this is the
// block's own thread count rather than the project-wide one.
constexpr int kGatherThreads = 128;

// Six values per node, so the gathered kernel reduces this many numbers per
// thread at the end of its sweep.
constexpr int kBlockValues = 36;

inline int blocks_for(int count) { return (count + kThreads - 1) / kThreads; }

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
// One linear functional of a node transform, and the row it contributes
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

}  // namespace gpu
}  // namespace nricp
