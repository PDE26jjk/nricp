#pragma once

#include <cstddef>

#include <cuda_runtime.h>

#include "cuda/device_math.cuh"

// ---------------------------------------------------------------------------
// The linear solve: block-diagonal preconditioner and one cooperative PCG kernel
// ---------------------------------------------------------------------------
namespace nricp {
namespace gpu {

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

struct PcgOutcome {
    bool ok = false;
    int iterations = 0;
    float residual = 0.0f;
};

// The assembled system, as the solve sees it: the blocks it reads, the vectors
// the recurrence works in, and the one scalar record it writes back. A struct
// rather than a parameter list because the cooperative launch takes fourteen
// arguments and every one of them has to be spelled out twice.
struct PcgSystem {
    Mat6f* diagonal = nullptr;
    Mat6f* factor = nullptr;
    Mat6f* edge_blocks = nullptr;
    int* edges = nullptr;
    float* rhs = nullptr;
    float* delta = nullptr;
    float* workspace = nullptr;
    PcgState* device_state = nullptr;
    int node_count = 0;
    int edge_count = 0;
};

PcgOutcome solve_linear_system(
    const PcgSystem& system, float damping, bool diagonal_scaling,
    float tolerance, int max_iterations, int grid_blocks, PcgState& host_state,
    cudaStream_t stream);

bool cooperative_launch_supported();

// Blocks that fit on the device at once, which is what a cooperative launch
// requires: the whole grid has to be resident, so it is sized from the kernel's
// own occupancy rather than guessed.
int cooperative_block_count();

}  // namespace gpu
}  // namespace nricp
