#include "cuda/pcg_solver.cuh"

#include <algorithm>

#include <cooperative_groups.h>

#include "cuda/cuda_error.h"
#include "cuda/solver_common.cuh"

namespace nricp {
namespace gpu {

namespace cg = cooperative_groups;

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

    // Last resort: the identity, which keeps the iteration finite. It has to be
    // written out rather than left as a zero-initialized block, because the
    // factor solve divides by the diagonal.
    Mat6f identity;
    for (int row = 0; row < 6; ++row) {
        identity.at(row, row) = 1.0f;
    }
    factor[node] = identity;
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

// One damped linear solve. The preconditioner is rebuilt for the current
// damping, the recurrence runs inside a single cooperative kernel, and the only
// traffic afterwards is the four scalars the outcome consists of.
PcgOutcome solve_linear_system(
    const PcgSystem& system, float damping, bool diagonal_scaling,
    float tolerance, int max_iterations, int grid_blocks, PcgState& host_state,
    cudaStream_t stream) {
    PcgOutcome outcome;
    if (system.node_count <= 0) {
        return outcome;
    }
    factor_kernel<<<blocks_for(system.node_count), kThreads, 0, stream>>>(
        system.diagonal, damping, diagonal_scaling, system.factor,
        system.node_count);
    throw_on_cuda_error(cudaGetLastError());
    throw_on_cuda_error(cudaMemsetAsync(
        system.workspace, 0,
        sizeof(float) * PcgWorkspace::floats_for(system.node_count, max_iterations),
        stream));
    throw_on_cuda_error(
        cudaMemsetAsync(system.device_state, 0, sizeof(PcgState), stream));

    // The cooperative launch takes its arguments by address, so every one of them
    // has to be a mutable local of exactly the kernel's parameter type.
    Mat6f* diagonal = system.diagonal;
    Mat6f* factor = system.factor;
    Mat6f* edge_blocks = system.edge_blocks;
    int* edges = system.edges;
    int edge_count = system.edge_count;
    float* rhs = system.rhs;
    int node_count = system.node_count;
    float* delta = system.delta;
    float* workspace = system.workspace;
    PcgState* pcg = system.device_state;
    void* arguments[] = {&diagonal,      &factor,    &edge_blocks,
                         &edges,         &edge_count, &rhs,
                         &damping,       &diagonal_scaling,
                         &node_count,    &tolerance,  &max_iterations,
                         &delta,         &workspace,  &pcg};
    throw_on_cuda_error(cudaLaunchCooperativeKernel(
        reinterpret_cast<const void*>(pcg_kernel), dim3(grid_blocks),
        dim3(kThreads), arguments, 0, stream));

    throw_on_cuda_error(cudaMemcpyAsync(&host_state, pcg, sizeof(PcgState),
                                        cudaMemcpyDeviceToHost, stream));
    throw_on_cuda_error(cudaStreamSynchronize(stream));
    outcome.ok = host_state.converged != 0.0f;
    outcome.iterations = host_state.iteration_count;
    outcome.residual = host_state.residual;
    return outcome;
}

}  // namespace gpu
}  // namespace nricp
