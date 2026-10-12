#include "cuda/graph_build.cuh"

#include <cooperative_groups.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "cuda/cuda_error.h"

namespace nricp {
namespace {

namespace cg = cooperative_groups;

using gpu::throw_on_cuda_error;

constexpr int kThreads = 256;

int blocks_for(int count) { return (count + kThreads - 1) / kThreads; }

// Squared distance of two three-value points, written the way core/vec3.h's dot
// writes it -- and with the rounding intrinsics, which matter here.
//
// nvcc contracts `x*x + y*y + z*z` into fused multiply-adds by default. That is
// the right default for the solver, and it is what made this builder disagree
// with the core layer's in the last place: two template vertices can sit at
// practically the same distance from the chosen set, and a difference in the
// last bit of that distance swaps the order the ties are broken in. The
// intrinsics force each product and each sum to round on its own, which is what
// the host compiler does, so the seeds come out identical rather than merely
// equivalent.
__device__ inline double squared_distance(const double* a, const double* b) {
    const double x = __dadd_rn(a[0], -b[0]);
    const double y = __dadd_rn(a[1], -b[1]);
    const double z = __dadd_rn(a[2], -b[2]);
    return __dadd_rn(__dadd_rn(__dmul_rn(x, x), __dmul_rn(y, y)),
                     __dmul_rn(z, z));
}

// Keeps a small list of the nearest entries, sorted by (distance, index): the
// rule the reference's partial sort applies.
__device__ inline void keep_nearest(
    double* squared, int* index, int& held, int capacity, double value, int entry) {
    if (held == capacity) {
        const bool beats_worst =
            value < squared[capacity - 1] ||
            (value == squared[capacity - 1] && entry < index[capacity - 1]);
        if (!beats_worst) {
            return;
        }
    }
    int at = held < capacity ? held++ : capacity - 1;
    while (at > 0 && (squared[at - 1] > value ||
                      (squared[at - 1] == value && index[at - 1] > entry))) {
        squared[at] = squared[at - 1];
        index[at] = index[at - 1];
        --at;
    }
    squared[at] = value;
    index[at] = entry;
}

// Farthest-point sampling, all of it in one cooperative launch.
//
// The greedy pass is sequential in its choices -- each seed depends on the set
// chosen before it -- but not in its work: one step is a sweep over every vertex
// plus a maximum. The steps are separated by grid barriers instead of by host
// round trips, so the whole sampling of a level costs one launch.
static __global__ void farthest_seeds_kernel(
    const double* __restrict__ points, int point_count, int seed_count,
    int* __restrict__ seeds, double* __restrict__ nearest,
    double* __restrict__ block_values, int* __restrict__ block_indices) {
    const cg::grid_group grid = cg::this_grid();
    const int stride = gridDim.x * blockDim.x;
    const int thread = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ double shared_value[kThreads];
    __shared__ int shared_index[kThreads];

    if (thread == 0) {
        seeds[0] = 0;
    }
    for (int i = thread; i < point_count; i += stride) {
        nearest[i] = squared_distance(points + static_cast<std::size_t>(i) * 3, points);
    }
    grid.sync();

    for (int step = 1; step < seed_count; ++step) {
        // Each thread's farthest point, then the block's, then the grid's. The
        // comparison is "greater value, or equal value and lower index", which
        // is what the reference's single sweep finds.
        double value = -1.0;
        int index = 0;
        for (int i = thread; i < point_count; i += stride) {
            if (nearest[i] > value) {
                value = nearest[i];
                index = i;
            }
        }
        shared_value[threadIdx.x] = value;
        shared_index[threadIdx.x] = index;
        __syncthreads();
        for (int span = blockDim.x / 2; span > 0; span >>= 1) {
            if (threadIdx.x < span) {
                const double other = shared_value[threadIdx.x + span];
                const int other_index = shared_index[threadIdx.x + span];
                if (other > shared_value[threadIdx.x] ||
                    (other == shared_value[threadIdx.x] &&
                     other_index < shared_index[threadIdx.x])) {
                    shared_value[threadIdx.x] = other;
                    shared_index[threadIdx.x] = other_index;
                }
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            block_values[blockIdx.x] = shared_value[0];
            block_indices[blockIdx.x] = shared_index[0];
        }
        grid.sync();
        if (thread == 0) {
            double best = block_values[0];
            int best_index = block_indices[0];
            for (int block = 1; block < static_cast<int>(gridDim.x); ++block) {
                const double candidate = block_values[block];
                const int candidate_index = block_indices[block];
                if (candidate > best ||
                    (candidate == best && candidate_index < best_index)) {
                    best = candidate;
                    best_index = candidate_index;
                }
            }
            seeds[step] = best_index;
        }
        grid.sync();
        const double* chosen =
            points + static_cast<std::size_t>(seeds[step]) * 3;
        for (int i = thread; i < point_count; i += stride) {
            const double squared =
                squared_distance(points + static_cast<std::size_t>(i) * 3, chosen);
            nearest[i] = nearest[i] < squared ? nearest[i] : squared;
        }
        grid.sync();
    }
}

// One thread per template vertex, walking that level's nodes in index order:
// exactly the reference's loop, so the four nodes and the weights match it.
static __global__ void bind_vertices_kernel(
    const double* __restrict__ vertices, int vertex_count,
    const double* __restrict__ nodes, int node_count, int support,
    int* __restrict__ out_nodes, double* __restrict__ out_weights) {
    const int vertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex >= vertex_count) {
        return;
    }
    const double* point = vertices + static_cast<std::size_t>(vertex) * 3;
    double best_squared[VertexBinding::kMaxNodes];
    int best_index[VertexBinding::kMaxNodes];
    int held = 0;
    for (int node = 0; node < node_count; ++node) {
        keep_nearest(best_squared, best_index, held, support,
                     squared_distance(point, nodes + static_cast<std::size_t>(node) * 3),
                     node);
    }
    const double dmax = sqrt(best_squared[support - 1]);
    double weights[VertexBinding::kMaxNodes];
    double total = 0.0;
    for (int slot = 0; slot < support; ++slot) {
        const double ratio = dmax > 0.0 ? sqrt(best_squared[slot]) / dmax : 0.0;
        const double falloff = 1.0 - ratio;
        out_nodes[vertex * VertexBinding::kMaxNodes + slot] = best_index[slot];
        weights[slot] = falloff * falloff;
        total += weights[slot];
    }
    for (int slot = 0; slot < support; ++slot) {
        out_weights[vertex * VertexBinding::kMaxNodes + slot] =
            total > 0.0 ? weights[slot] / total
                        : 1.0 / static_cast<double>(support);
    }
}

}  // namespace

bool device_graph_available() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

TemplateGraph build_graph_on_device(
    const Mesh& template_mesh, const GraphSettings& settings) {
    TemplateGraph graph;
    graph.normalization = compute_normalization(template_mesh);
    Mesh normalized = template_mesh;
    apply_normalization(normalized, graph.normalization);
    const int vertex_count = static_cast<int>(normalized.vertices.size());
    if (vertex_count <= 0) {
        throw_on_cuda_error(cudaErrorInvalidValue);
    }

    // The vertices are the same for every level, so they cross once.
    std::vector<double> host_points(static_cast<std::size_t>(vertex_count) * 3);
    for (int vertex = 0; vertex < vertex_count; ++vertex) {
        const Vec3& point = normalized.vertices[static_cast<std::size_t>(vertex)];
        host_points[vertex * 3 + 0] = point.x;
        host_points[vertex * 3 + 1] = point.y;
        host_points[vertex * 3 + 2] = point.z;
    }
    double* points = nullptr;
    throw_on_cuda_error(
        cudaMalloc(reinterpret_cast<void**>(&points), host_points.size() * sizeof(double)));
    throw_on_cuda_error(cudaMemcpy(points, host_points.data(),
                                   host_points.size() * sizeof(double),
                                   cudaMemcpyHostToDevice));

    int blocks_per_sm = 0;
    throw_on_cuda_error(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, farthest_seeds_kernel, kThreads, 0));
    cudaDeviceProp properties{};
    int device = 0;
    throw_on_cuda_error(cudaGetDevice(&device));
    throw_on_cuda_error(cudaGetDeviceProperties(&properties, device));
    const int grid_blocks =
        std::max(1, blocks_per_sm) * std::max(1, properties.multiProcessorCount);

    double* nearest = nullptr;
    double* block_values = nullptr;
    int* block_indices = nullptr;
    int* seeds = nullptr;
    throw_on_cuda_error(cudaMalloc(reinterpret_cast<void**>(&nearest),
                                   static_cast<std::size_t>(vertex_count) * sizeof(double)));
    throw_on_cuda_error(cudaMalloc(reinterpret_cast<void**>(&block_values),
                                   static_cast<std::size_t>(grid_blocks) * sizeof(double)));
    throw_on_cuda_error(cudaMalloc(reinterpret_cast<void**>(&block_indices),
                                   static_cast<std::size_t>(grid_blocks) * sizeof(int)));
    throw_on_cuda_error(cudaMalloc(reinterpret_cast<void**>(&seeds),
                                   static_cast<std::size_t>(vertex_count) * sizeof(int)));

    const int levels = std::max(1, settings.levels);
    graph.levels.reserve(static_cast<std::size_t>(levels));
    const int support = std::max(
        1, std::min({settings.max_nodes_per_vertex, vertex_count,
                     VertexBinding::kMaxNodes}));
    std::vector<int> host_nodes(static_cast<std::size_t>(vertex_count) *
                                VertexBinding::kMaxNodes);
    std::vector<double> host_weights(static_cast<std::size_t>(vertex_count) *
                                     VertexBinding::kMaxNodes);
    int* device_nodes = nullptr;
    double* device_weights = nullptr;
    throw_on_cuda_error(cudaMalloc(reinterpret_cast<void**>(&device_nodes),
                                   host_nodes.size() * sizeof(int)));
    throw_on_cuda_error(cudaMalloc(reinterpret_cast<void**>(&device_weights),
                                   host_weights.size() * sizeof(double)));

    for (int level = 0; level < levels; ++level) {
        const double growth =
            std::pow(std::max(1, settings.node_growth), static_cast<double>(level));
        const int requested =
            std::max(1, static_cast<int>(std::lround(settings.nodes_initial * growth)));
        const int count = std::min(requested, vertex_count);

        // A cooperative launch takes its arguments by address, so each one has to
        // be a mutable local of exactly the kernel's parameter type.
        int point_count = vertex_count;
        int seed_count = count;
        void* arguments[] = {&points, &point_count, &seed_count, &seeds, &nearest,
                             &block_values, &block_indices};
        throw_on_cuda_error(cudaLaunchCooperativeKernel(
            reinterpret_cast<const void*>(farthest_seeds_kernel), dim3(grid_blocks),
            dim3(kThreads), arguments, 0, nullptr));
        throw_on_cuda_error(cudaGetLastError());

        GraphLevel level_data;
        level_data.node_template_indices.resize(static_cast<std::size_t>(count));
        throw_on_cuda_error(cudaMemcpy(
            level_data.node_template_indices.data(), seeds,
            static_cast<std::size_t>(count) * sizeof(int), cudaMemcpyDeviceToHost));
        level_data.node_rest_positions.reserve(static_cast<std::size_t>(count));
        std::vector<double> node_points(static_cast<std::size_t>(count) * 3);
        for (int node = 0; node < count; ++node) {
            level_data.node_rest_positions.push_back(
                normalized.vertices[static_cast<std::size_t>(
                    level_data.node_template_indices[static_cast<std::size_t>(node)])]);
            for (int axis = 0; axis < 3; ++axis) {
                node_points[node * 3 + axis] =
                    host_points[static_cast<std::size_t>(
                                    level_data.node_template_indices[static_cast<std::size_t>(node)]) *
                                    3 +
                                axis];
            }
        }
        double* device_node_points = nullptr;
        throw_on_cuda_error(cudaMalloc(reinterpret_cast<void**>(&device_node_points),
                                       node_points.size() * sizeof(double)));
        throw_on_cuda_error(cudaMemcpy(device_node_points, node_points.data(),
                                       node_points.size() * sizeof(double),
                                       cudaMemcpyHostToDevice));

        int node_count = count;
        bind_vertices_kernel<<<blocks_for(vertex_count), kThreads>>>(
            points, vertex_count, device_node_points, node_count, support,
            device_nodes, device_weights);
        throw_on_cuda_error(cudaGetLastError());
        throw_on_cuda_error(cudaMemcpy(host_nodes.data(), device_nodes,
                                       host_nodes.size() * sizeof(int),
                                       cudaMemcpyDeviceToHost));
        throw_on_cuda_error(cudaMemcpy(host_weights.data(), device_weights,
                                       host_weights.size() * sizeof(double),
                                       cudaMemcpyDeviceToHost));
        throw_on_cuda_error(cudaFree(device_node_points));

        level_data.bindings.resize(static_cast<std::size_t>(vertex_count));
        for (int vertex = 0; vertex < vertex_count; ++vertex) {
            VertexBinding binding;
            binding.count = support;
            for (int slot = 0; slot < VertexBinding::kMaxNodes; ++slot) {
                binding.nodes[slot] =
                    slot < support ? host_nodes[vertex * VertexBinding::kMaxNodes + slot]
                                   : -1;
                binding.weights[slot] =
                    slot < support
                        ? host_weights[vertex * VertexBinding::kMaxNodes + slot]
                        : 0.0;
            }
            level_data.bindings[static_cast<std::size_t>(vertex)] = binding;
        }
        // The edge set is a few hundred nodes' worth of pairs, so it stays where
        // the rule already lives.
        level_data.edges = build_node_edges(level_data.node_rest_positions);
        graph.levels.push_back(std::move(level_data));
    }

    throw_on_cuda_error(cudaFree(device_nodes));
    throw_on_cuda_error(cudaFree(device_weights));
    throw_on_cuda_error(cudaFree(nearest));
    throw_on_cuda_error(cudaFree(block_values));
    throw_on_cuda_error(cudaFree(block_indices));
    throw_on_cuda_error(cudaFree(seeds));
    throw_on_cuda_error(cudaFree(points));
    return graph;
}

}  // namespace nricp
