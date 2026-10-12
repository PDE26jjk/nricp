#include "cuda/level_layout.cuh"

#include <algorithm>
#include <cstddef>
#include <vector>

#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/system/cuda/execution_policy.h>

#include "cuda/cuda_error.h"
#include "cuda/host_staging.cuh"
#include "cuda/solver_common.cuh"

namespace nricp {
namespace gpu {

// ---------------------------------------------------------------------------
// The node -> bound-vertex lists
// ---------------------------------------------------------------------------
//
// Built once per level, so the counting pass is the only atomic left on this
// path and it does not run per iteration. A vertex lands in the list of every
// node it binds to with a non-zero weight, which is what makes the gather cover
// the same rows the scatter did.
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
// The node -> incident-edge lists
// ---------------------------------------------------------------------------
//
// Both routes walk the edges once per pass, but route C's local step and its
// right-hand side need each node's own influences: the fit for node j sums only
// the terms that mention j, and the right-hand side is assembled per node. A
// CSR over the edges is what lets both of those passes own their data instead
// of firing atomics at a shared block.

static __global__ void count_node_edges_kernel(
    const int* __restrict__ edge_nodes, int edge_count, int* __restrict__ counts) {
    const int edge = blockIdx.x * blockDim.x + threadIdx.x;
    if (edge >= edge_count) {
        return;
    }
    atomicAdd(&counts[edge_nodes[edge * 2 + 0]], 1);
    atomicAdd(&counts[edge_nodes[edge * 2 + 1]], 1);
}

static __global__ void scatter_node_edges_kernel(
    const int* __restrict__ edge_nodes, int edge_count, int* __restrict__ cursor,
    int* __restrict__ out) {
    const int edge = blockIdx.x * blockDim.x + threadIdx.x;
    if (edge >= edge_count) {
        return;
    }
    const int node_a = edge_nodes[edge * 2 + 0];
    const int node_b = edge_nodes[edge * 2 + 1];
    out[atomicAdd(&cursor[node_a], 1)] = edge;
    out[atomicAdd(&cursor[node_b], 1)] = edge;
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

    // ---- the node -> incident-edge lists ----
    // The same two passes as the vertex lists above, over the graph's edges
    // instead of the vertices. Only the smoothness edges are listed: the
    // node-pair groups appended to the edge list later are a device-side
    // artifact of the assembly, not part of the graph.
    const int* edge_nodes = thrust::raw_pointer_cast(state.edges.data());
    thrust::fill(par, state.node_edge_counts.begin(),
                 state.node_edge_counts.begin() + state.node_count, 0);
    count_node_edges_kernel<<<blocks_for(state.edge_count), kThreads, 0, stream>>>(
        edge_nodes, state.edge_count,
        thrust::raw_pointer_cast(state.node_edge_counts.data()));
    throw_on_cuda_error(cudaGetLastError());
    thrust::fill(par, state.node_edge_begin.begin(),
                 state.node_edge_begin.begin() + 1, 0);
    thrust::inclusive_scan(par, state.node_edge_counts.begin(),
                           state.node_edge_counts.begin() + state.node_count,
                           state.node_edge_begin.begin() + 1);
    thrust::copy(par, state.node_edge_begin.begin(),
                 state.node_edge_begin.begin() + state.node_count,
                 state.node_edge_cursor.begin());
    scatter_node_edges_kernel<<<blocks_for(state.edge_count), kThreads, 0, stream>>>(
        edge_nodes, state.edge_count,
        thrust::raw_pointer_cast(state.node_edge_cursor.data()),
        thrust::raw_pointer_cast(state.node_edge_list.data()));
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

}  // namespace gpu
}  // namespace nricp
