#pragma once

#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include "core/graph.h"
#include "cuda/device_math.cuh"
#include "cuda/pcg_solver.cuh"

namespace nricp {
namespace gpu {

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
    // The smoothness edges that touch each node. Route B's assembly walks the
    // edges once, one thread per edge; route C's local step and its right-hand
    // side both need each node's own influences, and having them in a list is
    // what keeps those two passes atomic-free.
    thrust::device_vector<int> node_edge_begin;
    thrust::device_vector<int> node_edge_cursor;
    thrust::device_vector<int> node_edge_counts;
    thrust::device_vector<int> node_edge_list;
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
void reserve_level(
    GpuLevel& state, int max_nodes, int max_edges, int vertex_count,
    int pcg_max_iterations);

// The same level, seen as the linear system it defines.
PcgSystem pcg_system_of(GpuLevel& level);

}  // namespace gpu
}  // namespace nricp
