#pragma once

#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include "cuda/gpu_level.cuh"
#include "solver/correspondence.h"

// ---------------------------------------------------------------------------
// Where the template currently is, and what the correspondences look like
// ---------------------------------------------------------------------------
//
// One call per question the schedule asks between two solves: where the
// template has got to, which way it faces, how heavily each vertex should
// count, and what the correspondence set looks like. Keeping them here rather
// than in the schedule is what lets the schedule read as a sequence of
// decisions rather than as a sequence of launches.
namespace nricp {
namespace gpu {

// The deformed template (the spec's v'_i); one launch over the vertices.
void evaluate_deformation(
    GpuLevel& level, const float* template_points, int vertex_count,
    cudaStream_t stream);

// Area-weighted vertex normals of the deformed template (the spec's n'_i),
// accumulated from the template's faces and normalized.
void build_query_normals(
    GpuLevel& level, const thrust::device_vector<int>& faces, int triangle_count,
    int vertex_count, cudaStream_t stream);

// Huber weights for the correspondence residuals.
void update_robust_weights(
    GpuLevel& level, int vertex_count, float threshold, cudaStream_t stream);

// Why the correspondences were dropped, how far the accepted ones are, and the
// distance percentiles: one tally, one device sort, three floats back.
CorrespondenceCounts summarize_correspondences(
    GpuLevel& level, int vertex_count, cudaStream_t stream);

// The accepted transforms, saved before a step is tried and put back when it is
// rejected.
void save_accepted_state(GpuLevel& level, int node_count, cudaStream_t stream);
void restore_accepted_state(GpuLevel& level, int node_count, cudaStream_t stream);

// The tangent-space increment, applied in place: R <- exp([w]) R and t <- t + u.
void apply_increment(GpuLevel& level, int node_count, cudaStream_t stream);

}  // namespace gpu
}  // namespace nricp
