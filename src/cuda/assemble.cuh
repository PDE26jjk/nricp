#pragma once

#include <cuda_runtime.h>

#include "cuda/gpu_level.cuh"
#include "cuda/host_staging.cuh"

// ---------------------------------------------------------------------------
// The reduced system and the block energies, assembled on the device
// ---------------------------------------------------------------------------
//
// One pass over a level produces everything the schedule consumes from it: the
// blocks and right-hand side of the damped Newton system, and the five energy
// blocks the accept/reject test and the convergence test compare. The passes
// are separate kernels because they walk different things -- the data term
// walks nodes and node pairs, the smoothness term walks edges, the anchor walks
// nodes again -- and one launch each is cheaper than one kernel that walks all
// of them.
namespace nricp {
namespace gpu {

// What one accumulation pass needs beyond the level itself. The template's
// rest vertices and the landmark data belong to the session rather than to the
// level, because they are the same for every level.
struct AssembleInputs {
    const float* template_points = nullptr;
    const unsigned char* mask = nullptr;
    const float* landmark_flag = nullptr;     // one flag per vertex
    const float* landmark_target = nullptr;   // and its target, per vertex
    const int* landmark_vertices = nullptr;   // one marker per landmark
    const float* landmark_targets = nullptr;  // and the target it points at
    int landmark_count = 0;
    int vertex_count = 0;
    float point_to_point = 0.0f;
    float point_to_plane = 0.0f;
    float landmark = 0.0f;
    float smoothness = 0.0f;
    float anchor = 0.0f;
};

// One pass over the level: the reduced system's blocks and right-hand side when
// `with_jacobians`, and the five block energies either way.
EnergyReadout assemble_system(
    GpuLevel& level, const AssembleInputs& inputs, bool with_jacobians,
    PinnedFloats& staging, cudaStream_t stream);

}  // namespace gpu
}  // namespace nricp
