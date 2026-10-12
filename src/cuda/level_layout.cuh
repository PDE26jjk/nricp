#pragma once

#include <cuda_runtime.h>

#include "core/graph.h"
#include "cuda/gpu_level.cuh"

// ---------------------------------------------------------------------------
// One level's indexing, on the device
// ---------------------------------------------------------------------------
//
// The level's topology and its bindings do not change while it is being solved,
// so everything derived from them is derived once per level: the node -> bound
// vertex lists the gathered accumulation walks, and the node pairs that share a
// bound vertex, grouped so that each group owns one cross block of the system.
namespace nricp {
namespace gpu {

void load_level(
    GpuLevel& level, const GraphLevel& graph_level, int vertex_count,
    cudaStream_t stream);

}  // namespace gpu
}  // namespace nricp
