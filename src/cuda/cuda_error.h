#pragma once

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace nricp {
namespace gpu {

// Every device path in this layer reports failures the same way. It exists so
// the two translation units (the solver and the acceleration structure) can
// share one definition without one including the other's internals.
inline void throw_on_cuda_error(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error: ") +
                                 cudaGetErrorString(status));
    }
}

}  // namespace gpu
}  // namespace nricp
