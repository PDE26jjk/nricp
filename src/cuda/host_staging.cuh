#pragma once

#include <vector>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include "core/deformation.h"
#include "core/graph.h"
#include "core/vec3.h"
#include "cuda/cuda_error.h"
#include "cuda/device_math.cuh"

// ---------------------------------------------------------------------------
// The host side of the accelerator path
// ---------------------------------------------------------------------------
//
// Everything in here is about one boundary: the conversions between the core
// layer's types and the flat arrays the kernels take, the one stream the route
// runs on, and the staging a kernel's result goes through when the schedule
// needs to read it. None of it knows anything about the schedule.
namespace nricp {
namespace gpu {

thrust::device_vector<float> upload_vectors(const std::vector<Vec3>& values);
std::vector<Vec3> download_vectors(const thrust::device_vector<float>& values);
thrust::device_vector<int> upload_ints(const std::vector<int>& values);
thrust::device_vector<float> upload_floats(const std::vector<float>& values);
std::vector<int> flatten_binding_nodes(const std::vector<VertexBinding>& bindings);
std::vector<float> flatten_binding_weights(
    const std::vector<VertexBinding>& bindings);

// A pinned staging buffer for the handful of scalars the schedule has to see.
// Control flow -- is this step lowering the energy, has the solve converged --
// cannot be decided without them, but staging through pinned memory keeps each
// one a single transfer instead of a pageable copy that also drains the stream.
class PinnedFloats {
public:
    explicit PinnedFloats(int capacity) {
        throw_on_cuda_error(
            cudaMallocHost(reinterpret_cast<void**>(&data_), sizeof(float) * capacity));
    }
    ~PinnedFloats() {
        if (data_ != nullptr) {
            cudaFreeHost(data_);
        }
    }
    PinnedFloats(const PinnedFloats&) = delete;
    PinnedFloats& operator=(const PinnedFloats&) = delete;
    float* data() { return data_; }
    float at(int index) const { return data_[index]; }

private:
    float* data_ = nullptr;
};

void read_back(
    const thrust::device_vector<float>& source, int count,
    PinnedFloats& destination, cudaStream_t stream);

struct EnergyReadout {
    float total = 0.0f;
    float point_to_point = 0.0f;
    float point_to_plane = 0.0f;
    float smoothness = 0.0f;
    float anchor = 0.0f;
    float landmark = 0.0f;
};

EnergyReadout read_energies(
    const thrust::device_vector<float>& energies, PinnedFloats& staging,
    cudaStream_t stream);

// The stream the whole route runs on. One stream means the device work is one
// ordered pipeline, so the only synchronization the schedule pays for is the one
// it asks for.
//
// It is a *blocking* stream rather than a non-blocking one, and that is
// deliberate: the thrust upload helpers (device_vector::assign, the
// device-to-host copies that end a level) run on the legacy default stream,
// which only orders itself against blocking streams. With a non-blocking stream
// those copies would be free to overtake the kernels that produce their input.
class StreamGuard {
public:
    StreamGuard() {
        throw_on_cuda_error(cudaStreamCreate(&stream_));
    }
    ~StreamGuard() {
        if (stream_ != nullptr) {
            cudaStreamDestroy(stream_);
        }
    }
    StreamGuard(const StreamGuard&) = delete;
    StreamGuard& operator=(const StreamGuard&) = delete;
    cudaStream_t get() const { return stream_; }

private:
    cudaStream_t stream_ = nullptr;
};

void upload_state(
    const Deformation& deformation, thrust::device_vector<Mat3f>& rotations,
    thrust::device_vector<float>& translations);

Deformation download_state(
    const thrust::device_vector<Mat3f>& rotations,
    const thrust::device_vector<float>& translations,
    const std::vector<Vec3>& node_rest_positions);

}  // namespace gpu
}  // namespace nricp
