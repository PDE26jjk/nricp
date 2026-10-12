#include "cuda/host_staging.cuh"

#include <algorithm>
#include <cstddef>
#include <vector>

#include <cooperative_groups.h>
#include <thrust/copy.h>

#include "cuda/cuda_error.h"
#include "cuda/solver_common.cuh"

namespace nricp {
namespace gpu {

// ---------------------------------------------------------------------------
// Host-side plumbing
// ---------------------------------------------------------------------------
thrust::device_vector<float> upload_vectors(const std::vector<Vec3>& values) {
    thrust::device_vector<float> out(values.size() * 3);
    std::vector<float> flat(values.size() * 3);
    for (std::size_t i = 0; i < values.size(); ++i) {
        flat[i * 3 + 0] = static_cast<float>(values[i].x);
        flat[i * 3 + 1] = static_cast<float>(values[i].y);
        flat[i * 3 + 2] = static_cast<float>(values[i].z);
    }
    if (!flat.empty()) {
        out.assign(flat.begin(), flat.end());
    }
    return out;
}

std::vector<Vec3> download_vectors(const thrust::device_vector<float>& values) {
    std::vector<float> flat(values.size());
    if (!flat.empty()) {
        thrust::copy(values.begin(), values.end(), flat.begin());
    }
    std::vector<Vec3> out(flat.size() / 3);
    for (std::size_t i = 0; i < out.size(); ++i) {
        out[i] = {flat[i * 3 + 0], flat[i * 3 + 1], flat[i * 3 + 2]};
    }
    return out;
}

thrust::device_vector<int> upload_ints(const std::vector<int>& values) {
    thrust::device_vector<int> out(values.size());
    if (!values.empty()) {
        out.assign(values.begin(), values.end());
    }
    return out;
}

thrust::device_vector<float> upload_floats(const std::vector<float>& values) {
    thrust::device_vector<float> out(values.size());
    if (!values.empty()) {
        out.assign(values.begin(), values.end());
    }
    return out;
}

std::vector<int> flatten_binding_nodes(const std::vector<VertexBinding>& bindings) {
    std::vector<int> flat(bindings.size() * 4);
    for (std::size_t i = 0; i < bindings.size(); ++i) {
        for (int slot = 0; slot < 4; ++slot) {
            flat[i * 4 + slot] = bindings[i].nodes[slot];
        }
    }
    return flat;
}

std::vector<float> flatten_binding_weights(
    const std::vector<VertexBinding>& bindings) {
    std::vector<float> flat(bindings.size() * 4);
    for (std::size_t i = 0; i < bindings.size(); ++i) {
        for (int slot = 0; slot < 4; ++slot) {
            flat[i * 4 + slot] = static_cast<float>(bindings[i].weights[slot]);
        }
    }
    return flat;
}

// The only place the schedule waits for the device. Everything a decision needs
// has been reduced to a handful of scalars by then.
void read_back(
    const thrust::device_vector<float>& source, int count, PinnedFloats& destination,
    cudaStream_t stream) {
    if (count <= 0 || source.size() == 0) {
        return;
    }
    throw_on_cuda_error(cudaMemcpyAsync(
        destination.data(), thrust::raw_pointer_cast(source.data()),
        sizeof(float) * static_cast<std::size_t>(count), cudaMemcpyDeviceToHost,
        stream));
    throw_on_cuda_error(cudaStreamSynchronize(stream));
}

void upload_state(
    const Deformation& deformation, thrust::device_vector<Mat3f>& rotations,
    thrust::device_vector<float>& translations) {
    std::vector<Mat3f> host_rotations(deformation.nodes.size());
    std::vector<float> host_translations(deformation.nodes.size() * 3);
    for (std::size_t i = 0; i < deformation.nodes.size(); ++i) {
        for (int k = 0; k < 9; ++k) {
            host_rotations[i].m[k] =
                static_cast<float>(deformation.nodes[i].rotation.m[k]);
        }
        host_translations[i * 3 + 0] =
            static_cast<float>(deformation.nodes[i].translation.x);
        host_translations[i * 3 + 1] =
            static_cast<float>(deformation.nodes[i].translation.y);
        host_translations[i * 3 + 2] =
            static_cast<float>(deformation.nodes[i].translation.z);
    }
    if (!host_rotations.empty()) {
        rotations.assign(host_rotations.begin(), host_rotations.end());
        translations.assign(host_translations.begin(), host_translations.end());
    }
}

// Rebuilds the host-side deformation from the device state. This is the only
// download of the transforms in a level, and it happens at the level boundary,
// where the next level's warm start needs them.
Deformation download_state(
    const thrust::device_vector<Mat3f>& rotations,
    const thrust::device_vector<float>& translations,
    const std::vector<Vec3>& node_rest_positions) {
    std::vector<Mat3f> host_rotations(rotations.size());
    std::vector<float> host_translations(translations.size());
    if (!host_rotations.empty()) {
        thrust::copy(rotations.begin(), rotations.end(), host_rotations.begin());
    }
    if (!host_translations.empty()) {
        thrust::copy(translations.begin(), translations.end(), host_translations.begin());
    }
    Deformation deformation = identity_deformation(node_rest_positions);
    const std::size_t count = std::min(deformation.nodes.size(), host_rotations.size());
    for (std::size_t i = 0; i < count; ++i) {
        for (int k = 0; k < 9; ++k) {
            deformation.nodes[i].rotation.m[k] = host_rotations[i].m[k];
        }
        deformation.nodes[i].translation = {host_translations[i * 3 + 0],
                                            host_translations[i * 3 + 1],
                                            host_translations[i * 3 + 2]};
    }
    return deformation;
}

EnergyReadout read_energies(
    const thrust::device_vector<float>& energies, PinnedFloats& staging,
    cudaStream_t stream) {
    read_back(energies, kEnergyKinds, staging, stream);
    EnergyReadout out;
    out.point_to_point = staging.at(0);
    out.point_to_plane = staging.at(1);
    out.smoothness = staging.at(2);
    out.anchor = staging.at(3);
    out.landmark = staging.at(4);
    out.total = out.point_to_point + out.point_to_plane + out.smoothness +
                out.anchor + out.landmark;
    return out;
}

}  // namespace gpu
}  // namespace nricp
