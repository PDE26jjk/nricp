#include "cuda/deform_ops.cuh"

#include <vector>

#include <thrust/fill.h>
#include <thrust/sort.h>
#include <thrust/system/cuda/execution_policy.h>

#include "cuda/cuda_error.h"
#include "cuda/solver_common.cuh"

namespace nricp {
namespace gpu {

static __global__ void evaluate_kernel(
    const float* points, const int* binding_nodes, const float* binding_weights,
    const int* binding_counts, const Mat3f* rotations, const float* translations,
    const float* node_rest, int vertex_count, float* out) {
    const int vertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex >= vertex_count) {
        return;
    }
    const Vec3f point = load_vec3(points, vertex);
    const int count = binding_counts[vertex];
    Vec3f sum;
    for (int slot = 0; slot < count; ++slot) {
        const int node = binding_nodes[vertex * 4 + slot];
        const float weight = binding_weights[vertex * 4 + slot];
        if (weight == 0.0f || node < 0) {
            continue;
        }
        const Vec3f rest = load_vec3(node_rest, node);
        const Vec3f local = point - rest;
        const Vec3f translation = load_vec3(translations, node);
        sum += (rotations[node] * local + rest + translation) * weight;
    }
    store_vec3(out, vertex, sum);
}

// Backs the accepted state up before a step is attempted, and puts it back when
// the step is rejected. One kernel for both directions: the caller swaps the
// buffers.
static __global__ void copy_state_kernel(
    const Mat3f* __restrict__ rotations_in, Mat3f* __restrict__ rotations_out,
    const float* __restrict__ translations_in, float* __restrict__ translations_out,
    int node_count) {
    const int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= node_count) {
        return;
    }
    rotations_out[node] = rotations_in[node];
    for (int axis = 0; axis < 3; ++axis) {
        translations_out[node * 3 + axis] = translations_in[node * 3 + axis];
    }
}

// The tangent-space increment, applied in place: R <- exp([w]) R and t <- t + u.
// The same update core/deformation.cpp's retract() applies on the host, so the
// device walks the trajectory the specification describes.
static __global__ void retract_kernel(
    Mat3f* __restrict__ rotations, float* __restrict__ translations,
    const float* __restrict__ delta, int node_count) {
    const int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= node_count) {
        return;
    }
    const float* values = delta + node * 6;
    const Vec3f omega{values[0], values[1], values[2]};
    rotations[node] = gpu::exp_so3(omega) * rotations[node];
    float* translation = translations + node * 3;
    translation[0] += values[3];
    translation[1] += values[4];
    translation[2] += values[5];
}

// ---------------------------------------------------------------------------
// Deformed template normals
// ---------------------------------------------------------------------------

// Area-weighted vertex normals of the deformed template, the spec's n'_i. The
// host used to evaluate the deformed mesh and compute these every outer
// iteration; doing it here is what takes that download out of the loop.
static __global__ void accumulate_face_normals_kernel(
    const float* __restrict__ points, const int* __restrict__ faces,
    int triangle_count, float* __restrict__ normals) {
    const int triangle = blockIdx.x * blockDim.x + threadIdx.x;
    if (triangle >= triangle_count) {
        return;
    }
    const int* corners = faces + triangle * 3;
    const Vec3f a = load_vec3(points, corners[0]);
    const Vec3f b = load_vec3(points, corners[1]);
    const Vec3f c = load_vec3(points, corners[2]);
    const Vec3f weighted = cross(b - a, c - a);
    for (int corner = 0; corner < 3; ++corner) {
        float* slot = normals + corners[corner] * 3;
        atomicAdd(&slot[0], weighted.x);
        atomicAdd(&slot[1], weighted.y);
        atomicAdd(&slot[2], weighted.z);
    }
}

static __global__ void normalize_vectors_kernel(float* __restrict__ data, int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    store_vec3(data, index, normalized(load_vec3(data, index)));
}

// ---------------------------------------------------------------------------
// Correspondence filtering and reweighting
// ---------------------------------------------------------------------------

static __global__ void huber_weights_kernel(
    const float* deformed, const float* targets, const unsigned char* rejection,
    int count, float threshold, float* weights) {
    const int vertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex >= count) {
        return;
    }
    if (rejection[vertex] != 0 || threshold <= 0.0f) {
        weights[vertex] = 1.0f;
        return;
    }
    const float squared = squared_norm(load_vec3(deformed, vertex) -
                                       load_vec3(targets, vertex));
    const float squared_threshold = threshold * threshold;
    weights[vertex] = squared <= squared_threshold ? 1.0f : threshold / sqrtf(squared);
}

// Tallies why correspondences were dropped and sums the accepted distances. The
// report's percentiles need the distances ordered, which is a device sort at the
// end of the level; everything here is a plain atomic tally.
static __global__ void correspondence_counts_kernel(
    const unsigned char* __restrict__ rejection, const float* __restrict__ distances,
    int vertex_count, int* __restrict__ counts, float* __restrict__ distance_sum) {
    const int vertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex >= vertex_count) {
        return;
    }
    const unsigned char code = rejection[vertex];
    if (code == 0) {
        atomicAdd(&counts[0], 1);
        atomicAdd(distance_sum, distances[vertex]);
    } else if (code < kRejectionKinds) {
        atomicAdd(&counts[code], 1);
    }
}

// Rejected vertices are parked at infinity, so one sort of the whole array puts
// the accepted distances first, in order.
static __global__ void distances_with_infinity_kernel(
    const unsigned char* __restrict__ rejection, const float* __restrict__ distances,
    int vertex_count, float* __restrict__ out) {
    const int vertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex >= vertex_count) {
        return;
    }
    out[vertex] = rejection[vertex] == 0 ? distances[vertex] : 3.4e38f;
}

// Mean, p95 and max of the accepted distances, from the sorted array and the
// accepted count the tally produced. The percentile picks the same element the
// host implementation does.
static __global__ void percentile_kernel(
    const float* __restrict__ sorted, const int* __restrict__ counts,
    const float* __restrict__ distance_sum, float* __restrict__ stats) {
    const int accepted = counts[0];
    float mean = 0.0f;
    float p95 = 0.0f;
    float maximum = 0.0f;
    if (accepted > 0) {
        mean = distance_sum[0] / static_cast<float>(accepted);
        const int index = min(accepted - 1, static_cast<int>(0.95 * (accepted - 1)));
        p95 = sorted[index];
        maximum = sorted[accepted - 1];
    }
    stats[0] = mean;
    stats[1] = p95;
    stats[2] = maximum;
}

void evaluate_deformation(
    GpuLevel& level, const float* template_points, int vertex_count,
    cudaStream_t stream) {
    evaluate_kernel<<<blocks_for(vertex_count), kThreads, 0, stream>>>(
        template_points, thrust::raw_pointer_cast(level.binding_nodes.data()),
        thrust::raw_pointer_cast(level.binding_weights.data()),
        thrust::raw_pointer_cast(level.binding_counts.data()),
        thrust::raw_pointer_cast(level.rotations.data()),
        thrust::raw_pointer_cast(level.translations.data()),
        thrust::raw_pointer_cast(level.node_rest.data()), vertex_count,
        thrust::raw_pointer_cast(level.deformed.data()));
    throw_on_cuda_error(cudaGetLastError());
}

void build_query_normals(
    GpuLevel& level, const thrust::device_vector<int>& faces, int triangle_count,
    int vertex_count, cudaStream_t stream) {
    thrust::fill(thrust::cuda::par.on(stream), level.query_normals.begin(),
                 level.query_normals.end(), 0.0f);
    accumulate_face_normals_kernel<<<blocks_for(triangle_count), kThreads, 0,
                                     stream>>>(
        thrust::raw_pointer_cast(level.deformed.data()),
        thrust::raw_pointer_cast(faces.data()), triangle_count,
        thrust::raw_pointer_cast(level.query_normals.data()));
    throw_on_cuda_error(cudaGetLastError());
    normalize_vectors_kernel<<<blocks_for(vertex_count), kThreads, 0, stream>>>(
        thrust::raw_pointer_cast(level.query_normals.data()), vertex_count);
    throw_on_cuda_error(cudaGetLastError());
}

void update_robust_weights(
    GpuLevel& level, int vertex_count, float threshold, cudaStream_t stream) {
    huber_weights_kernel<<<blocks_for(vertex_count), kThreads, 0, stream>>>(
        thrust::raw_pointer_cast(level.deformed.data()),
        thrust::raw_pointer_cast(level.correspondence_points.data()),
        thrust::raw_pointer_cast(level.rejection.data()), vertex_count, threshold,
        thrust::raw_pointer_cast(level.robust_weights.data()));
    throw_on_cuda_error(cudaGetLastError());
}

// The correspondence statistics: an atomic tally of the filters and one device
// sort for the percentiles, so the host reads five integers and three floats
// rather than a slice of the correspondence arrays.
CorrespondenceCounts summarize_correspondences(
    GpuLevel& level, int vertex_count, cudaStream_t stream) {

    CorrespondenceCounts out;
    out.total = vertex_count;
    throw_on_cuda_error(cudaMemsetAsync(
        thrust::raw_pointer_cast(level.rejection_counts.data()), 0,
        sizeof(int) * kRejectionKinds, stream));
    throw_on_cuda_error(cudaMemsetAsync(
        thrust::raw_pointer_cast(level.distance_sum.data()), 0, sizeof(float),
        stream));
    correspondence_counts_kernel<<<blocks_for(vertex_count), kThreads, 0, stream>>>(
        thrust::raw_pointer_cast(level.rejection.data()),
        thrust::raw_pointer_cast(level.correspondence_distances.data()), vertex_count,
        thrust::raw_pointer_cast(level.rejection_counts.data()),
        thrust::raw_pointer_cast(level.distance_sum.data()));
    throw_on_cuda_error(cudaGetLastError());
    distances_with_infinity_kernel<<<blocks_for(vertex_count), kThreads, 0, stream>>>(
        thrust::raw_pointer_cast(level.rejection.data()),
        thrust::raw_pointer_cast(level.correspondence_distances.data()), vertex_count,
        thrust::raw_pointer_cast(level.sorted_distances.data()));
    throw_on_cuda_error(cudaGetLastError());

    thrust::sort(thrust::cuda::par.on(stream), level.sorted_distances.begin(),
                 level.sorted_distances.end());

    percentile_kernel<<<1, 1, 0, stream>>>(
        thrust::raw_pointer_cast(level.sorted_distances.data()),
        thrust::raw_pointer_cast(level.rejection_counts.data()),
        thrust::raw_pointer_cast(level.distance_sum.data()),
        thrust::raw_pointer_cast(level.statistics.data()));
    throw_on_cuda_error(cudaGetLastError());
    std::vector<int> host_counts(kRejectionKinds);
    std::vector<float> host_statistics(3);
    throw_on_cuda_error(cudaMemcpyAsync(
        host_counts.data(), thrust::raw_pointer_cast(level.rejection_counts.data()),
        sizeof(int) * kRejectionKinds, cudaMemcpyDeviceToHost, stream));
    throw_on_cuda_error(cudaMemcpyAsync(
        host_statistics.data(), thrust::raw_pointer_cast(level.statistics.data()),
        sizeof(float) * 3, cudaMemcpyDeviceToHost, stream));
    throw_on_cuda_error(cudaStreamSynchronize(stream));
    out.accepted = host_counts[0];
    out.rejected_distance = host_counts[1];
    out.rejected_normal = host_counts[2];
    out.rejected_slope = host_counts[3];
    out.rejected_masked = host_counts[4];
    out.distance_mean = host_statistics[0];
    out.distance_p95 = host_statistics[1];
    out.distance_max = host_statistics[2];
    return out;

}

void save_accepted_state(GpuLevel& level, int node_count, cudaStream_t stream) {
    copy_state_kernel<<<blocks_for(node_count), kThreads, 0, stream>>>(
        thrust::raw_pointer_cast(level.rotations.data()),
        thrust::raw_pointer_cast(level.rotations_backup.data()),
        thrust::raw_pointer_cast(level.translations.data()),
        thrust::raw_pointer_cast(level.translations_backup.data()), node_count);
    throw_on_cuda_error(cudaGetLastError());
}

void restore_accepted_state(GpuLevel& level, int node_count, cudaStream_t stream) {
    copy_state_kernel<<<blocks_for(node_count), kThreads, 0, stream>>>(
        thrust::raw_pointer_cast(level.rotations_backup.data()),
        thrust::raw_pointer_cast(level.rotations.data()),
        thrust::raw_pointer_cast(level.translations_backup.data()),
        thrust::raw_pointer_cast(level.translations.data()), node_count);
    throw_on_cuda_error(cudaGetLastError());
}

void apply_increment(GpuLevel& level, int node_count, cudaStream_t stream) {
    retract_kernel<<<blocks_for(node_count), kThreads, 0, stream>>>(
        thrust::raw_pointer_cast(level.rotations.data()),
        thrust::raw_pointer_cast(level.translations.data()),
        thrust::raw_pointer_cast(level.delta.data()), node_count);
    throw_on_cuda_error(cudaGetLastError());
}

}  // namespace gpu
}  // namespace nricp
