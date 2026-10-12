#include "cuda/local_global.cuh"

#include "cuda/cuda_error.h"
#include "cuda/solver_common.cuh"

namespace nricp {
namespace gpu {

// How much of the fitted rotation the local step takes.
//
// Taking all of it lets the rotations run ahead of the translations they were
// fitted against: on the benchmark pairs that costs cat0->cat3 its foldover gate
// (1.37% against 1.32% allowed) and lands every pair 1 to 4% worse. A fifth of
// the step removes that, without the extra inner iterations a smaller value
// needs. Measured over the four pairs at 0.05, 0.1, 0.2, 0.3, 0.5, 0.7 and 1.0:
// everything at or below 0.5 passes the foldover gate, and 0.2 reaches the best
// metrics of the lot at a third of the wall time of 0.1.
constexpr float kRotationRelaxation = 0.2f;

// ---------------------------------------------------------------------------
// Horn's quaternion
// ---------------------------------------------------------------------------
//
// The local step is a weighted fit of one rotation to point pairs (x, y): the
// R that minimises sum w |R x - y|^2. Its solution is the eigenvector of the
// largest eigenvalue of the symmetric 4x4 Horn builds from the correlation
// sum w x y^T, and going through a quaternion rather than through a singular
// value decomposition is what makes the result a proper rotation (det = +1)
// without a reflection branch: a unit quaternion always is one.

// Adds one weighted outer product to a 3x3 correlation, row major.
__device__ inline void accumulate_correlation(float* correlation, float weight,
                                              const Vec3f& source,
                                              const Vec3f& target) {
    const float x[3] = {source.x, source.y, source.z};
    const float y[3] = {target.x, target.y, target.z};
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            correlation[row * 3 + column] += weight * x[row] * y[column];
        }
    }
}

// The largest eigenvector of the symmetric 4x4 Horn's method builds from the
// correlation, by cyclic Jacobi rotations.
//
// A 4x4 needs a handful of sweeps to converge and the sweep order is fixed, so
// the fit is deterministic -- which the schedule's accept/reject test leans on:
// the same state has to produce the same rotation on every run. A correlation
// of zero (a node with no influences) leaves the matrix zero, and the routine
// then returns the identity, which is the only rotation a node with nothing to
// fit can be given.
__device__ inline void horn_quaternion(const float* correlation, float quaternion[4]) {
    const float sxx = correlation[0], sxy = correlation[1], sxz = correlation[2];
    const float syx = correlation[3], syy = correlation[4], syz = correlation[5];
    const float szx = correlation[6], szy = correlation[7], szz = correlation[8];
    float a[4][4];
    a[0][0] = sxx + syy + szz;
    a[0][1] = syz - szy;
    a[0][2] = szx - sxz;
    a[0][3] = sxy - syx;
    a[1][1] = sxx - syy - szz;
    a[1][2] = sxy + syx;
    a[1][3] = szx + sxz;
    a[2][2] = -sxx + syy - szz;
    a[2][3] = syz + szy;
    a[3][3] = -sxx - syy + szz;
    for (int row = 1; row < 4; ++row) {
        for (int column = 0; column < row; ++column) {
            a[row][column] = a[column][row];
        }
    }
    float vectors[4][4];
    for (int row = 0; row < 4; ++row) {
        for (int column = 0; column < 4; ++column) {
            vectors[row][column] = row == column ? 1.0f : 0.0f;
        }
    }
    for (int sweep = 0; sweep < 12; ++sweep) {
        float off = 0.0f;
        for (int row = 0; row < 4; ++row) {
            for (int column = row + 1; column < 4; ++column) {
                off += a[row][column] * a[row][column];
            }
        }
        if (off <= 0.0f) {
            break;
        }
        for (int p = 0; p < 3; ++p) {
            for (int q = p + 1; q < 4; ++q) {
                if (a[p][q] == 0.0f) {
                    continue;
                }
                const float theta = (a[q][q] - a[p][p]) / (2.0f * a[p][q]);
                const float sign = theta >= 0.0f ? 1.0f : -1.0f;
                const float t = sign / (fabsf(theta) + sqrtf(theta * theta + 1.0f));
                const float cosine = 1.0f / sqrtf(t * t + 1.0f);
                const float sine = t * cosine;
                for (int k = 0; k < 4; ++k) {
                    const float first = a[k][p];
                    const float second = a[k][q];
                    a[k][p] = cosine * first - sine * second;
                    a[k][q] = sine * first + cosine * second;
                }
                for (int k = 0; k < 4; ++k) {
                    const float first = a[p][k];
                    const float second = a[q][k];
                    a[p][k] = cosine * first - sine * second;
                    a[q][k] = sine * first + cosine * second;
                }
                for (int k = 0; k < 4; ++k) {
                    const float first = vectors[k][p];
                    const float second = vectors[k][q];
                    vectors[k][p] = cosine * first - sine * second;
                    vectors[k][q] = sine * first + cosine * second;
                }
            }
        }
    }
    int best = 0;
    for (int index = 1; index < 4; ++index) {
        if (a[index][index] > a[best][best]) {
            best = index;
        }
    }
    for (int k = 0; k < 4; ++k) {
        quaternion[k] = vectors[k][best];
    }
}

// The rotation a quaternion represents, row major. The quaternion is normalized
// first, because Jacobi leaves the eigenvector at whatever length it converged
// to, and the formula below assumes a unit one.
__device__ inline Mat3f rotation_from_quaternion(const float quaternion[4]) {
    Mat3f rotation;
    const float length = sqrtf(quaternion[0] * quaternion[0] +
                               quaternion[1] * quaternion[1] +
                               quaternion[2] * quaternion[2] +
                               quaternion[3] * quaternion[3]);
    if (!(length > 0.0f)) {
        return rotation;
    }
    const float w = quaternion[0] / length;
    const float x = quaternion[1] / length;
    const float y = quaternion[2] / length;
    const float z = quaternion[3] / length;
    rotation.m[0] = w * w + x * x - y * y - z * z;
    rotation.m[1] = 2.0f * (x * y - w * z);
    rotation.m[2] = 2.0f * (x * z + w * y);
    rotation.m[3] = 2.0f * (x * y + w * z);
    rotation.m[4] = w * w - x * x + y * y - z * z;
    rotation.m[5] = 2.0f * (y * z - w * x);
    rotation.m[6] = 2.0f * (x * z - w * y);
    rotation.m[7] = 2.0f * (y * z + w * x);
    rotation.m[8] = w * w - x * x - y * y + z * z;
    return rotation;
}

// ---------------------------------------------------------------------------
// The local step
// ---------------------------------------------------------------------------
//
// One block per node, walking that node's own influences: the vertices bound to
// it, and its smoothness partners. Nothing is atomic and nothing is written
// until the correlation is complete.
//
// The targets are the ones the energy asks for: a bound vertex's correspondence
// point, a partner node's own transform of the shared edge, and a marker's
// landmark target. Measured against the alternative, taking the other nodes'
// contributions out of a vertex's target first -- the exact block step, which
// divides the residual by the node's share of the vertex -- needs rotations two
// times as large for the same fit and lands 5 to 10% worse on the benchmark
// pairs: the share is a quarter or a half, so the exact step asks one node to
// move four times as far as its own contribution warrants.
//
// Two terms are deliberately outside the fit. The plane term's rotation
// dependence is rank one (only the normal direction constrains the node) and
// has no closed-form point fit; its weight is added to the point term's instead,
// because at a closest point the two ask the same vertex for the same
// direction. The anchor's rotation row is a penalty on the rotation vector
// rather than a correspondence. Both still enter the energy, and the schedule's
// accept/reject test is what keeps them honest.
//
// The fit reads no rotation -- its targets come from the template, the
// correspondences and the translations -- so it can write its result straight
// into the rotations it replaces; there is no node whose fit would see another
// node's new rotation.
static __global__ void fit_rotations_kernel(
    const float* __restrict__ points, const float* __restrict__ node_rest,
    const int* __restrict__ binding_nodes, const float* __restrict__ binding_weights,
    const int* __restrict__ binding_counts, const float* __restrict__ translations,
    const float* __restrict__ correspondence_points,
    const unsigned char* __restrict__ rejection,
    const float* __restrict__ robust_weights, const unsigned char* __restrict__ mask,
    const int* __restrict__ gather_begin, const int* __restrict__ gather_vertex,
    const float* __restrict__ landmark_flag, const float* __restrict__ landmark_target,
    const int* __restrict__ edge_nodes, const int* __restrict__ node_edge_begin,
    const int* __restrict__ node_edge_list, float point_weight,
    float smoothness_weight, float landmark_weight, float relaxation,
    int node_count, Mat3f* __restrict__ rotations) {
    const int node = blockIdx.x;
    if (node >= node_count) {
        return;
    }
    float correlation[9];
    for (int i = 0; i < 9; ++i) {
        correlation[i] = 0.0f;
    }
    const Vec3f rest_node = load_vec3(node_rest, node);
    const Vec3f translation_node = load_vec3(translations, node);

    const int begin = gather_begin[node];
    const int end = gather_begin[node + 1];
    for (int index = begin + threadIdx.x; index < end; index += blockDim.x) {
        const int vertex = gather_vertex[index];
        const int count = binding_counts[vertex];
        const Vec3f point = load_vec3(points, vertex);
        // How much of the vertex this node owns. It weights the node's rotation
        // by the share of the vertex that rotation actually moves.
        float own_weight = 0.0f;
        for (int slot = 0; slot < count; ++slot) {
            if (binding_nodes[vertex * 4 + slot] == node) {
                own_weight += binding_weights[vertex * 4 + slot];
            }
        }
        if (own_weight == 0.0f) {
            continue;
        }
        const Vec3f source = load_vec3(points, vertex) - rest_node;
        if (rejection[vertex] == 0 && (mask == nullptr || mask[vertex] != 0)) {
            const Vec3f target =
                load_vec3(correspondence_points, vertex) - rest_node - translation_node;
            accumulate_correlation(correlation,
                                   point_weight * robust_weights[vertex] * own_weight,
                                   source, target);
        }
        if (landmark_weight > 0.0f && landmark_flag[vertex] != 0.0f) {
            const Vec3f target =
                load_vec3(landmark_target, vertex) - rest_node - translation_node;
            accumulate_correlation(correlation, landmark_weight * own_weight, source,
                                   target);
        }
    }
    for (int index = node_edge_begin[node] + threadIdx.x;
         index < node_edge_begin[node + 1]; index += blockDim.x) {
        const int edge = node_edge_list[index];
        const int node_a = edge_nodes[edge * 2 + 0];
        // The smoothness residual of an edge is the first endpoint's transform
        // of the shared node position against the second endpoint's own: it
        // depends on the *first* node's rotation and on neither node's rotation
        // otherwise (the specification's J with respect to the second node's
        // rotation is zero). A node that is only ever the second endpoint of its
        // edges is therefore constrained here by the data term alone, exactly as
        // it is in the energy.
        if (node_a != node) {
            continue;
        }
        const int node_b = edge_nodes[edge * 2 + 1];
        const Vec3f rest_b = load_vec3(node_rest, node_b);
        const Vec3f source = rest_b - rest_node;
        const Vec3f target =
            rest_b + load_vec3(translations, node_b) - rest_node - translation_node;
        accumulate_correlation(correlation, smoothness_weight, source, target);
    }

    // One pad word per thread, so the reduction's stride is odd.
    __shared__ float staged[kGatherThreads][10];
    for (int i = 0; i < 9; ++i) {
        staged[threadIdx.x][i] = correlation[i];
    }
    __syncthreads();
    for (int span = blockDim.x / 2; span > 0; span >>= 1) {
        if (threadIdx.x < span) {
            for (int i = 0; i < 9; ++i) {
                staged[threadIdx.x][i] += staged[threadIdx.x + span][i];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float quaternion[4];
        horn_quaternion(&staged[0][0], quaternion);
        const Mat3f fitted = rotation_from_quaternion(quaternion);
        if (relaxation >= 1.0f) {
            rotations[node] = fitted;
            return;
        }
        // Under-relaxed: the fit is a heuristic (it cannot see the plane term or
        // the anchor's rotation row), so taking all of it at once lets the
        // rotations run ahead of the translations they are fitted against. The
        // step is scaled in the tangent space, which is where the rotations live.
        const Mat3f current = rotations[node];
        Mat3f transposed;
        for (int row = 0; row < 3; ++row) {
            for (int column = 0; column < 3; ++column) {
                transposed.m[row * 3 + column] = current.m[column * 3 + row];
            }
        }
        const Vec3f omega = gpu::log_so3(fitted * transposed);
        rotations[node] = gpu::exp_so3(omega * relaxation) * current;
    }
}

// ---------------------------------------------------------------------------
// The global step's right-hand side
// ---------------------------------------------------------------------------
//
// The one part of the system that does depend on the rotations, so it is
// rebuilt for every solve while the matrix is not: -J^T r for the data,
// smoothness, anchor and landmark terms, in the same six-values-per-node layout
// as everything else. The rotation half is written as exactly zero, which is
// what makes the solver's increment exactly zero there.
//
// One block per node again, over the same two lists as the local step, for the
// same reason: a node's rows are then computed by the block that stores them,
// with no atomic in the pass.
static __global__ void translation_rhs_kernel(
    const float* __restrict__ points, const float* __restrict__ node_rest,
    const int* __restrict__ binding_nodes, const float* __restrict__ binding_weights,
    const int* __restrict__ binding_counts, const Mat3f* __restrict__ rotations,
    const float* __restrict__ translations,
    const float* __restrict__ correspondence_points,
    const float* __restrict__ correspondence_normals,
    const unsigned char* __restrict__ rejection,
    const float* __restrict__ robust_weights, const unsigned char* __restrict__ mask,
    const int* __restrict__ gather_begin, const int* __restrict__ gather_vertex,
    const float* __restrict__ landmark_flag, const float* __restrict__ landmark_target,
    const int* __restrict__ edge_nodes, const int* __restrict__ node_edge_begin,
    const int* __restrict__ node_edge_list, float point_weight, float plane_weight,
    float smoothness_weight, float landmark_weight, float anchor_weight,
    int node_count, float* __restrict__ rhs) {
    const int node = blockIdx.x;
    if (node >= node_count) {
        return;
    }
    float local[3] = {0.0f, 0.0f, 0.0f};
    const Vec3f rest_node = load_vec3(node_rest, node);
    const Vec3f translation_node = load_vec3(translations, node);

    const int begin = gather_begin[node];
    const int end = gather_begin[node + 1];
    for (int index = begin + threadIdx.x; index < end; index += blockDim.x) {
        const int vertex = gather_vertex[index];
        const int count = binding_counts[vertex];
        const Vec3f point = load_vec3(points, vertex);
        Vec3f deformed;
        float own_weight = 0.0f;
        for (int slot = 0; slot < count; ++slot) {
            const int other = binding_nodes[vertex * 4 + slot];
            const float weight = binding_weights[vertex * 4 + slot];
            if (other < 0 || weight == 0.0f) {
                continue;
            }
            const Vec3f rest = load_vec3(node_rest, other);
            deformed += (rotations[other] * (point - rest) + rest +
                         load_vec3(translations, other)) *
                        weight;
            if (other == node) {
                own_weight += weight;
            }
        }
        if (own_weight == 0.0f) {
            continue;
        }
        const float robust = robust_weights[vertex];
        if (rejection[vertex] == 0 && (mask == nullptr || mask[vertex] != 0)) {
            const Vec3f offset = deformed - load_vec3(correspondence_points, vertex);
            if (point_weight > 0.0f) {
                const float scale = point_weight * robust * own_weight;
                local[0] -= scale * offset.x;
                local[1] -= scale * offset.y;
                local[2] -= scale * offset.z;
            }
            if (plane_weight > 0.0f) {
                const Vec3f normal = load_vec3(correspondence_normals, vertex);
                const float scale = plane_weight * robust * own_weight * dot(normal, offset);
                local[0] -= scale * normal.x;
                local[1] -= scale * normal.y;
                local[2] -= scale * normal.z;
            }
        }
        if (landmark_weight > 0.0f && landmark_flag[vertex] != 0.0f) {
            const Vec3f to_marker = deformed - load_vec3(landmark_target, vertex);
            const float scale = landmark_weight * own_weight;
            local[0] -= scale * to_marker.x;
            local[1] -= scale * to_marker.y;
            local[2] -= scale * to_marker.z;
        }
    }
    for (int index = node_edge_begin[node] + threadIdx.x;
         index < node_edge_begin[node + 1]; index += blockDim.x) {
        const int edge = node_edge_list[index];
        const int node_a = edge_nodes[edge * 2 + 0];
        const int node_b = edge_nodes[edge * 2 + 1];
        // The residual is the same vector whichever endpoint is being
        // assembled -- the first node's transform of the second node's position
        // against the second node's own -- so it is evaluated once, from the
        // first node's rotation, and not from whichever endpoint is walking the
        // edge. Only the sign differs: the row is +sqrt(w) I for the first
        // endpoint and -sqrt(w) I for the second.
        const Vec3f rest_a = load_vec3(node_rest, node_a);
        const Vec3f rest_b = load_vec3(node_rest, node_b);
        const Vec3f difference = rotations[node_a] * (rest_b - rest_a) + rest_a +
                                 load_vec3(translations, node_a) - rest_b -
                                 load_vec3(translations, node_b);
        // The first endpoint's row is +sqrt(w) I and the second's is -sqrt(w) I,
        // so the two right-hand sides are the residual with opposite signs.
        const float sign = node_a == node ? 1.0f : -1.0f;
        local[0] -= sign * smoothness_weight * difference.x;
        local[1] -= sign * smoothness_weight * difference.y;
        local[2] -= sign * smoothness_weight * difference.z;
    }

    local[0] -= anchor_weight * translation_node.x;
    local[1] -= anchor_weight * translation_node.y;
    local[2] -= anchor_weight * translation_node.z;

    __shared__ float staged[3][kGatherThreads];
    for (int i = 0; i < 3; ++i) {
        staged[i][threadIdx.x] = local[i];
    }
    __syncthreads();
    for (int span = blockDim.x / 2; span > 0; span >>= 1) {
        if (threadIdx.x < span) {
            for (int i = 0; i < 3; ++i) {
                staged[i][threadIdx.x] += staged[i][threadIdx.x + span];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float* node_rhs = rhs + node * 6;
        for (int i = 0; i < 3; ++i) {
            node_rhs[i] = 0.0f;  // the rotation half: no increment comes from it
            node_rhs[3 + i] = staged[i][0];
        }
    }
}

// ---------------------------------------------------------------------------
// Restricting the Hessian to its translation corner
// ---------------------------------------------------------------------------
//
// The global matrix is the same Hessian the simultaneous solve uses, restricted
// to the translations: the (t, t) corner of J^T W J is sum over the rows of
// J_t^T W J_t whether or not the rotation columns are there as well. So the
// expensive part of the assembly is not duplicated and cannot drift between the
// two routes -- what changes is only which part of each block survives.
//
// A diagonal block keeps its translation corner and gets the identity in the
// rotation half; an off-diagonal block (a smoothness edge or a node pair) keeps
// its translation corner and loses everything else.
static __global__ void restrict_diagonal_kernel(Mat6f* __restrict__ diagonal,
                                                int node_count) {
    const int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= node_count) {
        return;
    }
    float* block = diagonal[node].m;
    for (int row = 0; row < 6; ++row) {
        for (int column = 0; column < 6; ++column) {
            if (row < 3 && column < 3) {
                block[row * 6 + column] = row == column ? 1.0f : 0.0f;
            } else if (row < 3 || column < 3) {
                block[row * 6 + column] = 0.0f;
            }
        }
    }
}

static __global__ void restrict_offdiagonal_kernel(Mat6f* __restrict__ blocks,
                                                   int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    float* block = blocks[index].m;
    for (int row = 0; row < 6; ++row) {
        for (int column = 0; column < 6; ++column) {
            if (row < 3 || column < 3) {
                block[row * 6 + column] = 0.0f;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Host entry points
// ---------------------------------------------------------------------------

EnergyReadout assemble_translation_matrix(
    GpuLevel& level, const AssembleInputs& inputs, PinnedFloats& staging,
    cudaStream_t stream) {
    const EnergyReadout readout =
        assemble_system(level, inputs, /*with_jacobians=*/true, staging, stream);
    restrict_diagonal_kernel<<<blocks_for(level.node_count), kThreads, 0, stream>>>(
        thrust::raw_pointer_cast(level.diagonal.data()), level.node_count);
    throw_on_cuda_error(cudaGetLastError());
    if (level.total_edge_count > 0) {
        restrict_offdiagonal_kernel<<<blocks_for(level.total_edge_count), kThreads, 0,
                                      stream>>>(
            thrust::raw_pointer_cast(level.edge_blocks.data()),
            level.total_edge_count);
        throw_on_cuda_error(cudaGetLastError());
    }
    return readout;
}

void assemble_translation_rhs(GpuLevel& level, const AssembleInputs& inputs,
                              cudaStream_t stream) {
    if (level.node_count <= 0) {
        return;
    }
    translation_rhs_kernel<<<dim3(level.node_count), kGatherThreads, 0, stream>>>(
        inputs.template_points, thrust::raw_pointer_cast(level.node_rest.data()),
        thrust::raw_pointer_cast(level.binding_nodes.data()),
        thrust::raw_pointer_cast(level.binding_weights.data()),
        thrust::raw_pointer_cast(level.binding_counts.data()),
        thrust::raw_pointer_cast(level.rotations.data()),
        thrust::raw_pointer_cast(level.translations.data()),
        thrust::raw_pointer_cast(level.correspondence_points.data()),
        thrust::raw_pointer_cast(level.correspondence_normals.data()),
        thrust::raw_pointer_cast(level.rejection.data()),
        thrust::raw_pointer_cast(level.robust_weights.data()), inputs.mask,
        thrust::raw_pointer_cast(level.gather_begin.data()),
        thrust::raw_pointer_cast(level.gather_vertex.data()), inputs.landmark_flag,
        inputs.landmark_target,
        thrust::raw_pointer_cast(level.edges.data()),
        thrust::raw_pointer_cast(level.node_edge_begin.data()),
        thrust::raw_pointer_cast(level.node_edge_list.data()), inputs.point_to_point,
        inputs.point_to_plane, inputs.smoothness, inputs.landmark, inputs.anchor,
        level.node_count, thrust::raw_pointer_cast(level.rhs.data()));
    throw_on_cuda_error(cudaGetLastError());
}

void fit_rotations(GpuLevel& level, const AssembleInputs& inputs,
                   cudaStream_t stream) {
    if (level.node_count <= 0) {
        return;
    }
    // The data fit has one weight for both data terms: they ask the same vertex
    // to sit on the same point, one of them through the plane the point lies in.
    // At a closest point the two agree on the direction, which is what makes
    // folding them together a fit rather than an approximation of one term by
    // another.
    const float data_weight = inputs.point_to_point + inputs.point_to_plane;
    fit_rotations_kernel<<<dim3(level.node_count), kGatherThreads, 0, stream>>>(
        inputs.template_points, thrust::raw_pointer_cast(level.node_rest.data()),
        thrust::raw_pointer_cast(level.binding_nodes.data()),
        thrust::raw_pointer_cast(level.binding_weights.data()),
        thrust::raw_pointer_cast(level.binding_counts.data()),
        thrust::raw_pointer_cast(level.translations.data()),
        thrust::raw_pointer_cast(level.correspondence_points.data()),
        thrust::raw_pointer_cast(level.rejection.data()),
        thrust::raw_pointer_cast(level.robust_weights.data()), inputs.mask,
        thrust::raw_pointer_cast(level.gather_begin.data()),
        thrust::raw_pointer_cast(level.gather_vertex.data()), inputs.landmark_flag,
        inputs.landmark_target, thrust::raw_pointer_cast(level.edges.data()),
        thrust::raw_pointer_cast(level.node_edge_begin.data()),
        thrust::raw_pointer_cast(level.node_edge_list.data()), data_weight,
        inputs.smoothness, inputs.landmark, kRotationRelaxation, level.node_count,
        thrust::raw_pointer_cast(level.rotations.data()));
    throw_on_cuda_error(cudaGetLastError());
}

}  // namespace gpu
}  // namespace nricp
