#include "cuda/gpu_level.cuh"

#include <algorithm>

#include "cuda/solver_common.cuh"

namespace nricp {
namespace gpu {

void reserve_level(GpuLevel& state, int max_nodes, int max_edges, int vertex_count,
                   int pcg_max_iterations) {
    max_nodes = std::max(max_nodes, 1);
    max_edges = std::max(max_edges, 1);
    state.node_rest = thrust::device_vector<float>(max_nodes * 3);
    state.binding_nodes = thrust::device_vector<int>(vertex_count * 4);
    state.binding_weights = thrust::device_vector<float>(vertex_count * 4);
    state.binding_counts = thrust::device_vector<int>(vertex_count);
    state.edges = thrust::device_vector<int>(max_edges * 2);
    state.gather_begin = thrust::device_vector<int>(max_nodes + 1);
    state.gather_cursor = thrust::device_vector<int>(max_nodes);
    state.gather_counts = thrust::device_vector<int>(max_nodes);
    state.gather_vertex = thrust::device_vector<int>(vertex_count * 4);
    // An edge has two endpoints, so this is the most entries any level can
    // produce.
    state.node_edge_begin = thrust::device_vector<int>(max_nodes + 1);
    state.node_edge_cursor = thrust::device_vector<int>(max_nodes);
    state.node_edge_counts = thrust::device_vector<int>(max_nodes);
    state.node_edge_list = thrust::device_vector<int>(max_edges * 2);
    // A vertex has at most six slot pairs, so this is the most entries any level
    // can emit and the most groups it can produce.
    const std::size_t pair_capacity = static_cast<std::size_t>(vertex_count) * 6;
    state.pair_keys = thrust::device_vector<unsigned long long>(pair_capacity);
    state.pair_payload = thrust::device_vector<unsigned long long>(pair_capacity);
    state.pair_vertex_begin = thrust::device_vector<int>(vertex_count + 1);
    state.pair_flags = thrust::device_vector<int>(pair_capacity);
    state.pair_group_of_entry = thrust::device_vector<int>(pair_capacity);
    state.pair_group_begin = thrust::device_vector<int>(pair_capacity + 1);
    state.rotations = thrust::device_vector<Mat3f>(max_nodes);
    state.translations = thrust::device_vector<float>(max_nodes * 3);
    state.rotations_backup = thrust::device_vector<Mat3f>(max_nodes);
    state.translations_backup = thrust::device_vector<float>(max_nodes * 3);
    state.deformed = thrust::device_vector<float>(vertex_count * 3);
    state.query_normals = thrust::device_vector<float>(vertex_count * 3);
    state.correspondence_points = thrust::device_vector<float>(vertex_count * 3);
    state.correspondence_normals = thrust::device_vector<float>(vertex_count * 3);
    state.correspondence_distances = thrust::device_vector<float>(vertex_count);
    state.correspondence_triangles = thrust::device_vector<int>(vertex_count);
    state.rejection = thrust::device_vector<unsigned char>(vertex_count, 0);
    state.robust_weights = thrust::device_vector<float>(vertex_count, 1.0f);
    state.diagonal = thrust::device_vector<Mat6f>(max_nodes);
    state.edge_blocks = thrust::device_vector<Mat6f>(max_edges);
    state.factor = thrust::device_vector<Mat6f>(max_nodes);
    state.rhs = thrust::device_vector<float>(static_cast<std::size_t>(max_nodes) * 6);
    state.delta = thrust::device_vector<float>(static_cast<std::size_t>(max_nodes) * 6);
    state.pcg_workspace = thrust::device_vector<float>(
        PcgWorkspace::floats_for(max_nodes, pcg_max_iterations));
    state.pcg = thrust::device_vector<PcgState>(1);
    state.energies = thrust::device_vector<float>(kEnergyKinds);
    state.rejection_counts = thrust::device_vector<int>(kRejectionKinds);
    state.distance_sum = thrust::device_vector<float>(1);
    state.sorted_distances = thrust::device_vector<float>(vertex_count);
    state.statistics = thrust::device_vector<float>(3);
}

PcgSystem pcg_system_of(GpuLevel& level) {
    PcgSystem system;
    system.diagonal = thrust::raw_pointer_cast(level.diagonal.data());
    system.factor = thrust::raw_pointer_cast(level.factor.data());
    system.edge_blocks = thrust::raw_pointer_cast(level.edge_blocks.data());
    system.edges = thrust::raw_pointer_cast(level.edges.data());
    system.rhs = thrust::raw_pointer_cast(level.rhs.data());
    system.delta = thrust::raw_pointer_cast(level.delta.data());
    system.workspace = thrust::raw_pointer_cast(level.pcg_workspace.data());
    system.device_state = thrust::raw_pointer_cast(level.pcg.data());
    system.node_count = level.node_count;
    // The block solver sees the graph's edges and the node-pair groups as one
    // list: a group is an edge with two nodes.
    system.edge_count = level.total_edge_count;
    return system;
}

}  // namespace gpu
}  // namespace nricp
