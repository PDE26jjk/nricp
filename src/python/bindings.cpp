// Python entry point of the wrapping solver.
//
// Two kinds of entry point live here. The artifact identity (`build_info`) is
// what every run report carries. The rest are the solver's building blocks:
// normalization, cleaning, sampling, binding, deformation evaluation, level
// transfer, and closest-point queries. They are exposed so callers can prepare their own
// templates and so the verifications can exercise each stage in isolation.
#include "python/numpy_config.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <limits>
#include <string>
#include <vector>

#include "core/build_info.h"
#include "core/constraints.h"
#include "core/deformation.h"
#include "core/graph.h"
#include "cuda/graph_build.cuh"
#include "core/mesh.h"
#include "core/report.h"
#include "core/version.h"
#include "cuda/gpu_solver.cuh"
#include "energy/energy.h"
#include "energy/robust.h"
#include "geometry/closest_point.h"
#include "solver/correspondence.h"
#include "solver/direct_solve.h"
#include "solver/fused_assembly.h"
#include "solver/normal_equations.h"
#include "solver/route_a.h"

namespace py = pybind11;

namespace {

using DoubleArray = py::array_t<double, py::array::c_style | py::array::forcecast>;
using IndexArray =
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>;
using MatrixArray = py::array_t<double, py::array::c_style | py::array::forcecast>;

std::string pybind11_version() {
    return std::to_string(PYBIND11_VERSION_MAJOR) + "." +
           std::to_string(PYBIND11_VERSION_MINOR) + "." +
           std::to_string(PYBIND11_VERSION_PATCH);
}

std::vector<nricp::Vec3> to_points(const DoubleArray& array, const char* name) {
    if (array.ndim() != 2 || array.shape(1) != 3) {
        throw py::value_error(std::string(name) + " must have shape (n, 3)");
    }
    const auto view = array.unchecked<2>();
    std::vector<nricp::Vec3> points(static_cast<std::size_t>(view.shape(0)));
    for (py::ssize_t i = 0; i < view.shape(0); ++i) {
        points[static_cast<std::size_t>(i)] = {view(i, 0), view(i, 1), view(i, 2)};
    }
    return points;
}

py::array_t<double> from_points(const std::vector<nricp::Vec3>& points) {
    py::array_t<double> array(
        {static_cast<py::ssize_t>(points.size()), static_cast<py::ssize_t>(3)});
    auto view = array.mutable_unchecked<2>();
    for (std::size_t i = 0; i < points.size(); ++i) {
        view(static_cast<py::ssize_t>(i), 0) = points[i].x;
        view(static_cast<py::ssize_t>(i), 1) = points[i].y;
        view(static_cast<py::ssize_t>(i), 2) = points[i].z;
    }
    return array;
}

std::vector<nricp::Int3> to_faces(const IndexArray& array, const char* name) {
    if (array.ndim() != 2 || array.shape(1) != 3) {
        throw py::value_error(std::string(name) + " must have shape (n, 3)");
    }
    const auto view = array.unchecked<2>();
    std::vector<nricp::Int3> faces(static_cast<std::size_t>(view.shape(0)));
    for (py::ssize_t i = 0; i < view.shape(0); ++i) {
        faces[static_cast<std::size_t>(i)] = {static_cast<int>(view(i, 0)),
                                              static_cast<int>(view(i, 1)),
                                              static_cast<int>(view(i, 2))};
    }
    return faces;
}

py::array_t<std::int64_t> from_faces(const std::vector<nricp::Int3>& faces) {
    py::array_t<std::int64_t> array(
        {static_cast<py::ssize_t>(faces.size()), static_cast<py::ssize_t>(3)});
    auto view = array.mutable_unchecked<2>();
    for (std::size_t i = 0; i < faces.size(); ++i) {
        view(static_cast<py::ssize_t>(i), 0) = faces[i].v0;
        view(static_cast<py::ssize_t>(i), 1) = faces[i].v1;
        view(static_cast<py::ssize_t>(i), 2) = faces[i].v2;
    }
    return array;
}

py::array_t<std::int64_t> from_indices(const std::vector<int>& indices) {
    py::array_t<std::int64_t> array({static_cast<py::ssize_t>(indices.size())});
    auto view = array.mutable_unchecked<1>();
    for (std::size_t i = 0; i < indices.size(); ++i) {
        view(static_cast<py::ssize_t>(i)) = indices[i];
    }
    return array;
}

std::vector<nricp::Int2> to_edges(const IndexArray& array, const char* name) {
    if (array.ndim() != 2 || array.shape(1) != 2) {
        throw py::value_error(std::string(name) + " must have shape (n, 2)");
    }
    const auto view = array.unchecked<2>();
    std::vector<nricp::Int2> edges(static_cast<std::size_t>(view.shape(0)));
    for (py::ssize_t i = 0; i < view.shape(0); ++i) {
        edges[static_cast<std::size_t>(i)] = {static_cast<int>(view(i, 0)),
                                              static_cast<int>(view(i, 1))};
    }
    return edges;
}

py::array_t<std::int64_t> from_edges(const std::vector<nricp::Int2>& edges) {
    py::array_t<std::int64_t> array(
        {static_cast<py::ssize_t>(edges.size()), static_cast<py::ssize_t>(2)});
    auto view = array.mutable_unchecked<2>();
    for (std::size_t i = 0; i < edges.size(); ++i) {
        view(static_cast<py::ssize_t>(i), 0) = edges[i].a;
        view(static_cast<py::ssize_t>(i), 1) = edges[i].b;
    }
    return array;
}

// Optional per-vertex participation mask; `py::none()` means "no mask".
nricp::VertexMask to_mask(const py::object& mask) {
    nricp::VertexMask vertex_mask;
    if (mask.is_none()) {
        return vertex_mask;
    }
    const IndexArray array = mask.cast<IndexArray>();
    if (array.ndim() != 1) {
        throw py::value_error("mask must have shape (n,)");
    }
    vertex_mask.active.resize(static_cast<std::size_t>(array.shape(0)));
    for (py::ssize_t i = 0; i < array.shape(0); ++i) {
        vertex_mask.active[static_cast<std::size_t>(i)] =
            static_cast<unsigned char>(array.at(i) ? 1 : 0);
    }
    return vertex_mask;
}

double dict_double(const py::dict& values, const char* key, double fallback) {
    if (!values.contains(key)) {
        return fallback;
    }
    return py::cast<double>(values[key]);
}

int dict_int(const py::dict& values, const char* key, int fallback) {
    if (!values.contains(key)) {
        return fallback;
    }
    return py::cast<int>(values[key]);
}

bool dict_bool(const py::dict& values, const char* key, bool fallback) {
    if (!values.contains(key)) {
        return fallback;
    }
    return py::cast<bool>(values[key]);
}

std::string dict_string(
    const py::dict& values, const char* key, const std::string& fallback) {
    if (!values.contains(key)) {
        return fallback;
    }
    return py::cast<std::string>(values[key]);
}

nricp::LevelSettings to_level_settings(const py::dict& values) {
    nricp::LevelSettings settings;
    settings.weights.point_to_point = dict_double(values, "point_to_point", 1.0);
    settings.weights.point_to_plane = dict_double(values, "point_to_plane", 1.0);
    settings.weights.smoothness = dict_double(values, "smoothness", 1.0);
    settings.weights.anchor = dict_double(values, "anchor", 0.0);
    settings.weights.landmark = dict_double(values, "landmark", 1.0);
    settings.robust_kernel = nricp::robust_kernel_from_string(
        dict_string(values, "robust_kernel", "huber"));
    settings.robust_threshold = dict_double(values, "robust_threshold", 0.0);
    settings.max_outer_iterations = dict_int(values, "max_outer_iterations", 10);
    settings.max_inner_iterations = dict_int(values, "max_inner_iterations", 10);
    settings.outer_tolerance = dict_double(values, "outer_tolerance", 1e-3);
    settings.inner_tolerance = dict_double(values, "inner_tolerance", 1e-4);
    return settings;
}

// Everything the energy entry points need, parsed once so the block-based and
// the fused paths are guaranteed to be looking at the same state.
std::vector<nricp::VertexBinding> to_bindings(
    const IndexArray& nodes, const DoubleArray& weights, const IndexArray& counts);

struct EnergyCall {
    nricp::EnergyInputs inputs;
    nricp::EnergyWeights weights;
    nricp::VertexMask mask;
};

EnergyCall to_energy_call(
    const DoubleArray& template_points, const IndexArray& binding_nodes,
    const DoubleArray& binding_weights, const IndexArray& binding_counts,
    const DoubleArray& node_positions, const DoubleArray& correspondence_points,
    const DoubleArray& correspondence_normals,
    const IndexArray& correspondence_accepted,
    const DoubleArray& correspondence_robust, const IndexArray& edges,
    const IndexArray& landmark_indices, const DoubleArray& landmark_targets,
    const py::dict& weights, const py::object& mask) {
    EnergyCall call;
    call.inputs.template_points = to_points(template_points, "template_points");
    call.inputs.bindings =
        to_bindings(binding_nodes, binding_weights, binding_counts);
    call.inputs.node_positions = to_points(node_positions, "node_positions");
    call.inputs.smoothness_edges = to_edges(edges, "edges");

    const std::vector<nricp::Vec3> targets =
        to_points(correspondence_points, "correspondence_points");
    const std::vector<nricp::Vec3> normals =
        to_points(correspondence_normals, "correspondence_normals");
    if (static_cast<std::size_t>(correspondence_accepted.shape(0)) !=
            targets.size() ||
        static_cast<std::size_t>(correspondence_robust.shape(0)) != targets.size()) {
        throw py::value_error("correspondence arrays must agree in length");
    }
    const auto accepted_view = correspondence_accepted.unchecked<1>();
    const auto robust_view = correspondence_robust.unchecked<1>();
    call.inputs.correspondences.resize(targets.size());
    for (std::size_t i = 0; i < targets.size(); ++i) {
        call.inputs.correspondences[i] = {
            targets[i], normals[i], robust_view(static_cast<py::ssize_t>(i)),
            accepted_view(static_cast<py::ssize_t>(i)) != 0};
    }

    const std::vector<nricp::Vec3> landmark_points =
        to_points(landmark_targets, "landmark_targets");
    const auto landmark_view = landmark_indices.unchecked<1>();
    if (static_cast<std::size_t>(landmark_indices.shape(0)) !=
        landmark_points.size()) {
        throw py::value_error("landmark indices and targets must agree in length");
    }
    call.inputs.landmarks.resize(landmark_points.size());
    for (std::size_t i = 0; i < landmark_points.size(); ++i) {
        call.inputs.landmarks[i] = {
            static_cast<int>(landmark_view(static_cast<py::ssize_t>(i))),
            landmark_points[i]};
    }

    call.mask = to_mask(mask);
    // The mask pointer is deliberately set by the caller, after the struct is in
    // its final location: pointing into a struct that is about to be moved would
    // leave it dangling.
    for (const auto& item : weights) {
        const std::string key = py::cast<std::string>(item.first);
        const double value = py::cast<double>(item.second);
        if (key == "point_to_point") {
            call.weights.point_to_point = value;
        } else if (key == "point_to_plane") {
            call.weights.point_to_plane = value;
        } else if (key == "smoothness") {
            call.weights.smoothness = value;
        } else if (key == "anchor") {
            call.weights.anchor = value;
        } else if (key == "landmark") {
            call.weights.landmark = value;
        } else {
            throw py::value_error("unknown energy weight: " + key);
        }
    }
    return call;
}

std::vector<nricp::Mat3> to_matrices(const MatrixArray& array, const char* name) {
    if (array.ndim() != 3 || array.shape(1) != 3 || array.shape(2) != 3) {
        throw py::value_error(std::string(name) + " must have shape (n, 3, 3)");
    }
    const auto view = array.unchecked<3>();
    std::vector<nricp::Mat3> matrices(static_cast<std::size_t>(view.shape(0)));
    for (py::ssize_t i = 0; i < view.shape(0); ++i) {
        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 3; ++col) {
                matrices[static_cast<std::size_t>(i)].at(row, col) = view(i, row, col);
            }
        }
    }
    return matrices;
}

py::array_t<double> from_matrices(const std::vector<nricp::Mat3>& matrices) {
    py::array_t<double> array({static_cast<py::ssize_t>(matrices.size()),
                               static_cast<py::ssize_t>(3),
                               static_cast<py::ssize_t>(3)});
    auto view = array.mutable_unchecked<3>();
    for (std::size_t i = 0; i < matrices.size(); ++i) {
        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 3; ++col) {
                view(static_cast<py::ssize_t>(i), row, col) =
                    matrices[i].at(row, col);
            }
        }
    }
    return array;
}

py::dict from_bindings(const std::vector<nricp::VertexBinding>& bindings) {
    const py::ssize_t count = static_cast<py::ssize_t>(bindings.size());
    const py::ssize_t support = nricp::VertexBinding::kMaxNodes;
    py::array_t<std::int64_t> nodes({count, support});
    py::array_t<double> weights({count, support});
    py::array_t<std::int64_t> counts({count});
    auto node_view = nodes.mutable_unchecked<2>();
    auto weight_view = weights.mutable_unchecked<2>();
    auto count_view = counts.mutable_unchecked<1>();
    for (py::ssize_t i = 0; i < count; ++i) {
        const nricp::VertexBinding& binding =
            bindings[static_cast<std::size_t>(i)];
        count_view(i) = binding.count;
        for (py::ssize_t slot = 0; slot < support; ++slot) {
            node_view(i, slot) = binding.nodes[slot];
            weight_view(i, slot) = binding.weights[slot];
        }
    }
    py::dict result;
    result["nodes"] = nodes;
    result["weights"] = weights;
    result["counts"] = counts;
    return result;
}

std::vector<nricp::VertexBinding> to_bindings(
    const IndexArray& nodes, const DoubleArray& weights, const IndexArray& counts) {
    if (nodes.ndim() != 2 || nodes.shape(1) != nricp::VertexBinding::kMaxNodes) {
        throw py::value_error("binding nodes must have shape (n, 4)");
    }
    if (weights.ndim() != 2 || weights.shape(0) != nodes.shape(0) ||
        weights.shape(1) != nricp::VertexBinding::kMaxNodes) {
        throw py::value_error("binding weights must have shape (n, 4)");
    }
    if (counts.ndim() != 1 || counts.shape(0) != nodes.shape(0)) {
        throw py::value_error("binding counts must have shape (n,)");
    }
    const auto node_view = nodes.unchecked<2>();
    const auto weight_view = weights.unchecked<2>();
    const auto count_view = counts.unchecked<1>();
    std::vector<nricp::VertexBinding> bindings(
        static_cast<std::size_t>(nodes.shape(0)));
    for (py::ssize_t i = 0; i < nodes.shape(0); ++i) {
        nricp::VertexBinding& binding = bindings[static_cast<std::size_t>(i)];
        binding.count = static_cast<int>(count_view(i));
        if (binding.count < 1 || binding.count > nricp::VertexBinding::kMaxNodes) {
            throw py::value_error("binding count out of range at vertex " +
                                  std::to_string(i));
        }
        for (py::ssize_t slot = 0; slot < nricp::VertexBinding::kMaxNodes; ++slot) {
            binding.nodes[slot] = static_cast<int>(node_view(i, slot));
            binding.weights[slot] = weight_view(i, slot);
        }
    }
    return bindings;
}

nricp::Deformation to_deformation(
    const MatrixArray& rotations, const DoubleArray& translations,
    const DoubleArray& node_positions) {
    const std::vector<nricp::Mat3> matrices = to_matrices(rotations, "rotations");
    if (translations.ndim() != 2 || translations.shape(1) != 3 ||
        static_cast<std::size_t>(translations.shape(0)) != matrices.size()) {
        throw py::value_error("translations must have shape (n, 3)");
    }
    if (node_positions.ndim() != 2 || node_positions.shape(1) != 3 ||
        static_cast<std::size_t>(node_positions.shape(0)) != matrices.size()) {
        throw py::value_error("node_positions must have shape (n, 3)");
    }
    const auto view = translations.unchecked<2>();
    nricp::Deformation deformation;
    deformation.node_rest_positions = to_points(node_positions, "node_positions");
    deformation.nodes.resize(matrices.size());
    for (std::size_t i = 0; i < matrices.size(); ++i) {
        deformation.nodes[i].rotation = matrices[i];
        deformation.nodes[i].translation = {view(static_cast<py::ssize_t>(i), 0),
                                            view(static_cast<py::ssize_t>(i), 1),
                                            view(static_cast<py::ssize_t>(i), 2)};
    }
    return deformation;
}

py::dict deformation_to_dict(const nricp::Deformation& deformation) {
    std::vector<nricp::Mat3> rotations(deformation.nodes.size());
    py::array_t<double> translations(
        {static_cast<py::ssize_t>(deformation.nodes.size()),
         static_cast<py::ssize_t>(3)});
    auto view = translations.mutable_unchecked<2>();
    for (std::size_t i = 0; i < deformation.nodes.size(); ++i) {
        rotations[i] = deformation.nodes[i].rotation;
        view(static_cast<py::ssize_t>(i), 0) = deformation.nodes[i].translation.x;
        view(static_cast<py::ssize_t>(i), 1) = deformation.nodes[i].translation.y;
        view(static_cast<py::ssize_t>(i), 2) = deformation.nodes[i].translation.z;
    }
    py::dict result;
    result["rotations"] = from_matrices(rotations);
    result["translations"] = translations;
    return result;
}

// Rebuilds the block list from the arrays `energy_blocks` returns, so the
// linear-algebra stage can be exercised on exactly the state that was evaluated.
std::vector<nricp::ResidualBlock> to_blocks(
    const IndexArray& kind, const IndexArray& rows, const DoubleArray& values,
    const IndexArray& slots, const IndexArray& nodes,
    const DoubleArray& jacobians) {
    if (kind.ndim() != 1) {
        throw py::value_error("block kinds must have shape (n,)");
    }
    const py::ssize_t count = kind.shape(0);
    if (rows.shape(0) != count || values.shape(0) != count ||
        slots.shape(0) != count || nodes.shape(0) != count ||
        jacobians.shape(0) != count) {
        throw py::value_error("block arrays must agree in length");
    }
    if (values.shape(1) != nricp::kMaxBlockRows ||
        nodes.shape(1) != nricp::kMaxBlockSlots ||
        jacobians.shape(1) != nricp::kMaxBlockSlots ||
        jacobians.shape(2) != nricp::kMaxBlockRows || jacobians.shape(3) != 6) {
        throw py::value_error(
            "block arrays must have shapes (n, 6), (n, 4), (n, 4, 6, 6)");
    }
    const auto kind_view = kind.unchecked<1>();
    const auto row_view = rows.unchecked<1>();
    const auto value_view = values.unchecked<2>();
    const auto slot_view = slots.unchecked<1>();
    const auto node_view = nodes.unchecked<2>();
    const auto jacobian_view = jacobians.unchecked<4>();

    std::vector<nricp::ResidualBlock> blocks(static_cast<std::size_t>(count));
    for (py::ssize_t block = 0; block < count; ++block) {
        nricp::ResidualBlock& target = blocks[static_cast<std::size_t>(block)];
        target.kind = static_cast<nricp::BlockKind>(kind_view(block));
        target.row_count = static_cast<int>(row_view(block));
        target.slots = static_cast<int>(slot_view(block));
        for (int row = 0; row < nricp::kMaxBlockRows; ++row) {
            target.values[row] = value_view(block, row);
        }
        for (int slot = 0; slot < nricp::kMaxBlockSlots; ++slot) {
            target.nodes[slot] = static_cast<int>(node_view(block, slot));
            for (int row = 0; row < nricp::kMaxBlockRows; ++row) {
                for (int column = 0; column < 6; ++column) {
                    target.jacobians[slot][row][column] =
                        jacobian_view(block, slot, row, column);
                }
            }
        }
    }
    return blocks;
}

py::dict from_normal_equations(const nricp::NormalEquations& system) {
    const py::ssize_t nodes = static_cast<py::ssize_t>(system.node_count);
    py::array_t<double> diagonal({nodes, static_cast<py::ssize_t>(6),
                                 static_cast<py::ssize_t>(6)});
    py::array_t<double> rhs({nodes, static_cast<py::ssize_t>(6)});
    py::array_t<std::int64_t> edges(
        {static_cast<py::ssize_t>(system.off_diagonal.size()),
         static_cast<py::ssize_t>(2)});
    py::array_t<double> edge_blocks(
        {static_cast<py::ssize_t>(system.off_diagonal.size()),
         static_cast<py::ssize_t>(6), static_cast<py::ssize_t>(6)});
    auto diagonal_view = diagonal.mutable_unchecked<3>();
    auto rhs_view = rhs.mutable_unchecked<2>();
    auto edge_view = edges.mutable_unchecked<2>();
    auto block_view = edge_blocks.mutable_unchecked<3>();
    for (py::ssize_t node = 0; node < nodes; ++node) {
        for (int row = 0; row < 6; ++row) {
            for (int column = 0; column < 6; ++column) {
                diagonal_view(node, row, column) =
                    system.diagonal[static_cast<std::size_t>(node)].at(row, column);
            }
            rhs_view(node, row) =
                system.rhs[static_cast<std::size_t>(node) * 6 + row];
        }
    }
    for (std::size_t i = 0; i < system.off_diagonal.size(); ++i) {
        const nricp::NormalEquations::OffDiagonal& block = system.off_diagonal[i];
        edge_view(static_cast<py::ssize_t>(i), 0) = block.a;
        edge_view(static_cast<py::ssize_t>(i), 1) = block.b;
        for (int row = 0; row < 6; ++row) {
            for (int column = 0; column < 6; ++column) {
                block_view(static_cast<py::ssize_t>(i), row, column) =
                    block.value.at(row, column);
            }
        }
    }
    py::dict result;
    result["diagonal"] = diagonal;
    result["edges"] = edges;
    result["edge_blocks"] = edge_blocks;
    result["rhs"] = rhs;
    return result;
}

nricp::NormalEquations to_normal_equations(
    const DoubleArray& diagonal, const IndexArray& edges,
    const DoubleArray& edge_blocks, const DoubleArray& rhs) {
    if (diagonal.ndim() != 3 || diagonal.shape(1) != 6 || diagonal.shape(2) != 6) {
        throw py::value_error("diagonal must have shape (n, 6, 6)");
    }
    const py::ssize_t nodes = diagonal.shape(0);
    if (rhs.ndim() != 2 || rhs.shape(0) != nodes || rhs.shape(1) != 6) {
        throw py::value_error("rhs must have shape (n, 6)");
    }
    if (edges.ndim() != 2 || edges.shape(1) != 2 ||
        edge_blocks.ndim() != 3 || edge_blocks.shape(0) != edges.shape(0) ||
        edge_blocks.shape(1) != 6 || edge_blocks.shape(2) != 6) {
        throw py::value_error(
            "edges must be (e, 2) and edge_blocks (e, 6, 6)");
    }
    nricp::NormalEquations system;
    system.node_count = static_cast<int>(nodes);
    system.diagonal.resize(static_cast<std::size_t>(nodes));
    system.rhs.resize(static_cast<std::size_t>(nodes) * 6);
    const auto diagonal_view = diagonal.unchecked<3>();
    const auto rhs_view = rhs.unchecked<2>();
    for (py::ssize_t node = 0; node < nodes; ++node) {
        for (int row = 0; row < 6; ++row) {
            for (int column = 0; column < 6; ++column) {
                system.diagonal[static_cast<std::size_t>(node)].at(row, column) =
                    diagonal_view(node, row, column);
            }
            system.rhs[static_cast<std::size_t>(node) * 6 + row] =
                rhs_view(node, row);
        }
    }
    const auto edge_view = edges.unchecked<2>();
    const auto block_view = edge_blocks.unchecked<3>();
    system.off_diagonal.resize(static_cast<std::size_t>(edges.shape(0)));
    for (py::ssize_t i = 0; i < edges.shape(0); ++i) {
        nricp::NormalEquations::OffDiagonal& block =
            system.off_diagonal[static_cast<std::size_t>(i)];
        block.a = static_cast<int>(edge_view(i, 0));
        block.b = static_cast<int>(edge_view(i, 1));
        for (int row = 0; row < 6; ++row) {
            for (int column = 0; column < 6; ++column) {
                block.value.at(row, column) = block_view(i, row, column);
            }
        }
    }
    return system;
}

void raise_if_invalid(const nricp::ValidationIssues& issues, const char* what) {
    if (issues.ok) {
        return;
    }
    std::string message(what);
    message += ": ";
    for (std::size_t i = 0; i < issues.problems.size(); ++i) {
        if (i > 0) {
            message += "; ";
        }
        message += issues.problems[i];
    }
    throw py::value_error(message);
}

// Everything a route needs before it starts solving: both meshes in the graph's
// normalized frame, the cached template side, and the parsed settings. Shared by
// the route bindings so their inputs are prepared identically.
struct PreparedSolve {
    nricp::Mesh normalized_template;
    nricp::Mesh normalized_target;
    nricp::TemplateGraph graph;
    nricp::RouteSettings settings;
    nricp::VertexMask mask;
    std::vector<nricp::Landmark> landmarks;
};

PreparedSolve prepare_solve(
    const DoubleArray& template_points, const IndexArray& template_faces,
    const DoubleArray& target_points, const IndexArray& target_faces,
    const py::list& level_settings, const py::object& mask,
    const py::object& landmark_indices, const py::object& landmark_targets,
    double distance_threshold, double min_normal_cosine, double min_slope,
    double initial_damping, double damping_decrease, double damping_increase,
    int max_damping_attempts, bool diagonal_damping, int graph_levels,
    int nodes_initial, int growth, double pcg_tolerance, int pcg_max_iterations,
    bool use_device) {
    PreparedSolve prepared;
    nricp::Mesh template_mesh;
    template_mesh.vertices = to_points(template_points, "template_points");
    template_mesh.faces = to_faces(template_faces, "template_faces");
    raise_if_invalid(nricp::validate(template_mesh), "invalid template mesh");
    nricp::Mesh target_mesh;
    target_mesh.vertices = to_points(target_points, "target_points");
    target_mesh.faces = to_faces(target_faces, "target_faces");
    raise_if_invalid(nricp::validate(target_mesh), "invalid target mesh");

    nricp::GraphSettings graph_settings;
    graph_settings.levels = graph_levels;
    graph_settings.nodes_initial = nodes_initial;
    graph_settings.node_growth = growth;
    // The graph is template-side and independent of the target, so the
    // accelerated route builds it on the device -- the construction is two
    // sweeps of every vertex against every node of a level, and at a hundred
    // thousand vertices that, not the solve, is where a single wrap's time goes.
    // The core layer's builder stays as the reference the device one is checked
    // against, and as the path taken when there is no device.
    prepared.graph = use_device && nricp::device_graph_available()
                         ? nricp::build_graph_on_device(template_mesh, graph_settings)
                         : nricp::build_template_graph(template_mesh, graph_settings);

    // Both meshes (and the landmark targets) live in the graph's normalized
    // frame; the caller is responsible for the rigid alignment, which is what
    // puts them in a common frame at all.
    prepared.normalized_template = template_mesh;
    nricp::apply_normalization(prepared.normalized_template, prepared.graph.normalization);
    prepared.normalized_target = target_mesh;
    nricp::apply_normalization(prepared.normalized_target, prepared.graph.normalization);

    for (const auto& item : level_settings) {
        prepared.settings.levels.push_back(to_level_settings(item.cast<py::dict>()));
    }
    // Distances arrive in the caller's units while the solver works in the
    // graph's normalized frame, so they are converted once, here.
    const double units = prepared.graph.normalization.scale;
    prepared.settings.correspondence.distance_threshold = distance_threshold * units;
    prepared.settings.correspondence.min_normal_cosine = min_normal_cosine;
    prepared.settings.correspondence.min_slope = min_slope;
    for (nricp::LevelSettings& level : prepared.settings.levels) {
        if (level.robust_threshold > 0.0) {
            level.robust_threshold *= units;
        }
    }
    prepared.settings.initial_damping = initial_damping;
    prepared.settings.damping_decrease = damping_decrease;
    prepared.settings.damping_increase = damping_increase;
    prepared.settings.max_damping_attempts = max_damping_attempts;
    prepared.settings.diagonal_damping = diagonal_damping;
    prepared.settings.pcg_tolerance = static_cast<float>(pcg_tolerance);
    prepared.settings.pcg_max_iterations = pcg_max_iterations;

    prepared.mask = to_mask(mask);
    if (!landmark_indices.is_none() && !landmark_targets.is_none()) {
        const IndexArray indices = landmark_indices.cast<IndexArray>();
        const std::vector<nricp::Vec3> targets =
            to_points(landmark_targets.cast<DoubleArray>(), "landmark_targets");
        const auto index_view = indices.unchecked<1>();
        if (static_cast<std::size_t>(indices.shape(0)) != targets.size()) {
            throw py::value_error(
                "landmark indices and targets must agree in length");
        }
        prepared.landmarks.resize(targets.size());
        for (std::size_t i = 0; i < targets.size(); ++i) {
            prepared.landmarks[i] = {
                static_cast<int>(index_view(static_cast<py::ssize_t>(i))),
                (targets[i] - prepared.graph.normalization.center) *
                    prepared.graph.normalization.scale};
        }
        raise_if_invalid(
            nricp::validate_landmarks(
                prepared.landmarks,
                static_cast<int>(prepared.normalized_template.vertices.size())),
            "invalid landmarks");
    }
    return prepared;
}

py::dict route_result_to_dict(
    const nricp::RouteResult& solved, const nricp::TemplateGraph& graph) {
    // Report the fitted vertices back in the caller's scale.
    nricp::Mesh restored;
    restored.vertices = solved.vertices;
    nricp::undo_normalization(restored, graph.normalization);

    py::dict result;
    result["vertices"] = from_points(restored.vertices);
    std::vector<nricp::Mat3> rotations(solved.deformation.nodes.size());
    for (std::size_t i = 0; i < rotations.size(); ++i) {
        rotations[i] = solved.deformation.nodes[i].rotation;
    }
    result["rotations"] = from_matrices(rotations);
    py::array_t<double> translations(
        {static_cast<py::ssize_t>(solved.deformation.nodes.size()),
         static_cast<py::ssize_t>(3)});
    auto translation_view = translations.mutable_unchecked<2>();
    for (std::size_t i = 0; i < solved.deformation.nodes.size(); ++i) {
        translation_view(static_cast<py::ssize_t>(i), 0) =
            solved.deformation.nodes[i].translation.x;
        translation_view(static_cast<py::ssize_t>(i), 1) =
            solved.deformation.nodes[i].translation.y;
        translation_view(static_cast<py::ssize_t>(i), 2) =
            solved.deformation.nodes[i].translation.z;
    }
    result["translations"] = translations;
    result["node_positions"] = from_points(solved.deformation.node_rest_positions);
    result["report"] = nricp::to_json(solved.report);
    std::vector<int> level_counts;
    level_counts.reserve(graph.levels.size());
    for (const nricp::GraphLevel& level : graph.levels) {
        level_counts.push_back(level.node_count());
    }
    result["level_nodes"] = from_indices(level_counts);
    return result;
}

}  // namespace

PYBIND11_MODULE(nricp_dp, m) {
    m.doc() =
        "Non-rigid deformation-graph wrapping solver (NRICP): fits a "
        "fixed-topology template onto an arbitrary-topology target.";

    m.attr("__version__") = NRICP_VERSION_STRING;

    // Artifact identity: reported by every wrap call and stored with the
    // benchmark results, so a stored number can be traced to a build.
    m.def(
        "build_info",
        [] {
            py::dict info;
            info["version"] = nricp::version();
            info["build_type"] = nricp::build_type();
            info["cxx_compiler"] = nricp::cxx_compiler();
            info["cuda_toolkit"] = nricp::cuda_toolkit();
            info["cuda_architectures"] = nricp::cuda_architectures();
            info["accelerated_precision"] = nricp::accelerated_precision();
            info["python_version"] = std::string(PY_VERSION);
            info["pybind11_version"] = pybind11_version();
            return info;
        },
        "Report the version and build target of this artifact.");

    m.def(
        "bounding_box_diagonal",
        [](const DoubleArray& points) {
            nricp::Mesh mesh;
            mesh.vertices = to_points(points, "points");
            return nricp::bounding_box_diagonal(mesh);
        },
        py::arg("points"));

    m.def(
        "validate_mesh",
        [](const DoubleArray& points, const IndexArray& faces) {
            nricp::Mesh mesh;
            mesh.vertices = to_points(points, "points");
            mesh.faces = to_faces(faces, "faces");
            const nricp::ValidationIssues issues = nricp::validate(mesh);
            py::dict result;
            result["ok"] = issues.ok;
            result["problems"] = issues.problems;
            return result;
        },
        py::arg("points"), py::arg("faces"));

    m.def(
        "clean_mesh",
        [](const DoubleArray& points, const IndexArray& faces) {
            nricp::Mesh mesh;
            mesh.vertices = to_points(points, "points");
            mesh.faces = to_faces(faces, "faces");
            raise_if_invalid(nricp::validate(mesh), "invalid mesh");
            const nricp::CleanReport report = nricp::clean(mesh);
            py::dict result;
            result["points"] = from_points(mesh.vertices);
            result["faces"] = from_faces(mesh.faces);
            result["duplicate_faces"] = report.duplicate_faces;
            result["degenerate_faces"] = report.degenerate_faces;
            result["isolated_vertices"] = report.isolated_vertices;
            return result;
        },
        py::arg("points"), py::arg("faces"),
        "Drop duplicate faces, degenerate faces, and isolated vertices.");

    m.def(
        "normalization",
        [](const DoubleArray& points) {
            nricp::Mesh mesh;
            mesh.vertices = to_points(points, "points");
            const nricp::Normalization normalization =
                nricp::compute_normalization(mesh);
            py::dict result;
            result["center"] = from_points({normalization.center});
            result["scale"] = normalization.scale;
            return result;
        },
        py::arg("points"),
        "Centering and scaling that puts the bounding-box diagonal at 1.");

    m.def(
        "farthest_point_seeds",
        [](const DoubleArray& points, int count, int first_index) {
            const std::vector<nricp::Vec3> converted = to_points(points, "points");
            return from_indices(
                nricp::farthest_point_seeds(converted, count, first_index));
        },
        py::arg("points"), py::arg("count"), py::arg("first_index") = 0,
        "Deterministic farthest-point sampling; ties break to the lower index.");

    m.def(
        "bind_vertices",
        [](const DoubleArray& points, const DoubleArray& nodes, int max_nodes) {
            const std::vector<nricp::Vec3> vertices = to_points(points, "points");
            const std::vector<nricp::Vec3> node_points = to_points(nodes, "nodes");
            return from_bindings(
                nricp::bind_vertices(vertices, node_points, max_nodes));
        },
        py::arg("points"), py::arg("nodes"), py::arg("max_nodes") = 4,
        "Bind every vertex to its nearest nodes with distance-ratio weights.");

    m.def(
        "evaluate_deformation",
        [](const DoubleArray& points, const IndexArray& binding_nodes,
           const DoubleArray& binding_weights, const IndexArray& binding_counts,
           const MatrixArray& rotations, const DoubleArray& translations,
           const DoubleArray& node_positions) {
            const nricp::Deformation deformation =
                to_deformation(rotations, translations, node_positions);
            const std::vector<nricp::VertexBinding> bindings =
                to_bindings(binding_nodes, binding_weights, binding_counts);
            const std::vector<nricp::Vec3> vertices = to_points(points, "points");
            if (vertices.size() != bindings.size()) {
                throw py::value_error(
                    "points and bindings must have the same length (" +
                    std::to_string(vertices.size()) + " vs " +
                    std::to_string(bindings.size()) + ")");
            }
            return from_points(nricp::evaluate(vertices, bindings, deformation));
        },
        py::arg("points"), py::arg("binding_nodes"), py::arg("binding_weights"),
        py::arg("binding_counts"), py::arg("rotations"), py::arg("translations"),
        py::arg("node_positions"));

    m.def(
        "transfer_level",
        [](const DoubleArray& from_nodes, const MatrixArray& rotations,
           const DoubleArray& translations, const DoubleArray& to_nodes,
           int max_nodes) {
            const std::vector<nricp::Vec3> from_node_points =
                to_points(from_nodes, "from_nodes");
            const nricp::Deformation from_deformation =
                to_deformation(rotations, translations, from_nodes);
            return deformation_to_dict(nricp::transfer_level(
                from_node_points, from_deformation,
                to_points(to_nodes, "to_nodes"), max_nodes));
        },
        py::arg("from_nodes"), py::arg("rotations"), py::arg("translations"),
        py::arg("to_nodes"), py::arg("max_nodes") = 4,
        "Sample a deformation at new node positions by rigid fitting.");

    m.def(
        "fit_rigid",
        [](const DoubleArray& source_points, const DoubleArray& target_points,
           const DoubleArray& weights) {
            const std::vector<nricp::Vec3> sources =
                to_points(source_points, "source_points");
            const std::vector<nricp::Vec3> targets =
                to_points(target_points, "target_points");
            if (weights.ndim() != 1 ||
                static_cast<std::size_t>(weights.shape(0)) != sources.size() ||
                sources.size() != targets.size()) {
                throw py::value_error(
                    "fit_rigid expects (n, 3) point pairs and (n,) weights");
            }
            const auto weight_view = weights.unchecked<1>();
            std::vector<double> weight_values(sources.size());
            for (std::size_t i = 0; i < sources.size(); ++i) {
                weight_values[i] = weight_view(static_cast<py::ssize_t>(i));
            }
            const nricp::RigidTransform fit =
                nricp::fit_rigid(sources, targets, weight_values);
            py::dict result;
            result["rotation"] = from_matrices({fit.rotation});
            result["translation"] = from_points({fit.translation});
            return result;
        },
        py::arg("source_points"), py::arg("target_points"), py::arg("weights"),
        "Weighted rigid fit (reflection excluded) between point pairs.");

    m.def(
        "build_template_graph",
        [](const DoubleArray& points, const IndexArray& faces, int levels,
           int nodes_initial, int growth, int max_nodes_per_vertex,
           bool use_device) {
            nricp::Mesh mesh;
            mesh.vertices = to_points(points, "points");
            mesh.faces = to_faces(faces, "faces");
            raise_if_invalid(nricp::validate(mesh), "invalid mesh");
            nricp::GraphSettings settings;
            settings.levels = levels;
            settings.nodes_initial = nodes_initial;
            settings.node_growth = growth;
            settings.max_nodes_per_vertex = max_nodes_per_vertex;
            const nricp::TemplateGraph graph =
                use_device && nricp::device_graph_available()
                    ? nricp::build_graph_on_device(mesh, settings)
                    : nricp::build_template_graph(mesh, settings);
            py::dict result;
            py::dict normalization;
            normalization["center"] = from_points({graph.normalization.center});
            normalization["scale"] = graph.normalization.scale;
            result["normalization"] = normalization;
            py::list level_list;
            for (const nricp::GraphLevel& level : graph.levels) {
                py::dict entry;
                entry["node_positions"] = from_points(level.node_rest_positions);
                entry["node_template_indices"] =
                    from_indices(level.node_template_indices);
                entry["bindings"] = from_bindings(level.bindings);
                entry["edges"] = from_edges(level.edges);
                level_list.append(entry);
            }
            result["levels"] = level_list;
            return result;
        },
        py::arg("points"), py::arg("faces"), py::arg("levels") = 4,
        py::arg("nodes_initial") = 100, py::arg("growth") = 2,
        py::arg("max_nodes_per_vertex") = 4, py::arg("use_device") = false,
        "Prepare the cached template side: nodes and bindings per level.");

    m.def(
        "validate_landmarks",
        [](const IndexArray& indices, const DoubleArray& targets,
           int template_vertex_count) {
            if (indices.ndim() != 1) {
                throw py::value_error("landmark indices must have shape (n,)");
            }
            if (targets.ndim() != 2 || targets.shape(1) != 3 ||
                targets.shape(0) != indices.shape(0)) {
                throw py::value_error(
                    "landmark targets must have shape (n, 3), matching indices");
            }
            const auto index_view = indices.unchecked<1>();
            const auto target_view = targets.unchecked<2>();
            std::vector<nricp::Landmark> landmarks(
                static_cast<std::size_t>(indices.shape(0)));
            for (py::ssize_t i = 0; i < indices.shape(0); ++i) {
                landmarks[static_cast<std::size_t>(i)] = {
                    static_cast<int>(index_view(i)),
                    {target_view(i, 0), target_view(i, 1), target_view(i, 2)}};
            }
            raise_if_invalid(
                nricp::validate_landmarks(landmarks, template_vertex_count),
                "invalid landmarks");
        },
        py::arg("indices"), py::arg("targets"), py::arg("template_vertex_count"),
        "Raise with a descriptive message when landmarks are unusable.");

    m.def(
        "validate_mask",
        [](const IndexArray& mask, int template_vertex_count) {
            nricp::VertexMask vertex_mask;
            if (mask.size() > 0) {
                vertex_mask.active.resize(static_cast<std::size_t>(mask.size()));
                for (py::ssize_t i = 0; i < mask.size(); ++i) {
                    vertex_mask.active[static_cast<std::size_t>(i)] =
                        static_cast<unsigned char>(
                            mask.at(static_cast<py::ssize_t>(i)) ? 1 : 0);
                }
            }
            raise_if_invalid(
                nricp::validate_mask(vertex_mask, template_vertex_count),
                "invalid mask");
            return nricp::active_vertex_count(vertex_mask);
        },
        py::arg("mask"), py::arg("template_vertex_count"),
        "Validate a participation mask and return how many vertices stay active.");

    m.def(
        "closest_points",
        [](const DoubleArray& target_points, const IndexArray& target_faces,
           const DoubleArray& query_points) {
            nricp::Mesh target;
            target.vertices = to_points(target_points, "target_points");
            target.faces = to_faces(target_faces, "target_faces");
            raise_if_invalid(nricp::validate(target), "invalid target mesh");
            const nricp::TargetQueries queries(target);
            const std::vector<nricp::Vec3> queries_in =
                to_points(query_points, "query_points");
            std::vector<nricp::Vec3> points_out;
            std::vector<nricp::Vec3> normals_out;
            std::vector<double> distances;
            std::vector<int> triangles;
            points_out.reserve(queries_in.size());
            normals_out.reserve(queries_in.size());
            distances.reserve(queries_in.size());
            triangles.reserve(queries_in.size());
            for (const nricp::Vec3& point : queries_in) {
                const nricp::ClosestPointResult result = queries.query(point);
                points_out.push_back(result.point);
                normals_out.push_back(result.normal);
                distances.push_back(result.distance);
                triangles.push_back(result.triangle);
            }
            py::dict out;
            out["points"] = from_points(points_out);
            out["normals"] = from_points(normals_out);
            out["distances"] =
                py::array_t<double>(static_cast<py::ssize_t>(distances.size()),
                                    distances.data());
            out["triangles"] = from_indices(triangles);
            out["vertex_normals"] = from_points(queries.vertex_normals());
            return out;
        },
        py::arg("target_points"), py::arg("target_faces"), py::arg("query_points"),
        "Closest point and interpolated normal on the target surface.");

    m.def(
        "vertex_normals",
        [](const DoubleArray& points, const IndexArray& faces) {
            nricp::Mesh mesh;
            mesh.vertices = to_points(points, "points");
            mesh.faces = to_faces(faces, "faces");
            raise_if_invalid(nricp::validate(mesh), "invalid mesh");
            return from_points(nricp::vertex_normals(mesh));
        },
        py::arg("points"), py::arg("faces"),
        "Area-weighted vertex normals of a consistently wound mesh.");

    m.def(
        "retract_deformation",
        [](const MatrixArray& rotations, const DoubleArray& translations,
           const DoubleArray& node_positions, const DoubleArray& delta) {
            nricp::Deformation deformation =
                to_deformation(rotations, translations, node_positions);
            if (delta.ndim() != 1 ||
                static_cast<std::size_t>(delta.shape(0)) !=
                    deformation.nodes.size() * 6) {
                throw py::value_error(
                    "delta must have shape (6 * node_count,): rotation increment "
                    "then translation increment per node");
            }
            const auto view = delta.unchecked<1>();
            std::vector<double> values(static_cast<std::size_t>(delta.shape(0)));
            for (py::ssize_t i = 0; i < delta.shape(0); ++i) {
                values[static_cast<std::size_t>(i)] = view(i);
            }
            nricp::retract(deformation, values);
            return deformation_to_dict(deformation);
        },
        py::arg("rotations"), py::arg("translations"), py::arg("node_positions"),
        py::arg("delta"),
        "Apply a tangent-space increment: R <- exp([w]) R, t <- t + dt.");

    m.def(
        "tangent_difference",
        [](const MatrixArray& rotations_from, const DoubleArray& translations_from,
           const MatrixArray& rotations_to, const DoubleArray& translations_to) {
            const nricp::Deformation from = to_deformation(
                rotations_from, translations_from,
                from_points(std::vector<nricp::Vec3>(
                    static_cast<std::size_t>(rotations_from.shape(0)))));
            const nricp::Deformation to = to_deformation(
                rotations_to, translations_to,
                from_points(std::vector<nricp::Vec3>(
                    static_cast<std::size_t>(rotations_to.shape(0)))));
            const std::vector<double> difference =
                nricp::tangent_difference(from, to);
            py::array_t<double> array(
                {static_cast<py::ssize_t>(difference.size())});
            auto view = array.mutable_unchecked<1>();
            for (std::size_t i = 0; i < difference.size(); ++i) {
                view(static_cast<py::ssize_t>(i)) = difference[i];
            }
            return array;
        },
        py::arg("rotations_from"), py::arg("translations_from"),
        py::arg("rotations_to"), py::arg("translations_to"),
        "Tangent-space increment taking one deformation to another.");

    m.def(
        "robust_weights",
        [](const std::string& kernel, const DoubleArray& squared_residuals,
           double threshold) {
            const nricp::RobustKernel selected =
                nricp::robust_kernel_from_string(kernel);
            const auto view = squared_residuals.unchecked<1>();
            py::array_t<double> weights(squared_residuals.shape(0));
            auto output = weights.mutable_unchecked<1>();
            for (py::ssize_t i = 0; i < squared_residuals.shape(0); ++i) {
                output(i) = nricp::robust_weight(selected, view(i), threshold);
            }
            return weights;
        },
        py::arg("kernel"), py::arg("squared_residuals"), py::arg("threshold"),
        "IRLS weights for a robust kernel ('none', 'huber', 'geman_mcclure').");

    m.def(
        "energy_blocks",
        [](const DoubleArray& template_points, const IndexArray& binding_nodes,
           const DoubleArray& binding_weights, const IndexArray& binding_counts,
           const MatrixArray& rotations, const DoubleArray& translations,
           const DoubleArray& node_positions,
           const DoubleArray& correspondence_points,
           const DoubleArray& correspondence_normals,
           const IndexArray& correspondence_accepted,
           const DoubleArray& correspondence_robust, const IndexArray& edges,
           const IndexArray& landmark_indices, const DoubleArray& landmark_targets,
           const py::dict& weights, const py::object& mask) {
            EnergyCall call = to_energy_call(
                template_points, binding_nodes, binding_weights, binding_counts,
                node_positions, correspondence_points, correspondence_normals,
                correspondence_accepted, correspondence_robust, edges,
                landmark_indices, landmark_targets, weights, mask);
            call.inputs.mask = call.mask.active.empty() ? nullptr : &call.mask;
            const nricp::EnergyWeights& energy_weights = call.weights;
            const nricp::Deformation deformation =
                to_deformation(rotations, translations, node_positions);
            const std::vector<nricp::ResidualBlock> blocks =
                nricp::build_blocks(call.inputs, deformation, energy_weights);

            const py::ssize_t count = static_cast<py::ssize_t>(blocks.size());
            const py::ssize_t row_capacity =
                static_cast<py::ssize_t>(nricp::kMaxBlockRows);
            const py::ssize_t slot_capacity =
                static_cast<py::ssize_t>(nricp::kMaxBlockSlots);
            py::array_t<std::int64_t> kinds({count});
            py::array_t<std::int64_t> rows({count});
            py::array_t<double> values({count, row_capacity});
            py::array_t<std::int64_t> slots({count});
            py::array_t<std::int64_t> nodes({count, slot_capacity});
            py::array_t<double> jacobians(
                {count, slot_capacity, row_capacity, static_cast<py::ssize_t>(6)});
            auto kind_view = kinds.mutable_unchecked<1>();
            auto row_view = rows.mutable_unchecked<1>();
            auto value_view = values.mutable_unchecked<2>();
            auto slot_view = slots.mutable_unchecked<1>();
            auto node_view = nodes.mutable_unchecked<2>();
            auto jacobian_view = jacobians.mutable_unchecked<4>();
            for (py::ssize_t block = 0; block < count; ++block) {
                const nricp::ResidualBlock& source =
                    blocks[static_cast<std::size_t>(block)];
                kind_view(block) = static_cast<std::int64_t>(source.kind);
                row_view(block) = source.row_count;
                slot_view(block) = source.slots;
                for (int row = 0; row < nricp::kMaxBlockRows; ++row) {
                    value_view(block, row) = source.values[row];
                }
                for (int slot = 0; slot < nricp::kMaxBlockSlots; ++slot) {
                    node_view(block, slot) = source.nodes[slot];
                    for (int row = 0; row < nricp::kMaxBlockRows; ++row) {
                        for (int column = 0; column < 6; ++column) {
                            jacobian_view(block, slot, row, column) =
                                source.jacobians[slot][row][column];
                        }
                    }
                }
            }

            const nricp::BlockEnergies energies = nricp::block_energies(blocks);
            py::dict energy_dict;
            energy_dict["point_to_point"] = energies.point_to_point;
            energy_dict["point_to_plane"] = energies.point_to_plane;
            energy_dict["smoothness"] = energies.smoothness;
            energy_dict["anchor"] = energies.anchor;
            energy_dict["landmark"] = energies.landmark;
            energy_dict["total"] = energies.total();

            py::dict block_dict;
            block_dict["kind"] = kinds;
            block_dict["rows"] = rows;
            block_dict["values"] = values;
            block_dict["slots"] = slots;
            block_dict["nodes"] = nodes;
            block_dict["jacobians"] = jacobians;

            py::dict result;
            result["energies"] = energy_dict;
            result["blocks"] = block_dict;
            return result;
        },
        py::arg("template_points"), py::arg("binding_nodes"),
        py::arg("binding_weights"), py::arg("binding_counts"),
        py::arg("rotations"), py::arg("translations"), py::arg("node_positions"),
        py::arg("correspondence_points"), py::arg("correspondence_normals"),
        py::arg("correspondence_accepted"), py::arg("correspondence_robust"),
        py::arg("edges"), py::arg("landmark_indices"), py::arg("landmark_targets"),
        py::arg("weights"), py::arg("mask") = py::none(),
        "Residual blocks, their analytic Jacobians, and the block energies.");

    m.def(
        "find_correspondences",
        [](const DoubleArray& target_points, const IndexArray& target_faces,
           const DoubleArray& query_points, const DoubleArray& query_normals,
           double distance_threshold, double min_normal_cosine, double min_slope,
           const py::object& mask) {
            nricp::Mesh target;
            target.vertices = to_points(target_points, "target_points");
            target.faces = to_faces(target_faces, "target_faces");
            raise_if_invalid(nricp::validate(target), "invalid target mesh");
            nricp::VertexMask vertex_mask = to_mask(mask);
            nricp::CorrespondenceSettings settings;
            settings.distance_threshold = distance_threshold;
            settings.min_normal_cosine = min_normal_cosine;
            settings.min_slope = min_slope;
            const nricp::TargetQueries queries(target);
            const nricp::CorrespondenceSet set = nricp::find_correspondences(
                queries, to_points(query_points, "query_points"),
                to_points(query_normals, "query_normals"), settings,
                vertex_mask.active.empty() ? nullptr : &vertex_mask);

            std::vector<nricp::Vec3> points;
            std::vector<nricp::Vec3> normals;
            std::vector<int> triangles;
            std::vector<int> rejection;
            std::vector<int> accepted;
            std::vector<double> distances;
            points.reserve(set.items.size());
            for (const nricp::Correspondence& item : set.items) {
                points.push_back(item.point);
                normals.push_back(item.normal);
                distances.push_back(item.distance);
                triangles.push_back(item.triangle);
                rejection.push_back(static_cast<int>(item.rejection));
                accepted.push_back(item.accepted() ? 1 : 0);
            }
            py::dict counts;
            counts["total"] = set.counts.total;
            counts["accepted"] = set.counts.accepted;
            counts["rejected_distance"] = set.counts.rejected_distance;
            counts["rejected_normal"] = set.counts.rejected_normal;
            counts["rejected_slope"] = set.counts.rejected_slope;
            counts["rejected_masked"] = set.counts.rejected_masked;
            counts["distance_mean"] = set.counts.distance_mean;
            counts["distance_p95"] = set.counts.distance_p95;
            counts["distance_max"] = set.counts.distance_max;
            py::dict result;
            result["points"] = from_points(points);
            result["normals"] = from_points(normals);
            result["distances"] = py::array_t<double>(
                static_cast<py::ssize_t>(distances.size()), distances.data());
            result["triangles"] = from_indices(triangles);
            result["rejection"] = from_indices(rejection);
            result["accepted"] = from_indices(accepted);
            result["counts"] = counts;
            return result;
        },
        py::arg("target_points"), py::arg("target_faces"), py::arg("query_points"),
        py::arg("query_normals"),
        py::arg("distance_threshold") = std::numeric_limits<double>::infinity(),
        py::arg("min_normal_cosine") = -1.0,
        py::arg("min_slope") = 0.0,
        py::arg("mask") = py::none(),
        "Closest-point correspondences with the distance, normal, slope and mask "
        "filters applied, plus the rejection breakdown.");

    m.def(
        "serialize_report",
        [](const std::string& route, const std::string& precision,
           double setup_seconds, double solve_seconds, const py::dict& energy,
           const py::dict& correspondence, const py::list& levels,
           const py::dict& settings) {
            nricp::RunReport report;
            report.route = route;
            report.precision = precision;
            report.version = nricp::version();
            report.setup_seconds = setup_seconds;
            report.solve_seconds = solve_seconds;
            report.energy.point_to_point =
                py::cast<double>(energy["point_to_point"]);
            report.energy.point_to_plane =
                py::cast<double>(energy["point_to_plane"]);
            report.energy.smoothness = py::cast<double>(energy["smoothness"]);
            report.energy.anchor = py::cast<double>(energy["anchor"]);
            report.energy.landmark = py::cast<double>(energy["landmark"]);
            report.correspondence.total = py::cast<int>(correspondence["total"]);
            report.correspondence.accepted = py::cast<int>(correspondence["accepted"]);
            report.correspondence.rejected_distance =
                py::cast<int>(correspondence["rejected_distance"]);
            report.correspondence.rejected_normal =
                py::cast<int>(correspondence["rejected_normal"]);
            report.correspondence.rejected_slope =
                py::cast<int>(correspondence["rejected_slope"]);
            report.correspondence.rejected_masked =
                py::cast<int>(correspondence["rejected_masked"]);
            report.correspondence.distance_mean =
                py::cast<double>(correspondence["distance_mean"]);
            report.correspondence.distance_p95 =
                py::cast<double>(correspondence["distance_p95"]);
            report.correspondence.distance_max =
                py::cast<double>(correspondence["distance_max"]);
            for (const auto& item : levels) {
                const py::dict entry = item.cast<py::dict>();
                nricp::LevelView level;
                level.level = py::cast<int>(entry["level"]);
                level.nodes = py::cast<int>(entry["nodes"]);
                level.outer_iterations = py::cast<int>(entry["outer_iterations"]);
                level.inner_iterations = py::cast<int>(entry["inner_iterations"]);
                level.energy_start = py::cast<double>(entry["energy_start"]);
                level.energy_end = py::cast<double>(entry["energy_end"]);
                level.correspondence_mean =
                    py::cast<double>(entry["correspondence_mean"]);
                level.converged = py::cast<bool>(entry["converged"]);
                report.levels.push_back(level);
            }
            for (const auto& item : settings) {
                report.settings.emplace_back(
                    py::cast<std::string>(item.first),
                    py::cast<double>(item.second));
            }
            return nricp::to_json(report);
        },
        py::arg("route"), py::arg("precision"), py::arg("setup_seconds"),
        py::arg("solve_seconds"), py::arg("energy"), py::arg("correspondence"),
        py::arg("levels"), py::arg("settings"),
        "Serialize a run report as JSON, carrying the artifact version.");

    m.def(
        "assemble_normal_equations",
        [](const IndexArray& kind, const IndexArray& rows, const DoubleArray& values,
           const IndexArray& slots, const IndexArray& nodes,
           const DoubleArray& jacobians, int node_count) {
            const std::vector<nricp::ResidualBlock> blocks = to_blocks(
                kind, rows, values, slots, nodes, jacobians);
            return from_normal_equations(
                nricp::assemble_normal_equations(blocks, node_count));
        },
        py::arg("kind"), py::arg("rows"), py::arg("values"), py::arg("slots"),
        py::arg("nodes"), py::arg("jacobians"), py::arg("node_count"),
        "Assemble the reduced system (J^T J, -J^T r) from residual blocks.");

    m.def(
        "fused_system",
        [](const DoubleArray& template_points, const IndexArray& binding_nodes,
           const DoubleArray& binding_weights, const IndexArray& binding_counts,
           const MatrixArray& rotations, const DoubleArray& translations,
           const DoubleArray& node_positions,
           const DoubleArray& correspondence_points,
           const DoubleArray& correspondence_normals,
           const IndexArray& correspondence_accepted,
           const DoubleArray& correspondence_robust, const IndexArray& edges,
           const IndexArray& landmark_indices, const DoubleArray& landmark_targets,
           const py::dict& weights, const py::object& mask) {
            EnergyCall call = to_energy_call(
                template_points, binding_nodes, binding_weights, binding_counts,
                node_positions, correspondence_points, correspondence_normals,
                correspondence_accepted, correspondence_robust, edges,
                landmark_indices, landmark_targets, weights, mask);
            call.inputs.mask = call.mask.active.empty() ? nullptr : &call.mask;
            const nricp::Deformation deformation =
                to_deformation(rotations, translations, node_positions);
            const nricp::FusedSystem fused =
                nricp::assemble_and_energy(call.inputs, deformation, call.weights);

            py::dict result = from_normal_equations(fused.system);
            py::dict energies;
            energies["point_to_point"] = fused.energies.point_to_point;
            energies["point_to_plane"] = fused.energies.point_to_plane;
            energies["smoothness"] = fused.energies.smoothness;
            energies["anchor"] = fused.energies.anchor;
            energies["landmark"] = fused.energies.landmark;
            energies["total"] = fused.energies.total();
            result["energies"] = energies;
            return result;
        },
        py::arg("template_points"), py::arg("binding_nodes"),
        py::arg("binding_weights"), py::arg("binding_counts"),
        py::arg("rotations"), py::arg("translations"), py::arg("node_positions"),
        py::arg("correspondence_points"), py::arg("correspondence_normals"),
        py::arg("correspondence_accepted"), py::arg("correspondence_robust"),
        py::arg("edges"), py::arg("landmark_indices"), py::arg("landmark_targets"),
        py::arg("weights"), py::arg("mask") = py::none(),
        "Fused single-pass system and energies (the path the solver runs).");

    m.def(
        "solve_normal_equations",
        [](const DoubleArray& diagonal, const IndexArray& edges,
           const DoubleArray& edge_blocks, const DoubleArray& rhs, double lambda,
           bool diagonal_scaling) {
            nricp::NormalEquations system =
                to_normal_equations(diagonal, edges, edge_blocks, rhs);
            if (lambda > 0.0) {
                system = nricp::damped(system, lambda, diagonal_scaling);
            }
            const nricp::DirectSolveResult solved = nricp::solve_direct(system);
            py::array_t<double> delta(
                {static_cast<py::ssize_t>(solved.delta.size())});
            auto view = delta.mutable_unchecked<1>();
            for (std::size_t i = 0; i < solved.delta.size(); ++i) {
                view(static_cast<py::ssize_t>(i)) = solved.delta[i];
            }
            py::dict result;
            result["delta"] = delta;
            result["ok"] = solved.ok;
            result["residual_norm"] = solved.residual_norm;
            return result;
        },
        py::arg("diagonal"), py::arg("edges"), py::arg("edge_blocks"),
        py::arg("rhs"), py::arg("damping") = 0.0,
        py::arg("diagonal_scaling") = false,
        "Solve the reduced system (sparse Cholesky, double precision).");

    m.def(
        "normal_equations_dense",
        [](const DoubleArray& diagonal, const IndexArray& edges,
           const DoubleArray& edge_blocks, const DoubleArray& rhs) {
            const nricp::NormalEquations system =
                to_normal_equations(diagonal, edges, edge_blocks, rhs);
            const std::vector<double> dense = nricp::dense_matrix(system);
            const py::ssize_t size =
                static_cast<py::ssize_t>(system.node_count) * 6;
            py::array_t<double> array({size, size});
            auto view = array.mutable_unchecked<2>();
            for (py::ssize_t row = 0; row < size; ++row) {
                for (py::ssize_t column = 0; column < size; ++column) {
                    view(row, column) =
                        dense[static_cast<std::size_t>(row) * size + column];
                }
            }
            return array;
        },
        py::arg("diagonal"), py::arg("edges"), py::arg("edge_blocks"),
        py::arg("rhs"),
        "Dense expansion of the reduced system (verification only).");

    const auto route_arguments = [](py::module_& module, const char* name,
                                    const char* doc, bool use_device) {
        module.def(
            name,
            [use_device](
                const DoubleArray& template_points, const IndexArray& template_faces,
                const DoubleArray& target_points, const IndexArray& target_faces,
                const py::list& level_settings, const py::object& mask,
                const py::object& landmark_indices,
                const py::object& landmark_targets, double distance_threshold,
                double min_normal_cosine, double min_slope, double initial_damping,
                double damping_decrease, double damping_increase,
                int max_damping_attempts, bool diagonal_damping, int graph_levels,
                int nodes_initial, int growth, double pcg_tolerance,
                int pcg_max_iterations) {
                PreparedSolve prepared = prepare_solve(
                    template_points, template_faces, target_points, target_faces,
                    level_settings, mask, landmark_indices, landmark_targets,
                    distance_threshold, min_normal_cosine, min_slope, initial_damping,
                    damping_decrease, damping_increase, max_damping_attempts,
                    diagonal_damping, graph_levels, nodes_initial, growth,
                    pcg_tolerance, pcg_max_iterations, use_device);
                const nricp::VertexMask* mask_pointer =
                    prepared.mask.active.empty() ? nullptr : &prepared.mask;
                // The accelerated route builds its own query structure on the
                // device, so the core layer's is only built for the route that
                // traverses it.
                nricp::RouteResult solved;
                if (use_device) {
                    solved = nricp::solve_route_b(
                        prepared.normalized_template, prepared.normalized_target,
                        prepared.graph, prepared.settings, mask_pointer,
                        prepared.landmarks);
                } else {
                    const nricp::TargetQueries queries(prepared.normalized_target);
                    solved = nricp::solve_route_a(
                        prepared.normalized_template, queries, prepared.graph,
                        prepared.settings, mask_pointer, prepared.landmarks);
                }
                return route_result_to_dict(solved, prepared.graph);
            },
            py::arg("template_points"), py::arg("template_faces"),
            py::arg("target_points"), py::arg("target_faces"),
            py::arg("level_settings"), py::arg("mask") = py::none(),
            py::arg("landmark_indices") = py::none(),
            py::arg("landmark_targets") = py::none(),
            py::arg("distance_threshold") = std::numeric_limits<double>::infinity(),
            py::arg("min_normal_cosine") = -1.0, py::arg("min_slope") = 0.0,
            py::arg("initial_damping") = 1e-3, py::arg("damping_decrease") = 0.3,
            py::arg("damping_increase") = 4.0, py::arg("max_damping_attempts") = 6,
            py::arg("diagonal_damping") = true, py::arg("graph_levels") = 4,
            py::arg("nodes_initial") = 100, py::arg("growth") = 2,
            py::arg("pcg_tolerance") = 1e-5, py::arg("pcg_max_iterations") = 60, doc);
    };

    const char* route_doc =
        "Fit the template onto the target and return the result plus the run "
        "report. Distances (distance_threshold, robust_threshold) are given in "
        "the caller's units and converted into the solver's normalized frame "
        "internally; the caller hands in the template and target already aligned "
        "to each other.";
    route_arguments(m, "solve_route_a", route_doc, /*use_device=*/false);
    route_arguments(m, "solve_route_b", route_doc, /*use_device=*/true);

    const char* session_doc =
        "The route B schedule with the caller holding the loop: construct the "
        "session, then call step() until finished(). Between steps the current "
        "shape is available through vertices(), so a host that draws the "
        "template can redraw as the fit progresses and stop early instead of "
        "waiting for a result that no longer makes sense.";

    // The same schedule with the caller holding the loop. This is the shape
    // qmyidp's cloth solver exposes (initialize, then step until satisfied), and
    // it exists so a host that draws the template -- Blender, for instance -- can
    // redraw between steps and stop early instead of waiting out the whole
    // coarse-to-fine schedule. `step` releases the interpreter lock, so a
    // host that runs this on a worker thread does not freeze its UI.
    py::class_<nricp::RouteBSession>(m, "RouteBSession", session_doc)
        .def(
            py::init([](
                         const DoubleArray& template_points,
                         const IndexArray& template_faces,
                         const DoubleArray& target_points,
                         const IndexArray& target_faces,
                         const py::list& level_settings, const py::object& mask,
                         const py::object& landmark_indices,
                         const py::object& landmark_targets,
                         double distance_threshold, double min_normal_cosine,
                         double min_slope, double initial_damping,
                         double damping_decrease, double damping_increase,
                         int max_damping_attempts, bool diagonal_damping,
                         int graph_levels, int nodes_initial, int growth,
                         double pcg_tolerance, int pcg_max_iterations) {
                PreparedSolve prepared = prepare_solve(
                    template_points, template_faces, target_points, target_faces,
                    level_settings, mask, landmark_indices, landmark_targets,
                    distance_threshold, min_normal_cosine, min_slope,
                    initial_damping, damping_decrease, damping_increase,
                    max_damping_attempts, diagonal_damping, graph_levels,
                    nodes_initial, growth, pcg_tolerance, pcg_max_iterations,
                    /*use_device=*/true);
                const nricp::VertexMask* mask_pointer =
                    prepared.mask.active.empty() ? nullptr : &prepared.mask;
                return nricp::RouteBSession(
                    prepared.normalized_template, prepared.normalized_target,
                    prepared.graph, prepared.settings, mask_pointer,
                    prepared.landmarks);
            }),
            py::arg("template_points"), py::arg("template_faces"),
            py::arg("target_points"), py::arg("target_faces"),
            py::arg("level_settings"), py::arg("mask") = py::none(),
            py::arg("landmark_indices") = py::none(),
            py::arg("landmark_targets") = py::none(),
            py::arg("distance_threshold") = std::numeric_limits<double>::infinity(),
            py::arg("min_normal_cosine") = -1.0, py::arg("min_slope") = 0.0,
            py::arg("initial_damping") = 1e-3, py::arg("damping_decrease") = 0.3,
            py::arg("damping_increase") = 4.0, py::arg("max_damping_attempts") = 6,
            py::arg("diagonal_damping") = true, py::arg("graph_levels") = 4,
            py::arg("nodes_initial") = 100, py::arg("growth") = 2,
            py::arg("pcg_tolerance") = 1e-5, py::arg("pcg_max_iterations") = 60)
        .def(
            "step",
            [](nricp::RouteBSession& session) {
                nricp::WrapProgress progress;
                {
                    py::gil_scoped_release released;
                    progress = session.step();
                }
                py::dict out;
                out["level"] = progress.level;
                out["outer"] = progress.outer;
                out["inner"] = progress.inner;
                out["finished"] = progress.finished;
                out["level_converged"] = progress.level_converged;
                out["energy"] = progress.energy;
                out["correspondence_mean"] = progress.correspondence_mean;
                return out;
            },
            "Advance the schedule by one unit of work and report where it got to.")
        .def("finished", &nricp::RouteBSession::finished,
             "True once the schedule has nothing left to do.")
        .def(
            "vertices",
            [](const nricp::RouteBSession& session) {
                std::vector<nricp::Vec3> vertices;
                {
                    py::gil_scoped_release released;
                    vertices = session.vertices();
                }
                return from_points(vertices);
            },
            "The template as it currently stands, for redrawing between steps.")
        .def(
            "report",
            [](const nricp::RouteBSession& session) {
                py::gil_scoped_release released;
                return nricp::to_json(session.report());
            },
            "What has run so far, as JSON; the level in progress is the last entry.")
        .def(
            "result",
            [](const nricp::RouteBSession& session) {
                nricp::RouteResult solved;
                {
                    py::gil_scoped_release released;
                    solved = session.result();
                }
                return route_result_to_dict(solved, session.graph());
            },
            "The same payload solve_route_b returns, for the state reached so "
            "far; calling it before the schedule finishes gives the partial fit.");

    m.def(
        "device_info",
        [] {
            py::dict info;
            info["available"] = nricp::device_available();
            info["device"] = nricp::device_description();
            return info;
        },
        "Report whether the accelerated path can run here, and on what.");
}

