#include "solver/normal_equations.h"

#include <algorithm>
#include <map>
#include <utility>

namespace nricp {
namespace {

// One row of a residual block turned into the 6-vectors the assembly consumes.
void row_vectors(
    const ResidualBlock& block, int row, int slot, double values[6]) {
    for (int column = 0; column < 6; ++column) {
        values[column] = block.jacobians[slot][row][column];
    }
}

}  // namespace

NormalEquations assemble_normal_equations(
    const std::vector<ResidualBlock>& blocks, int node_count) {
    NormalEquations system;
    system.node_count = node_count;
    system.diagonal.assign(static_cast<std::size_t>(node_count), Mat6{});
    system.rhs.assign(static_cast<std::size_t>(node_count) * 6, 0.0);

    // Interaction blocks are accumulated through a map keyed by the node pair,
    // so the emitted order is a function of the input alone.
    std::map<std::pair<int, int>, Mat6> interactions;

    for (const ResidualBlock& block : blocks) {
        for (int slot = 0; slot < block.slots; ++slot) {
            const int node = block.nodes[slot];
            if (node < 0 || node >= node_count) {
                continue;
            }
            for (int row = 0; row < block.row_count; ++row) {
                double jacobian[6];
                row_vectors(block, row, slot, jacobian);
                // Diagonal block: J_s^T J_s.
                system.diagonal[static_cast<std::size_t>(node)].add_outer_product(
                    jacobian, jacobian);
                // Right-hand side: -J_s^T r.
                for (int column = 0; column < 6; ++column) {
                    system.rhs[static_cast<std::size_t>(node) * 6 + column] -=
                        jacobian[column] * block.values[row];
                }
                // Off-diagonal blocks with every later slot of the same block.
                for (int other = slot + 1; other < block.slots; ++other) {
                    const int other_node = block.nodes[other];
                    if (other_node < 0 || other_node >= node_count) {
                        continue;
                    }
                    double other_jacobian[6];
                    row_vectors(block, row, other, other_jacobian);
                    const std::pair<int, int> key =
                        node < other_node ? std::make_pair(node, other_node)
                                          : std::make_pair(other_node, node);
                    Mat6& target = interactions[key];
                    if (node < other_node) {
                        target.add_outer_product(jacobian, other_jacobian);
                    } else {
                        target.add_outer_product(other_jacobian, jacobian);
                    }
                }
            }
        }
    }

    system.off_diagonal.reserve(interactions.size());
    for (auto& entry : interactions) {
        NormalEquations::OffDiagonal block;
        block.a = entry.first.first;
        block.b = entry.first.second;
        block.value = entry.second;
        system.off_diagonal.push_back(block);
    }
    return system;
}

NormalEquations damped(
    const NormalEquations& system, double lambda, bool diagonal_scaling) {
    NormalEquations result = system;
    if (lambda <= 0.0) {
        return result;
    }
    for (std::size_t node = 0; node < result.diagonal.size(); ++node) {
        for (int row = 0; row < 6; ++row) {
            const double scale =
                diagonal_scaling ? std::max(result.diagonal[node].at(row, row), 1e-12)
                                 : 1.0;
            result.diagonal[node].at(row, row) += lambda * scale;
        }
    }
    return result;
}

std::vector<double> dense_matrix(const NormalEquations& system) {
    const std::size_t size = static_cast<std::size_t>(system.node_count) * 6;
    std::vector<double> dense(size * size, 0.0);
    for (int node = 0; node < system.node_count; ++node) {
        for (int row = 0; row < 6; ++row) {
            for (int column = 0; column < 6; ++column) {
                dense[(static_cast<std::size_t>(node) * 6 + row) * size +
                      static_cast<std::size_t>(node) * 6 + column] +=
                    system.diagonal[static_cast<std::size_t>(node)].at(row, column);
            }
        }
    }
    for (const NormalEquations::OffDiagonal& block : system.off_diagonal) {
        for (int row = 0; row < 6; ++row) {
            for (int column = 0; column < 6; ++column) {
                const double value = block.value.at(row, column);
                dense[(static_cast<std::size_t>(block.a) * 6 + row) * size +
                      static_cast<std::size_t>(block.b) * 6 + column] += value;
                // The pair block is the transpose contribution in the opposite
                // corner, which is what makes the assembled matrix symmetric.
                dense[(static_cast<std::size_t>(block.b) * 6 + column) * size +
                      static_cast<std::size_t>(block.a) * 6 + row] += value;
            }
        }
    }
    return dense;
}

}  // namespace nricp
