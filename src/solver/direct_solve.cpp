#include "solver/direct_solve.h"

#include <Eigen/Sparse>

#include <cmath>
#include <vector>

namespace nricp {

DirectSolveResult solve_direct(const NormalEquations& system) {
    DirectSolveResult result;
    const int size = system.node_count * 6;
    if (size <= 0) {
        return result;
    }

    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(static_cast<std::size_t>(system.node_count) * 36 +
                     system.off_diagonal.size() * 72);
    for (int node = 0; node < system.node_count; ++node) {
        const Mat6& block = system.diagonal[static_cast<std::size_t>(node)];
        for (int row = 0; row < 6; ++row) {
            for (int column = 0; column < 6; ++column) {
                triplets.emplace_back(node * 6 + row, node * 6 + column,
                                      block.at(row, column));
            }
        }
    }
    for (const NormalEquations::OffDiagonal& block : system.off_diagonal) {
        for (int row = 0; row < 6; ++row) {
            for (int column = 0; column < 6; ++column) {
                const double value = block.value.at(row, column);
                triplets.emplace_back(block.a * 6 + row, block.b * 6 + column, value);
                triplets.emplace_back(block.b * 6 + column, block.a * 6 + row, value);
            }
        }
    }

    Eigen::SparseMatrix<double> matrix(size, size);
    matrix.setFromTriplets(triplets.begin(), triplets.end());

    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
    solver.compute(matrix);
    if (solver.info() != Eigen::Success) {
        return result;
    }

    Eigen::VectorXd rhs(size);
    for (int i = 0; i < size; ++i) {
        rhs[i] = system.rhs[static_cast<std::size_t>(i)];
    }
    const Eigen::VectorXd solution = solver.solve(rhs);
    if (solver.info() != Eigen::Success) {
        return result;
    }

    result.delta.resize(static_cast<std::size_t>(size));
    for (int i = 0; i < size; ++i) {
        result.delta[static_cast<std::size_t>(i)] = solution[i];
    }
    result.residual_norm = (matrix * solution - rhs).norm();
    result.ok = true;
    return result;
}

}  // namespace nricp
