#pragma once

#include <cmath>

namespace nricp {

// Row-major 6x6 block. The solver's unknowns are six per node (rotation
// increment first, then translation), so every element of the reduced system is
// one of these.
struct Mat6 {
    double m[36] = {};

    double at(int row, int column) const { return m[row * 6 + column]; }
    double& at(int row, int column) { return m[row * 6 + column]; }

    void add_outer_product(const double lhs[6], const double rhs[6], double weight = 1.0) {
        for (int row = 0; row < 6; ++row) {
            for (int column = 0; column < 6; ++column) {
                at(row, column) += weight * lhs[row] * rhs[column];
            }
        }
    }

    void add_scaled(const Mat6& other, double scale) {
        for (int i = 0; i < 36; ++i) {
            m[i] += scale * other.m[i];
        }
    }

    // Symmetrizes in place: the expected blocks are symmetric by construction,
    // and averaging removes the drift that accumulation order introduces.
    void symmetrize() {
        for (int row = 0; row < 6; ++row) {
            for (int column = row + 1; column < 6; ++column) {
                const double average = 0.5 * (at(row, column) + at(column, row));
                at(row, column) = average;
                at(column, row) = average;
            }
        }
    }
};

}  // namespace nricp
