#pragma once

#include <vector>

#include "core/mat6.h"
#include "energy/energy.h"

namespace nricp {

// The reduced (Gauss-Newton) system: A delta = rhs, where A is block sparse with
// one 6x6 block per node on the diagonal and one per interacting node pair off
// it. `rhs` is -J^T r, so solving gives the step that lowers the energy.
struct NormalEquations {
    int node_count = 0;
    std::vector<Mat6> diagonal;

    struct OffDiagonal {
        int a = -1;  // always the lower index
        int b = -1;
        Mat6 value;
    };
    std::vector<OffDiagonal> off_diagonal;  // upper triangle, a < b
    std::vector<double> rhs;                // six values per node
};

NormalEquations assemble_normal_equations(
    const std::vector<ResidualBlock>& blocks, int node_count);

// Levenberg-Marquardt damping: A + lambda * D. With diagonal scaling, D is the
// diagonal of A, otherwise the identity.
NormalEquations damped(
    const NormalEquations& system, double lambda, bool diagonal_scaling = false);

// Dense expansion ((node_count * 6) square, row major). Verification only: it
// exists so an independent check can compare the sparse assembly against a dense
// reference.
std::vector<double> dense_matrix(const NormalEquations& system);

}  // namespace nricp
