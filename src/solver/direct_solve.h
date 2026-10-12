#pragma once

#include <vector>

#include "solver/normal_equations.h"

namespace nricp {

struct DirectSolveResult {
    bool ok = false;
    double residual_norm = 0.0;  // |A x - rhs| after the solve
    std::vector<double> delta;
};

// Route A's linear solve: a sparse Cholesky factorization of the block-sparse
// system, in double precision. The system is symmetric positive definite once
// damping (or an anchor) is present.
DirectSolveResult solve_direct(const NormalEquations& system);

}  // namespace nricp
