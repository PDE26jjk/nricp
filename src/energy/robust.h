#pragma once

#include <string>

namespace nricp {

enum class RobustKernel : int {
    None = 0,
    Huber = 1,
    GemanMcClure = 2,
};

RobustKernel robust_kernel_from_string(const std::string& name);
const char* robust_kernel_name(RobustKernel kernel);

// Iterative-reweighting weight u(s) for a residual whose squared norm is
// `squared_residual`. Both kernels return 1 at zero residual and fall off
// monotonically, so a residual that disagrees with the rest of the fit loses
// influence instead of dragging the surface with it. `threshold` is the
// kernel's scale: Huber's knee, Geman-McClure's c.
double robust_weight(RobustKernel kernel, double squared_residual, double threshold);

}  // namespace nricp
