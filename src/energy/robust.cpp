#include "energy/robust.h"

#include <cmath>

namespace nricp {

RobustKernel robust_kernel_from_string(const std::string& name) {
    if (name == "huber" || name == "Huber") {
        return RobustKernel::Huber;
    }
    if (name == "geman_mcclure" || name == "geman-mcclure" ||
        name == "GemanMcClure") {
        return RobustKernel::GemanMcClure;
    }
    return RobustKernel::None;
}

const char* robust_kernel_name(RobustKernel kernel) {
    switch (kernel) {
        case RobustKernel::Huber:
            return "huber";
        case RobustKernel::GemanMcClure:
            return "geman_mcclure";
        case RobustKernel::None:
        default:
            return "none";
    }
}

double robust_weight(
    RobustKernel kernel, double squared_residual, double threshold) {
    if (squared_residual < 0.0) {
        squared_residual = 0.0;
    }
    switch (kernel) {
        case RobustKernel::Huber: {
            if (threshold <= 0.0) {
                return 1.0;
            }
            const double squared_threshold = threshold * threshold;
            if (squared_residual <= squared_threshold) {
                return 1.0;
            }
            return threshold / std::sqrt(squared_residual);
        }
        case RobustKernel::GemanMcClure: {
            if (threshold <= 0.0) {
                return 1.0;
            }
            const double ratio = squared_residual / (threshold * threshold);
            const double denominator = 1.0 + ratio;
            return 1.0 / (denominator * denominator);
        }
        case RobustKernel::None:
        default:
            return 1.0;
    }
}

}  // namespace nricp
