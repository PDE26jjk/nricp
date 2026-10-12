#include "cuda/gpu_solver.cuh"

#include <cuda_runtime.h>

#include <string>

// Route B's schedule -- the coarse-to-fine loop and the accept/reject test --
// lives in session.cu, because it is a sequence of decisions rather than a
// sequence of kernels. What stays here is the module's surface: the batch entry
// point, and the two device queries a caller uses to decide whether the
// accelerated route can run at all.
namespace nricp {

RouteResult solve_route_b(
    const Mesh& normalized_template, const Mesh& normalized_target,
    const TemplateGraph& graph, const RouteSettings& settings,
    const VertexMask* mask, const std::vector<Landmark>& landmarks) {
    RouteBSession session(normalized_template, normalized_target, graph, settings,
                          mask, landmarks);
    while (!session.finished()) {
        session.step();
    }
    return session.result();
}

// Route C is the same session with the other route: the schedule, the energies,
// the correspondences and the reports are route B's, and only the inner step
// differs.
RouteResult solve_route_c(
    const Mesh& normalized_template, const Mesh& normalized_target,
    const TemplateGraph& graph, const RouteSettings& settings,
    const VertexMask* mask, const std::vector<Landmark>& landmarks) {
    RouteBSession session(normalized_template, normalized_target, graph, settings,
                          mask, landmarks, DeviceRoute::LocalGlobal);
    while (!session.finished()) {
        session.step();
    }
    return session.result();
}

bool device_available() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

std::string device_description() {
    if (!device_available()) {
        return "no CUDA device";
    }
    cudaDeviceProp properties{};
    if (cudaGetDeviceProperties(&properties, 0) != cudaSuccess) {
        return "CUDA device present, properties unavailable";
    }
    return std::string(properties.name) + " (cc " +
           std::to_string(properties.major) + "." +
           std::to_string(properties.minor) + ")";
}

}  // namespace nricp
