# Device side of route B, one translation unit per concern: the module's surface
# (gpu_solver.cu), the schedule (session.cu), the device-side vocabulary
# (solver_common.cuh), the level's buffers (gpu_level), the level's indexing
# (level_layout), the assembly (assemble), the deformation kernels (deform_ops),
# the linear solve (pcg_solver), the host/device boundary (host_staging), the
# target's acceleration structure (lbvh) and the template graph builder
# (graph_build).
#
# No __device__ function crosses a translation unit: the shared ones are inline
# in the headers, so each unit that needs them compiles its own copy, and every
# other one stays where it is used. That keeps device symbols out of the link
# step, so separable compilation (CUDA_SEPARABLE_COMPILATION plus
# CUDA_RESOLVE_DEVICE_SYMBOLS on the module) is not needed. Adding a __device__
# function that one unit calls from another -- or one that a __global__ kernel
# of another unit calls -- would change that.
add_library(nricp_cuda STATIC)

target_sources(nricp_cuda PRIVATE
    assemble.cu
    assemble.cuh
    cuda_error.h
    deform_ops.cu
    deform_ops.cuh
    device_math.cuh
    graph_build.cu
    graph_build.cuh
    gpu_level.cu
    gpu_level.cuh
    gpu_solver.cu
    gpu_solver.cuh
    host_staging.cu
    host_staging.cuh
    lbvh.cu
    lbvh.cuh
    level_layout.cu
    level_layout.cuh
    pcg_solver.cu
    pcg_solver.cuh
    session.cu
    solver_common.cuh
)

target_include_directories(nricp_cuda PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(nricp_cuda PUBLIC nricp_core nricp_energy nricp_solver)

target_compile_options(nricp_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

if(MSVC)
    # IncludePath is what the CUDA integration hands to IntelliSense. It is
    # overwritten here, so the toolkit headers have to be spelled out or every
    # CUDA type goes unresolved in the editor. device_math.cuh covers the device
    # built-ins the same way, because the toolkit's own atomic headers declare
    # nothing unless __CUDACC__ is defined.
    set_target_properties(nricp_cuda PROPERTIES
        VS_GLOBAL_IncludePath
        "${PROJECT_SOURCE_DIR}/src;${CUDAToolkit_INCLUDE_DIRS}")
endif()
