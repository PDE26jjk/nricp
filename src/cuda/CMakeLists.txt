# Device side of the solver: the solver itself (gpu_solver.cu) and the target's
# acceleration structure (lbvh.cu). The two translation units share no device
# symbol -- every __device__ function in lbvh.cu is internal to it and lbvh.cuh
# only declares host methods -- so no device link step is needed. Adding a
# __device__ function that one unit calls from the other would change that, and
# separable compilation (CUDA_SEPARABLE_COMPILATION plus
# CUDA_RESOLVE_DEVICE_SYMBOLS on the module) would have to be turned on here.
add_library(nricp_cuda STATIC)

target_sources(nricp_cuda PRIVATE
    cuda_error.h
    device_math.cuh
    gpu_solver.cu
    gpu_solver.cuh
    lbvh.cu
    lbvh.cuh
)

target_include_directories(nricp_cuda PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(nricp_cuda PUBLIC nricp_core nricp_energy nricp_solver)

target_compile_options(nricp_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

if(MSVC)
    # IncludePath is what the CUDA integration hands to IntelliSense. It is
    # overwritten here, so the toolkit headers have to be spelled out or every
    # CUDA type goes unresolved in the editor. The device built-ins are covered
    # separately by ide_cuda_shims.h, because the toolkit's own atomic headers
    # declare nothing unless __CUDACC__ is defined.
    set_target_properties(nricp_cuda PROPERTIES
        VS_GLOBAL_IncludePath
        "${PROJECT_SOURCE_DIR}/src;${CUDAToolkit_INCLUDE_DIRS}")
endif()
