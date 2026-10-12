add_library(nricp_core STATIC)

# Headers are listed as sources so the IDE shows them in the project tree
# (the same convention the sibling cloth project uses).
target_sources(nricp_core PRIVATE
    build_info.cpp
    build_info.h
    constraints.cpp
    constraints.h
    deformation.cpp
    deformation.h
    graph.cpp
    graph.h
    mat3.h
    mat6.h
    mesh.cpp
    mesh.h
    quaternion.h
    report.cpp
    report.h
    rotation.h
    sampling.cpp
    sampling.h
    vec3.h
    version.h
)

target_include_directories(nricp_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

# Eigen is header-only and used for the 3x3 singular value decomposition inside
# the rigid fits. It is compiled in, never shipped as a library.
target_include_directories(nricp_core PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../extern/eigen
)

target_compile_definitions(nricp_core PUBLIC
    NRICP_VERSION="${PROJECT_VERSION}"
    NRICP_BUILD_TYPE="$<CONFIG>"
    NRICP_CXX_COMPILER_ID="${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}"
    NRICP_CUDA_TOOLKIT="${CUDAToolkit_VERSION}"
    NRICP_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES}"
)

if(MSVC)
    set_target_properties(nricp_core PROPERTIES
        VS_GLOBAL_IncludePath "${PROJECT_SOURCE_DIR}/src")
endif()
