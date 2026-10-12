add_library(nricp_solver STATIC)

target_sources(nricp_solver PRIVATE
    correspondence.cpp
    correspondence.h
    direct_solve.cpp
    direct_solve.h
    fused_assembly.cpp
    fused_assembly.h
    normal_equations.cpp
    normal_equations.h
    route_a.cpp
    route_a.h
    route_shared.h
)

target_include_directories(nricp_solver PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

# Eigen is header-only and provides the sparse Cholesky factorization behind
# route A. It is compiled in, never shipped as a library.
target_include_directories(nricp_solver PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../extern/eigen
)

target_link_libraries(nricp_solver PUBLIC nricp_core nricp_geometry)

if(MSVC)
    set_target_properties(nricp_solver PROPERTIES
        VS_GLOBAL_IncludePath "${PROJECT_SOURCE_DIR}/src")
endif()
