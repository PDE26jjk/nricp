# Extension module consumed by Blender-hosted Python and by the harness.
# Pybind11 derives the file name suffix from the interpreter found above, so
# the produced file is importable as `nricp_dp` in that interpreter.
pybind11_add_module(nricp_dp)

target_sources(nricp_dp PRIVATE
    bindings.cpp
    numpy_config.h
)

if(MSVC)
    set_target_properties(nricp_dp PROPERTIES
        VS_GLOBAL_IncludePath "${PROJECT_SOURCE_DIR}/src")
endif()

target_link_libraries(nricp_dp PRIVATE
    nricp_core
    nricp_geometry
    nricp_energy
    nricp_solver
    nricp_cuda
)


# Array interchange uses the interpreter's own numpy (the module imports nothing
# else); see numpy_config.h for the API level it targets.
target_include_directories(nricp_dp PRIVATE ${Python_NumPy_INCLUDE_DIRS})

# Keep the artifact at a stable, documented location (build/<config>/) instead
# of mirroring the source tree: every consumer (tests, benchmark, Blender) adds
# that one directory to the interpreter path.
set_target_properties(nricp_dp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>"
)

if(MSVC)
    # The module is a shared library: keep its symbols inside a debug build so
    # that a crash in the interpreter is still resolvable to a line.
    target_link_options(nricp_dp PRIVATE $<$<CONFIG:Debug>:/DEBUG>)
endif()
