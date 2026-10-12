cmake_minimum_required(VERSION 3.24)

# Deformation-graph wrapping solver. Two consumers drive the same native code:
# the standalone benchmark/test harness, and Blender-hosted Python (5.2 embeds
# Python 3.13), so the extension module is built for the interpreter given by
# -DPython_EXECUTABLE and is expected to be ABI-compatible with both.
project(nricp VERSION 0.1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Reference target is an RTX 3070 laptop (sm_86). Override with
# -DNRICP_CUDA_ARCHS=<list>; keeping the list short keeps the build time down.
# CMAKE_CUDA_ARCHITECTURES itself defaults to the toolkit's own default (52 for
# CUDA 12.1), which would not run on the reference GPU, so it is always derived
# from this variable.
set(NRICP_CUDA_ARCHS "86" CACHE STRING "CUDA architectures to build for")
set(CMAKE_CUDA_ARCHITECTURES "${NRICP_CUDA_ARCHS}" CACHE STRING "CUDA architectures to build for" FORCE)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

if(MSVC)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/MP>)
    # Keep the CUDA device code readable in the debug configuration; the host
    # side keeps symbols, the device side does not need -G.
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /Zi")
    set(CMAKE_SHARED_LINKER_FLAGS_DEBUG "${CMAKE_SHARED_LINKER_FLAGS_DEBUG} /DEBUG")
endif()

find_package(CUDAToolkit REQUIRED)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module NumPy)

# The module targets the numpy 2.0 C API so one build stays usable by any numpy
# 2.x runtime (Blender ships 2.3.4). Building against a numpy 1.x header set
# would silently produce a module whose compatibility is undefined.
if(Python_NumPy_VERSION VERSION_LESS 2.0)
    message(FATAL_ERROR
        "numpy ${Python_NumPy_VERSION} is too old for the build interpreter; "
        "the module targets the numpy 2.0 API (Blender bundles numpy 2.3.4)")
endif()

message(STATUS "nricp ${PROJECT_VERSION}")
message(STATUS "Python:     ${Python_EXECUTABLE} (${Python_VERSION}, ABI ${Python_SOABI})")
message(STATUS "NumPy:      ${Python_NumPy_VERSION} at ${Python_NumPy_INCLUDE_DIRS}")
message(STATUS "CUDA:       ${CUDAToolkit_VERSION} at ${CUDAToolkit_DIR}")
message(STATUS "CUDA archs: ${CMAKE_CUDA_ARCHITECTURES}")

# Vendored dependencies only: the sub-project has to build on a fresh clone
# without relying on anything installed system-wide.
add_subdirectory(extern/pybind11)

# Layer directories are wired in as they gain sources:
#   src/energy    residual blocks and Jacobians
#   src/reference double-precision CPU path
add_subdirectory(src/core)
add_subdirectory(src/geometry)
add_subdirectory(src/energy)
add_subdirectory(src/solver)
add_subdirectory(src/cuda)
add_subdirectory(src/python)
