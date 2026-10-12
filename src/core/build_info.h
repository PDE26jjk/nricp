#pragma once

namespace nricp {

// Compile-time facts about this artifact. The values are injected by the build
// system (see src/core/CMakeLists.txt) and are reported to callers so that a
// stored result can be attributed to a specific build.
const char* version();
const char* build_type();
const char* cxx_compiler();
const char* cuda_toolkit();
const char* cuda_architectures();

// Precision of the accelerated path. The reference path is always double.
const char* accelerated_precision();

}  // namespace nricp
