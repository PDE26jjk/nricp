#include "core/build_info.h"

#include "core/version.h"

namespace nricp {
namespace {

#ifndef NRICP_BUILD_TYPE
#define NRICP_BUILD_TYPE "unknown"
#endif
#ifndef NRICP_CXX_COMPILER_ID
#define NRICP_CXX_COMPILER_ID "unknown"
#endif
#ifndef NRICP_CUDA_TOOLKIT
#define NRICP_CUDA_TOOLKIT "none"
#endif
#ifndef NRICP_CUDA_ARCHITECTURES
#define NRICP_CUDA_ARCHITECTURES "none"
#endif

}  // namespace

const char* version() { return NRICP_VERSION_STRING; }
const char* build_type() { return NRICP_BUILD_TYPE; }
const char* cxx_compiler() { return NRICP_CXX_COMPILER_ID; }
const char* cuda_toolkit() { return NRICP_CUDA_TOOLKIT; }
const char* cuda_architectures() { return NRICP_CUDA_ARCHITECTURES; }
const char* accelerated_precision() { return "float"; }

}  // namespace nricp
