#pragma once

#include <cuda_runtime.h>

#include "cuda/assemble.cuh"
#include "cuda/host_staging.cuh"

// ---------------------------------------------------------------------------
// Route C: the global/local alternation
// ---------------------------------------------------------------------------
//
// Two steps alternate inside one inner iteration. The local step replaces every
// node's rotation by the closed-form fit of the influences that mention it; the
// global step holds those rotations fixed and solves for the translations.
//
// What makes the global step cheap is not that it is smaller but that its
// matrix does not move. With the rotation fixed, a data residual's Jacobian
// with respect to a translation is the binding weight (times the plane normal
// for the plane term, and the same shape for the landmark rows), the smoothness
// term's is two copies of the identity, and the anchor's is the identity again:
// none of them mentions a rotation. So the matrix is assembled once per
// correspondence update -- `assemble_translation_matrix` -- and only the
// right-hand side, which does depend on the rotations, follows the local step.
//
// In the shared six-values-per-node layout that matrix is the same Hessian
// restricted to its translation corner: the rotation half of a diagonal block
// becomes the identity, so the shared preconditioner stays well conditioned,
// and the coupling between the halves is dropped. The rotation half of the
// right-hand side is exactly zero, so the increment the shared solver returns
// has exactly zero rotation components -- and `exp(0)` is exactly the identity,
// so the rotations the local step fitted survive the retraction untouched.
namespace nricp {
namespace gpu {

// The global step's matrix, plus the energies of the state it was assembled
// from. One call per correspondence update.
EnergyReadout assemble_translation_matrix(
    GpuLevel& level, const AssembleInputs& inputs, PinnedFloats& staging,
    cudaStream_t stream);

// The global step's right-hand side. One call per solve.
void assemble_translation_rhs(
    GpuLevel& level, const AssembleInputs& inputs, cudaStream_t stream);

// The local step: every node's rotation, replaced by the closed-form fit of the
// influences that touch it.
void fit_rotations(
    GpuLevel& level, const AssembleInputs& inputs, cudaStream_t stream);

}  // namespace gpu
}  // namespace nricp
