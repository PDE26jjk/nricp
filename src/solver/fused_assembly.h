#pragma once

#include "energy/energy.h"
#include "solver/normal_equations.h"

namespace nricp {

struct FusedSystem {
    NormalEquations system;
    BlockEnergies energies;
};

// Single pass over the current state: accumulates the reduced system and the
// block energies straight from the residual formulas, one row at a time, without
// materializing a residual block (and its 4x6x6 Jacobian array) per residual.
// At production scale that materialization is hundreds of megabytes per
// iteration; this path allocates only the system itself.
//
// The block-based route (build_blocks + assemble_normal_equations) stays as the
// reference the fast path is verified against, and as the Python-facing
// verification interface.
FusedSystem assemble_and_energy(
    const EnergyInputs& inputs,
    const Deformation& deformation,
    const EnergyWeights& weights);

// Energy only, with no Jacobian arithmetic at all: what the accept/reject test
// needs.
BlockEnergies accumulate_energy(
    const EnergyInputs& inputs,
    const Deformation& deformation,
    const EnergyWeights& weights);

}  // namespace nricp
