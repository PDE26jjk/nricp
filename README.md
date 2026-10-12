# nricp — deformation-graph wrapping solver

Fits a fixed-topology template mesh onto a target surface of arbitrary topology
by minimizing an embedded-deformation-graph energy, so that the difference
between the fitted and the rest template is a same-topology morph
(`ΔV = V_wrapped − V_template`). The algorithm is the one behind the "wrapping"
step of commercial head-wrapping tools: non-rigid ICP over a sampled deformation
graph, coarse to fine, with correspondence filtering and robust weighting.

The solver is native C++/CUDA and is consumed as the extension module
`nricp_dp`, which loads in Blender-hosted Python and in a standalone interpreter
of the same ABI.

## Status

Scaffold only: the extension module reports its own version and build target,
and nothing solves yet. The implementation plan lives in the change
`nricp-wrap` of the parent workspace (`openspec/changes/nricp-wrap/`); this
repository carries the algorithm specification it implements:

- `docs/algorithm-spec.md` — energy, Jacobians, sparsity, schedules, pitfalls,
  and the acceptance numbers. This is the specification the code follows.
- `docs/wrap-wrapping-algorithm-research.md` — how the commercial node's
  algorithm was recovered (documentation and binary symbol evidence).
- `docs/cpp-cuda-deformation-graph-survey.md` — module breakdown, candidate
  implementations, and the GPU route analysis.

## Layout

```
src/core/       meshes, normalization, sampling, binding, levels, masks,
                settings, run report
src/geometry/   closest-point query structure over the target
src/energy/     residual blocks and their analytic Jacobians
src/solver/     outer driver and the three solve routes
src/reference/  double-precision CPU path (the numerical reference)
src/cuda/       device-side kernels
src/python/     extension module `nricp_dp`
tests/          pytest suites
bench/          TOSCA benchmark harness and golden regeneration
docs/           specification and surveys
extern/         vendored build dependencies (submodules)
```

## Dependencies

The produced module has no runtime dependency of its own: it needs the
interpreter it was built for, plus that interpreter's own numpy if and only if
array arguments are used (Blender 5.2 bundles Python 3.13 and numpy 2.3.4).
Nothing has to be installed into Blender's Python.

Build-time dependencies are vendored as submodules so a fresh clone builds
without installing anything system-wide:

- `extern/pybind11` — pinned to v3.0.2
- `extern/eigen` — pinned to 3.4.0 (header-only, used by the CPU reference
  route; compiled in, never shipped as a library)

Development-only dependencies (test harness, benchmark, golden regeneration)
are listed in `requirements-dev.txt`. The mesh library there is the reference
implementation the acceptance numbers were measured with; it is never imported
by the module or by anything the module needs.

## Build

Windows, Visual Studio 2022, and the CUDA 12.1 toolkit:

```powershell
$env:CUDA_PATH='C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1'
$env:CudaToolkitDir='C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\'
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -T "cuda=12.1" `
      -DPython_EXECUTABLE=<python 3.13>
cmake --build build --config Release --parallel 16
```

The module lands in `build/Release/nricp_dp.cp313-win_amd64.pyd`. Always pass
`--parallel`: the Visual Studio generator compiles CUDA translation units
one at a time otherwise. After editing a header used by CUDA sources, force a
full rebuild — the generator does not track those dependencies, and a stale
device-side caller fails at run time.

## Tests and benchmark

```powershell
$env:NRICP_BUILD_DIR='build/Release'
python -m pytest -q                     # unit and API tests
python bench\prepare_data.py --source <dataset directory>
python bench\golden_regen.py            # golden fixtures for the reference
python bench\run_bench.py --backend reference
```

The benchmark datasets are not distributed with this repository. See
`bench/README.md` for what has to be prepared locally and what the acceptance
gates are.

## Third-party material

Licenses and attribution for vendored dependencies and for the reference
material consulted during development are recorded in `NOTICE`. Benchmark
datasets are not part of the repository, and no code without a license is
copied into it.

## License

GNU General Public License version 2 (see `LICENSE`), matching the Blender
add-on this solver serves. The vendored dependencies keep their own licenses
(`NOTICE`); everything ships as one GPL-2.0 work.

Two consequences worth knowing before adding dependencies: GPL-2.0 cannot be
combined with Apache-2.0 code, so a GPU BVH library under Apache-2.0 (NVIDIA's
cuBQL, for example) could not be linked in as-is — an MIT-style or
Apache-compatible alternative, or the project's own query structure, has to be
used instead. And the license is version 2.0 only, not "or later": if a later
choice of dependency needs that, the license text has to change before the
dependency does.
