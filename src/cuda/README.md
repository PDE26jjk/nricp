Device side of the solver. Route B is split by concern, one translation unit
each, so the file to open is the file for the question being asked.

- `device_math.cuh` — device scalar types and small linear algebra (3x3, 6x6,
  SO(3) exponential and logarithm, the Cholesky factorisations), plus the atomic
  declarations an editor needs when it parses a `.cu` without `__CUDACC__`.
- `solver_common.cuh` — what every other kernel is written in terms of: the
  launch constants, the block reduction, the vector accessors, and the row a
  linear functional of a node transform contributes.
- `gpu_solver.cuh`, `gpu_solver.cu` — route B's surface: `solve_route_b` (the
  batch entry point), `solve_route_c` (the same thing solved by the global/local
  alternation), `RouteBSession` (initialize, then one `step()` per iteration,
  for a caller that wants to watch the fit and stop early; its `route` argument
  selects which of the two device routes it runs), and the device queries.
- `session.cu` — the schedule both routes share: the coarse-to-fine loop, the
  correspondence updates, and the accept/reject test on the damping. The two
  routes differ only where they have to, in one inner step and in which system
  is assembled once per correspondence update.
- `local_global.cuh`, `local_global.cu` — route C's two steps: the closed-form
  rotation fit (Horn's quaternion method, one block per node walking its own
  influences) and the translation-only global step, whose matrix is route B's
  assembly restricted to its translation corner and is therefore assembled once
  per correspondence update while only the right-hand side follows the fit.
- `gpu_level.cuh`, `gpu_level.cu` — `GpuLevel`, everything one level needs on the
  device, sized once for the whole run and reused by every level and every step.
- `level_layout.cuh`, `level_layout.cu` — the level's indexing: the node ->
  bound-vertex lists the gathered accumulation walks, and the node pairs that
  share a bound vertex, grouped so that each group owns one cross block of the
  system.
- `assemble.cuh`, `assemble.cu` — one pass over a level: the reduced system's
  blocks and right-hand side, and the five block energies.
- `deform_ops.cuh`, `deform_ops.cu` — the deformed template, its normals, the
  robust weights, and the correspondence statistics.
- `pcg_solver.cuh`, `pcg_solver.cu` — the block-diagonal preconditioner and the
  whole damped solve as a single cooperative kernel.
- `host_staging.cuh`, `host_staging.cu` — the host/device boundary: the
  conversions between the core layer's types and the flat arrays the kernels
  take, the one stream the route runs on, and the pinned staging the schedule
  reads its decisions from.
- `graph_build.cuh`, `graph_build.cu` — the template's deformation graph (nodes
  by farthest-point sampling, bindings, edges), built on the device.
- `lbvh.cuh`, `lbvh.cu` — the target's Morton-code LBVH: codes from the triangle
  centroids, a thrust radix sort, Karras' linear build, a bottom-up box fit, and
  the nearest-point traversal with the correspondence filters. Built and kept on
  the device; the host never reads the hierarchy back.
- `cuda_error.h` — the one error-reporting helper every translation unit shares.

No `__device__` function crosses a translation unit: the shared ones are inline
in the headers, so each unit that needs them compiles its own copy, and the rest
stay where they are used. That keeps device symbols out of the link step, which
is why the build does not turn on separable compilation; adding a `__device__`
function that one unit calls from another would change that.
