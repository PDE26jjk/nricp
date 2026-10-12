Device side of the solver.

- `device_math.cuh` — device scalar types and small linear algebra (3x3, 6x6,
  SO(3) exponential and logarithm, the Cholesky factorisations), plus the atomic
  declarations an editor needs when it parses a `.cu` without `__CUDACC__`.
- `lbvh.cuh`, `lbvh.cu` — the target's Morton-code LBVH: codes from the triangle
  centroids, a thrust radix sort, Karras' linear build, a bottom-up box fit, and
  the nearest-point traversal with the correspondence filters. Built and kept on
  the device; the host never reads the hierarchy back.
- `gpu_solver.cuh`, `gpu_solver.cu` — route B: the deformation and normal
  kernels, correspondence filtering and reweighting, the fused accumulation of
  the reduced system, and the damped solve as a single cooperative kernel. The
  header exposes `solve_route_b` (the batch entry point) and `RouteBSession`
  (initialize, then one `step()` per iteration, for a caller that wants to watch
  the fit and stop early).
- `cuda_error.h` — the one error-reporting helper both translation units share.
