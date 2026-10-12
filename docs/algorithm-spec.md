# NRICP Deformation-Graph Wrapping — Implementation Spec & Survey

Working document for the self-developed C++/CUDA implementation of a Wrap-style
wrapping solver: fit a fixed-topology template mesh (e.g. a basemesh) onto an
arbitrary-topology target surface, so that `ΔV = V_wrapped − V_template` becomes
a same-topology morph.

Algorithm family: non-rigid ICP over an embedded deformation graph
(Amberg et al. 2007 skeleton + Sumner & Popović 2004 correspondence stage +
Sumner et al. 2007 deformation graph), i.e. the kernel of the Faceform Wrap
"Wrapping" node (verified by symbol analysis of Wrap 2025.10.8, see
`../docs/wrap-wrapping-algorithm-research.md`).

---

## 1. Notation

| Symbol | Meaning |
|---|---|
| $v_i,\ i=1..n$ | template vertices (n = tens of thousands) |
| $g_j,\ j=1..m$ | deformation-graph node rest positions (m ≈ 10²–10³, FPS-sampled from template) |
| $N(i)$ | binding of vertex i: its 4 nearest nodes (fixed at init) |
| $w_{ij}$ | binding weight, fixed at init |
| $T_j = (R_j, t_j)$ | node rigid transform; variables per node: $(\omega_j, t_j) \in \mathbb{R}^6$, $R_j = \exp([\omega_j]_\times)$ |
| $y_i, n_i$ | closest point on target + its (interpolated) normal, re-estimated each outer iteration |
| $q_l$ | landmark target position for marked template vertex $l$ |
| $u_i$ | IRLS robust-kernel weight of residual block i |
| $\delta = (\delta\omega_j, \delta t_j)_{j=1..m}$ | tangent-space increment, $6m$ unknowns per GN step |

Normalize both meshes to unit bbox diagonal first — all constants below assume
that scale.

## 2. Deformation model and binding

Binding (computed once on the **undeformed** template, then frozen):

$$
d_{ij}=\|v_i - g_j\|,\qquad
w_{ij} = \frac{(1 - d_{ij}/d_{max})^2}{\sum_{k\in N(i)}(1 - d_{ik}/d_{max})^2},
\qquad |N(i)|=4
$$

Deformation evaluation:

$$
v'_i \;=\; \sum_{j\in N(i)} w_{ij}\,\big(R_j\,(v_i - g_j) + g_j + t_j\big)
\;\equiv\; \sum_{j\in N(i)} w_{ij}\,T_j(v_i)
$$

Multi-level schedules rebuild nodes/bindings per level (§7.4); within a level
the binding is constant, which is what makes the Jacobians below cheap.

## 3. Energy and residual blocks

$$
E \;=\; \lambda_c E_{data} + \lambda_p E_{plane} + \lambda_s E_{smooth} + \lambda_a E_{anchor} + \lambda_l E_{lm} \;(+\; \lambda_{sym} E_{sym})
$$

Each block lists the residual rows and the analytic Jacobians w.r.t. the
tangent-space increment of node j, with the perturbation convention
$R_j \leftarrow \exp([\delta\omega_j]_\times)\,R_j$. Define per vertex
$a_{ij} = R_j(v_i - g_j)$; for smoothness edges $(j,k)$ define
$a_{jk} = R_j(g_k - g_j)$.

### 3.1 Data term — point-to-point (3 rows)

$$
r_i = \sqrt{\lambda_c u_i}\,\big(v'_i - y_i\big),\qquad
\frac{\partial r_i}{\partial \delta\omega_j} = -\sqrt{\lambda_c u_i}\; w_{ij}\,[a_{ij}]_\times,\qquad
\frac{\partial r_i}{\partial \delta t_j} = \sqrt{\lambda_c u_i}\; w_{ij}\, I_3
$$

(`∂v'/∂δω = −w [a]×` comes from $R(v-g)$ perturbed on the left:
$\exp([\delta]_\times) R (v-g) \approx (I + [\delta]_\times) a = a + \delta \times a$.)

### 3.2 Data term — point-to-plane (1 row; primary workhorse)

$$
r_i = \sqrt{\lambda_p u_i}\; n_i^\top (v'_i - y_i),\qquad
\frac{\partial r_i}{\partial \delta\omega_j} = -\sqrt{\lambda_p u_i}\; w_{ij}\; n_i^\top [a_{ij}]_\times,\qquad
\frac{\partial r_i}{\partial \delta t_j} = \sqrt{\lambda_p u_i}\; w_{ij}\, n_i^\top
$$

Keep a p2p component in the mix: a pure plane term is blind to rotations about
$n_i$.

### 3.3 Smoothness — graph edge (j,k), 3 rows

j's transform applied to k's node position must agree with k's own transform:

$$
r_{jk} = \sqrt{\lambda_s}\big(T_j(g_k) - T_k(g_k)\big),\qquad
T_j(g_k) = R_j(g_k - g_j) + g_j + t_j
$$

$$
\frac{\partial r_{jk}}{\partial \delta\omega_j} = -\sqrt{\lambda_s}\,[a_{jk}]_\times,\quad
\frac{\partial r_{jk}}{\partial \delta t_j} = \sqrt{\lambda_s} I,\quad
\frac{\partial r_{jk}}{\partial \delta\omega_k} = 0,\quad
\frac{\partial r_{jk}}{\partial \delta t_k} = -\sqrt{\lambda_s} I
$$

(ARAP-style `SmoothRotationResidual` variant, used by Wrap for symmetry
stiffening: same structure with the right side replaced by a mirrored/
rotation-consistent target.)

### 3.4 Anchor — 6 rows per node

Pulls unobserved nodes to rest and, together with LM damping, guarantees
positive-definiteness of the reduced system:

$$
r_j = \sqrt{\lambda_a}\begin{bmatrix}\omega_j\\ t_j\end{bmatrix},\qquad J = \sqrt{\lambda_a}\,I_6
$$

### 3.5 Landmarks — 3 rows per marker (same form as 3.1)

$$
r = \sqrt{\lambda_l}\,(v'_l - q_l)
$$

Landmark weight must be **generous**: on the TOSCA benchmark the tet-variant
reference implementation defaults to $w_l = 1000$ and reaches ~1% diag error,
while a weak schedule (10) leaves 3–5× larger error.

### 3.6 Mirror symmetry (optional)

For a symmetry plane $(n, c)$ with mirror $M = I - 2nn^\top$ and vertex pairs
$(i,\bar i)$:

$$
r = \sqrt{\lambda_{sym}}\,(M\,v'_i - v'_{\bar i}),\qquad
J = \sqrt{\lambda_{sym}}\,(M\,J_{v'_i} - J_{v'_{\bar i}})
$$

Also usable to *establish* symmetric vertex indices on near-symmetric targets
(Wrap's `ARAPWithMirrorResidual`).

## 4. Normal equations — derivation

Weighted nonlinear least squares (W absorbs block sizes, λ-constants and IRLS
weights $u_i = \rho'(s_i)$ from the robust kernel):

$$
\min_x\; E(x) = r(x)^\top W r(x)
$$

Gauss–Newton: linearize at the current iterate in the tangent space,

$$
r(x \oplus \delta) \approx r + J\delta
$$

Substitute (using $W^\top = W$):

$$
E(\delta) \approx (r + J\delta)^\top W (r + J\delta)
= \delta^\top J^\top W J\, \delta + 2\,r^\top W J\, \delta + \text{const}
$$

Gradient vanishes:

$$
\nabla_\delta E = 2J^\top W J\,\delta + 2J^\top W r = 0
\;\Longrightarrow\;
(J^\top W J + \lambda_{LM} D)\,\delta = -J^\top W r
$$

LM damped variant on the right; $D = I$ or $\mathrm{diag}(J^\top W J)$.
Adapt $\lambda_{LM}$ by the gain ratio
$\kappa = \Delta E_{real} / \Delta E_{pred}$ (increase λ and re-solve if
$\kappa \le 0$; shrink λ when $\kappa \approx 1$). Update with retraction
$x \leftarrow x \oplus \delta$, i.e.
$\omega_j \leftarrow \mathrm{Log}(\exp([\delta\omega_j]_\times) R_j)$,
$R_j \leftarrow \exp([\omega_j]_\times)$.

**Sparsity.** $J^\top WJ$ is 6m×6m block-sparse: data/landmark/anchor blocks
contribute only diagonal 6×6 blocks $A_{jj}$; each graph edge contributes one
symmetric off-diagonal pair $A_{jk}$. RHS $-J^\top W r$ is a sum of per-block
6-vectors. Block assembly mirrors cloth-Hessian assembly: one 6×6 accumulator
per node (atomic scatter), one 6×6 per edge (one thread per edge, no
contention).

## 5. Solver options

1. **Block-sparse GN + PCG** with node-diagonal 6×6 preconditioner (invert m
   blocks once per solve; m ≤ few k ⇒ trivial memory). Matrix-free variant:
   two kernels $y = J v$, $z = J^\top y$; no global matrix assembly.
   This is the SurfelWarp route and matches Wrap's
   `BlockDiagonalPreconditioner` symbol.
2. **Direct solve** (cuDSS / CHOLMOD / Eigen SimplicialLDLT): 6m ≈ 12–24k
   unknowns ⇒ milliseconds; smallest code; needs one JᵀWJ assembly per GN
   step (atomic CSR assembly). Recommended for the first version.
3. **Local/global (rotation fitting)** — simplest to get right first:
   *Global step:* with $R_j$ fixed, $v'_i = c_i + \sum_{j\in N(i)} w_{ij} t_j$
   where $c_i = \sum_j w_{ij}(R_j(v_i-g_j) + g_j)$; smoothness becomes
   $r_{jk} = (R_j - I)(g_k - g_j) + t_j - t_k$. Linear least squares in
   $\{t_j\}$.
   *Local step:* closed-form SVD fit of each $R_j$. With
   $e_{jk} = g_k - g_j$, desired $f_{jk} = (g_k + t_k) - (g_j + t_j)$:

   $$
   S_j = \sum_k u_{jk} f_{jk} e_{jk}^\top,\qquad
   S_j = U \Sigma V^\top,\qquad
   R_j = V\,\mathrm{diag}(1,1,\det(VU^\top))\,U^\top
   $$

All three converge to the same minimizer up to tolerance; (3) is the fastest to
implement and validates against the golden reference, (1)/(2) are the
performance targets.

## 6. Outer ICP loop

Per outer iteration:

1. **Correspondence update** (BVH over target, rebuilt per level):
   $y_i$ = closest point, $n_i$ = interpolated normal. Reject:
   $\|v'_i - y_i\| > \tau_d$ (≈5% diag); normal angle
   $\angle(n'_i, n_i) > \theta_n$; sliding connections
   $|n_i^\top (v'_i - y_i)| / \|v'_i - y_i\| < s$; masked vertices (hair, eyes,
   mouth interior) excluded.
2. **IRLS weights** from squared norm $s_i = \|r_i\|^2$:
   Huber: $u = 1$ if $s \le \delta^2$ else $\delta/\sqrt{s}$;
   Geman–McClure: $u = 1/(1 + s/c^2)^2$.
3. **Solve** GN (5–20 inner iterations; stop on relative energy decrease
   < 1e-4; warm-start from previous solution; LM schedule as §4).
4. **Anneal** → next level.

**Multi-level schedule** (maps 1:1 to Wrap node parameters):

| Quantity | Schedule |
|---|---|
| Levels L | 4–6 (Wrap `Subdivisions`) |
| Node count | ×2 per level (Wrap `Sampling Initial→Final`) |
| Bindings | rebuilt per level; new-node transforms initialized by sampling the previous level's deformation field (rigid fit at the node) |
| $\lambda_s$ (smoothness) | large → small, $\lambda_s^{(\ell)} = \lambda_0 \gamma^\ell$, γ ≈ 0.3–0.5 (Wrap `Stiffness/Smoothness Initial→Final`) |
| $\lambda_c, \lambda_p$ (data) | small → large (opposite direction) |
| $\lambda_l$ (landmarks) | constant, **high** |
| LM damping | large → small (Wrap `Dp`) |
| correspondence samples | ×2 per level |

Rationale for annealing direction: early strong regularization confines the
solution near the rigid manifold (correspondence errors stay recoverable);
late weak regularization lets the data term capture detail.

## 7. Initialization and pitfalls

- **Rigid init:** Kabsch on landmarks, $H = \sum w\,(p-\bar p)(q-\bar q)^\top$,
  SVD with $\det = +1$ branch; then seed every node with the global rigid
  field: $R_j = R_0$, $t_j = (R_0 - I)g_j + t_0$.
- Compute bindings **once** on the undeformed template per level; never
  re-bind on the deformed shape mid-iteration.
- $\lambda_a = 0$ + unobserved nodes ⇒ rank deficiency; rely on anchor + LM.
- Pure point-to-plane is rotation-blind around $n_i$; always mix in p2p.
- Correspondences are latent variables of the objective, **not forces**:
  re-estimate + filter every outer iteration; the regularizer only prevents
  tearing. Spring-force intuition gives the wrong convergence semantics.
- Numerical-gradient check every residual block (central differences,
  ε ≈ 1e-6, relative error < 1e-4) before trusting the analytic Jacobian.
- Degenerate triangles / duplicate faces in the target break the BVH; clean
  or dedupe first.

## 8. GPU notes

- Embarrassingly parallel per outer iteration: closest-point query (one BVH,
  cuBQL — NVIDIA official, Apache-2.0 — or ToruNiina/lbvh, MIT-style),
  residual/J·v/Jᵀ·v kernels, binding precompute.
- Block assembly: node accumulators via atomicAdd on 6×6 tiles (float or
  double; SoA layout `[R0..R8, t0..t2]` per node); edges one-thread-each.
- Solver: start with cuDSS direct (route 2); move to block-diagonal-PCG
  matrix-free (route 1) if profiling demands. Keep a CPU double-precision
  fallback for numerical arbitration.
- Precision: float is viable (SurfelWarp does it) after unit-scale
  normalization; verify each kernel against the CPU golden reference
  (SurfelWarp's `GradientCheck` / `SanityCheck` pattern).

## 9. Reference implementations and licenses

| Project | What to take from it | License | Use |
|---|---|---|---|
| trimesh `registration.py` (`nricp_sumner`, `nricp_amberg`) | full tet-embedded-deformation NRICP; matrix assembly of every cost block (`_construct_*`) is directly translatable | MIT | **golden reference** for numerics |
| weigao95/surfelwarp | CUDA warp field: residual/Jacobian kernels, matrix-free JtJ (`ApplyJtJMatrixFreeAtomic.cu`), GN + block-diagonal PCG, GradientCheck | BSD-3 | **architecture template** |
| dgrzech/sobfu | alternative CUDA non-rigid backend (active 2026) | BSD-3 | secondary reference |
| mihaibujanca/dynamicfusion | most complete CPU ED-graph pipeline (node insertion/removal) | BSD-3 | engineering reference |
| hwdong/deformation_graph | geodesic-FPS graph construction (no solver, no license) | none | ideas only — do not copy code |
| NVIDIA/cuBQL | CUDA BVH build + closest-point query | Apache-2.0 | GPU queries |
| ToruNiina/lbvh | header-only parallel LBVH | custom permissive | alternative queries |
| Embree / nanoflann | CPU queries (Wrap itself links Embree) | Apache-2.0 | CPU stage |
| libigl `arap.h`, `arap_dof.h` | ARAP solver, subspace handle deformation | MPL-2.0 | CPU regularizer reference |
| MeshMonk (jsnyde0 rewrite) | registration loop structure, symmetric regularization | Apache-2.0 | reference |
| Eigen + CHOLMOD / cuDSS | block-sparse direct solvers | MPL-2.0 / NVIDIA terms | solvers |

Avoid: CGAL mesh-deformation package (GPL), dolphin-li/DynamicFusion
(no license).

## 10. Benchmark protocol and acceptance targets

Test assets (already local): TOSCA high-res pairs with shared triangulation
⇒ exact vertex ground truth, `.test_tmp/nricp_bench/data/hires/`
(cat0→cat3, david0→david1, michael0→michael2, victoria0→victoria2).
Setup mirrors the reference runs: template decimated to ~4k via FPS grouping,
target an **independent** 3.5k grouping (cross-topology input for the solver;
shared indexing used only to score), 25 FPS landmarks with ground-truth
targets, rigid Kabsch+ICP init.

Metrics: mean/p95/max vertex correspondence error (absolute and % of bbox
diagonal), one-sided Chamfer A→B, edge-length stretch mean/p95, normal-flip
rate, wall time.

Reference numbers to match or beat (trimesh, CPU, victoria0→victoria2):

| method | corr_mean | %diag | chamfer A→B | time |
|---|---|---|---|---|
| rigid ICP only | 25.9 | 12.7 | 8.0 | 0.2 s |
| amberg, no landmarks | 19.9 | 9.8 | 1.5 | 10.4 s |
| amberg +25 lm (tuned wl) | 5.1 | 2.5 | 1.2 | 12.0 s |
| sumner(tet) +25 lm | **2.05** | **1.0** | **0.46** | **3.8 s** |

Acceptance: corr_mean ≤ 2.1 (≈1% diag) with 25 landmarks on all four pairs;
stretch p95 ≤ 0.5; zero inverted faces. Performance goals: CPU (single
thread+SIMD) < 1 s/pair; CUDA < 100 ms/pair at ~50k-vertex templates.

## 11. References

- B. Amberg, S. Romdhani, T. Vetter. *Optimal Step Nonrigid ICP Algorithms for
  Surface Registration.* CVPR 2007.
- R. W. Sumner, J. Popović. *Deformation Transfer for Triangle Meshes.*
  SIGGRAPH 2004 (§4 correspondence computation = the registration stage).
- R. W. Sumner, J. Schmid, M. Pauly. *Embedded Deformation for Shape
  Manipulation.* SIGGRAPH 2007 (sampled-node deformation graph, w=(1−d/dmax)²).
- O. Sorkine, M. Alexa. *As-Rigid-As-Possible Surface Modeling.* SGP 2007.
- M. Lourakis. *sparseLM* (+ SuiteSparse CHOLMOD) — the solver stack Wrap
  links (`splm_cholmod.c`).
- Z. Wu et al. *Quasi-Newton Solver for Robust Non-Rigid Registration*
  (SurfelWarp), CVPR 2018.
- Faceform Wrap documentation (Wrapping node) and symbol analysis in
  `../docs/wrap-wrapping-algorithm-research.md`.
