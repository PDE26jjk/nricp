Residual blocks of the wrapping energy (point-to-point, point-to-plane,
graph smoothness, anchor, landmarks) together with their analytic Jacobians and
the per-block energy values that the run report exposes (`energy.{h,cpp}`), plus
the IRLS robust kernels that reweight them (`robust.{h,cpp}`). Every route
solves this same energy; nothing route-specific belongs here.
