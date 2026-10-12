Closest-point query structures over the target surface: build once per wrap
call (the target never moves), then answer closest-point and interpolated-normal
queries for the moving template vertices. CPU implementation first, then the
accelerated one.
