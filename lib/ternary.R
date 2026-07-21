# lib/ternary.R
# Utility functions for K=3 ternary / simplex visualisations.

# Standard equilateral triangle embedding:
#   Vertex 1 (top,          b1=1): (0.5, sqrt(3)/2)
#   Vertex 2 (bottom-left,  b2=1): (0, 0)
#   Vertex 3 (bottom-right, b3=1): (1, 0)
#
# Forward:  x = 0.5*b1 + b3,  y = sqrt(3)/2 * b1
# Inverse:  b1 = 2y/sqrt(3),  b3 = x - y/sqrt(3),  b2 = 1 - b1 - b3

#' @export
bary_to_cart <- function(b1, b2, b3) {
  list(x = 0.5 * b1 + b3,
       y = sqrt(3) / 2 * b1)
}

# Returns a data frame with one row per grid cell (res*res rows total), ordered
# so that row k corresponds to full[k] in the to_z_mat convention:
#   t(matrix(full, nrow=res, ncol=res))[row, col] = full[(row-1)*res + col]
# i.e. x (col) iterates fastest (inner), y (row) iterates slowest (outer).
# Columns: b1, b2, b3 (barycentric), valid (logical).
#' @export
simplex_grid <- function(res) {
  xseq <- seq(0, 1,           length.out = res)
  yseq <- seq(0, sqrt(3) / 2, length.out = res)

  # expand.grid iterates its FIRST argument fastest, giving x-inner, y-outer.
  g <- expand.grid(x_idx = seq_len(res), y_idx = seq_len(res))
  x <- xseq[g$x_idx]
  y <- yseq[g$y_idx]

  sqr3 <- sqrt(3)
  b1   <- 2 * y / sqr3
  b3   <- x - y / sqr3
  b2   <- 1 - b1 - b3

  data.frame(b1 = b1, b2 = b2, b3 = b3, valid = b1 >= 0 & b2 >= 0 & b3 >= 0)
}

# H0 / H1 boundary for K=3 plurality (candidate 1 as the reference winner).
# H0 = { theta_2 >= theta_1 } ∪ { theta_3 >= theta_1 }.
# The boundary has two line segments, each running from a mid-edge point to
# the centroid, returned as a two-row data frame with x, y, xend, yend.
#   Segment 1: theta_2 = theta_1  (b1=b2=1/2,b3=0)  →  centroid (1/3,1/3,1/3)
#   Segment 2: theta_3 = theta_1  (b1=b3=1/2,b2=0)  →  centroid
#' @export
h0_boundary_k3 <- function() {
  sqr3 <- sqrt(3)
  data.frame(
    x    = c(0.25, 0.75),
    y    = c(sqr3 / 4, sqr3 / 4),
    xend = c(0.5,  0.5),
    yend = c(sqr3 / 6, sqr3 / 6)
  )
}
