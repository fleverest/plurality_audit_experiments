#' Grid of probability vectors along the 2D ordered simplex path
#'
#' Traces a path on the 3-category simplex from (1/2, 1/2, 0) through the
#' uniform point (1/3, 1/3, 1/3) to (1/2, 0, 1/2). All points satisfy the
#' ordering constraint theta_1 >= theta_3, making this grid suitable for
#' testing ordered-parameter hypotheses.
#'
#' @param points Number of grid points. Must be >= 3. Default: 21.
#' @return Named list of length `points`, each element a numeric vector of
#'   length 3 summing to 1.
#' @export
make_2d_simplex_grid <- function(points = 21) {
  if (!is.numeric(points) || length(points) != 1 || points < 3L) {
    stop("`points` must be a single numeric >= 3")
  }
  s1_len <- ceiling((points + 1) / 2)
  s2_len <- floor((points + 1) / 2)
  seq1 <- seq(1 / 2, 1 / 3, length.out = s1_len)
  seq2 <- seq(1 / 3, 1 / 2, length.out = s2_len)
  seq2_tail <- if (length(seq2) > 1L) utils::tail(seq2, -1) else numeric(0)

  c(
    lapply(seq1, function(t) c(t, t, 1 - 2 * t)),
    lapply(seq2_tail, function(t) c(t, 1 - 2 * t, t))
  )
}

#' Grid of integer vectors on the simplex
#' 
#' Generates all non-negative integer vectors of length `d` that sum to `n`.
#' The result is a matrix where each row is such a vector. This can be used to
#' construct a lattice grid over the standard (d-1)-simplex by normalising the
#' rows to sum to 1.
#' @export
simplex_lattice <- function(d, n) {
  if (d == 1L) return(matrix(n, nrow = 1L, ncol = 1L))
  do.call(rbind, lapply(0L:n, function(k) cbind(k, simplex_lattice(d - 1L, n - k))))
}
