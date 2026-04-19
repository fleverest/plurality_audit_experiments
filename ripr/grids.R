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
make_simplex_grid <- function(points = 21) {
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
