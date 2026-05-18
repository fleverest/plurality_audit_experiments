box::use(
  ripr / mixture[mixture_mnom]
)

#' Log mixture-multinomial mass
#'
#' Computes log sum_j w_j Multinomial(counts; n, theta_j) for a `mixture_mnom`.
#' Zero counts are masked before the log evaluation to avoid 0 * (-Inf) = NaN.
#'
#' @param counts Integer vector of length K — observed category counts.
#' @param mixture A `mixture_mnom`.
#' @return Scalar log-probability.
#' @export
log_mixture_mass <- function(counts, mixture) {
  n <- sum(counts)
  log_base <- lgamma(n + 1L) - sum(lgamma(counts + 1L))
  nz <- counts > 0L
  log_comps <- as.vector(counts[nz] %*% log(mixture@atoms[nz, , drop = FALSE]))
  log_comps_w <- log(mixture@weights) + log_comps
  finite <- is.finite(log_comps_w)
  if (!any(finite)) {
    return(-Inf)
  }
  m <- max(log_comps_w[finite])
  log_base + log(sum(exp(log_comps_w[finite] - m))) + m
}
