# The following code is based on the original MATLAB code from the paper
# "Exceedance Probabilities for the Dirichlet Distribution" by Joram Scotch
# and Carsten Allefeld.
# It computes exceedance probabilities for Dirichlet random variables, i.e.
# the probability that a given category has the largest proportion.

# Integrand function for numerical integration
integrand <- function(x, aj, ak, log = FALSE) {
  y <- sum(stats::pgamma(x, ak, log.p = TRUE)) +
    (aj - 1) * log(x) -
    x -
    lgamma(aj)
  if (log) {
    return(y)
  } else {
    return(exp(y))
  }
}

#' @title Dirichlet Exceedance Probabilities
#' @description
#' The probability that, given a random variable P drawn from Dirichlet(alpha),
#' the j-th category has the largest proportion, i.e. P_j > P_i for all i != j.
#' @param alpha vector of positive reals for the Dirichlet parameters.
#' @param j the index of the category of interest, or NULL to return the full
#'          vector of exceedance probabilities, one for each category.
#' @export
dirichlet_exceedance <- function(alpha, j = NULL) {
  k <- length(alpha)
  exc_p <- numeric(k)

  if (is.null(j)) {
    j <- seq_len(k)
  }

  # Analytical computation, if bivariate Dirichlet
  if (k == 2) {
    # Using the Beta CDF
    exc_p[1] <- 1 - stats::pbeta(0.5, alpha[1], alpha[2])
    exc_p[2] <- 1 - exc_p[1]
    return(exc_p[j])
  }

  # Numerical integration, if multivariate Dirichlet
  if (k > 2) {
    # Integrate over Gamma CDFs
    sapply(
      j,
      \(cat) {
        # Estimate the mode of the category's marginal distribution
        # to set useful integration limits.
        mode <- stats::optimise(
          integrand,
          aj = alpha[cat],
          ak = alpha[-cat],
          log = TRUE,
          interval = c(0, max(alpha)),
          maximum = TRUE
        )$maximum
        cubature::hcubature(
          integrand,
          lower = 0,
          upper = mode,
          aj = alpha[cat],
          ak = alpha[-cat],
          log = FALSE
        )$integral +
          cubature::hcubature(
            integrand,
            lower = mode,
            upper = Inf,
            aj = alpha[cat],
            ak = alpha[-cat],
            log = FALSE
          )$integral
      }
    )
  }
}
