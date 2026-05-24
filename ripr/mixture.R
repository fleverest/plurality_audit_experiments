box::use(
  S7[
    new_object,
    new_class,
    new_generic,
    new_property,
    method,
    class_numeric,
    `method<-`,
    S7_object
  ],
  torch[torch_tensor, torch_logsumexp, torch_lgamma],
  ripr / torch_settings[device, dtype],
  ripr / multinomial[mnom_logpmf, make_multinomial_likelihood],
  lib / dirichlet_exceedance[dirichlet_exceedance]
)


#' Distribution on the (K-1)-simplex
#'
#' Abstract base class. Concrete subclasses provide a distribution on
#' `Delta^{K-1}`. Each concrete subclass defines methods for the generics
#' [log_pmf()], [n_categories()].
#'
#' `log_pmf` is the multinomial-mixture PMF induced by the simplex
#' mixture W, i.e. \int_{\Delta^{K-1}} Multinomial(theta, n) dW(theta). The
#' simplex mixture W itself is a measure on Delta^{K-1}, not on the multinomial
#' outcomes.
#' @slot mean Numeric vector of length K with the mean of the distribution.
#' @slot mode Numeric vector of length K with the mode of the distribution, or
#' NAs if the mode is not defined or not implemented.
#' @export
simplex_mixture <- new_class(
  "simplex_mixture",
  properties = list(
    mean = class_numeric,
    mode = class_numeric
  ),
  abstract = TRUE
)


#' Log multinomial-mixture PMF for outcomes X of size n
#'
#' Computes `log E_theta[Multinomial(X | theta, n)]` where the expectation
#' is taken over the simplex distribution.
#'
#' @param simplex_mixture A [simplex_mixture] subclass instance.
#' @param X Tensor of shape (N, K) of count vectors; n inferred from row sums.
#' @return Tensor of shape (N,) with log P(x) for each row x of X.
#' @export
log_pmf <- new_generic("log_pmf", "simplex_mixture")

#' Number of simplex categories
#' @export
n_categories <- new_generic("n_categories", "simplex_mixture")

#' Discrete mixture on the simplex
#'
#' Represents mixtures of the form `sum_j w_j * delta_{theta_j}`. The induced
#' multinomial-mixture PMF is then `sum_j w_j * Multinomial(. | theta_j, n)`.
#'
#' @slot atoms Numeric matrix of shape (K, M) -- columns are probability vectors.
#' @slot weights Numeric vector of length M summing to 1.
#' @slot n Optional integer multinomial sample size. Can be set when the
#' distribution represents the RIPr distribution, which is defined relative
#' to a particular n, then calling `log_pmf()` will check that the n inferred
#' from X matches this n and warn if not.
#' @export
discrete_simplex_mixture <- new_class(
  "discrete_simplex_mixture",
  parent = simplex_mixture,
  properties = list(
    atoms = class_numeric,
    weights = class_numeric,
    n = new_property(default = NULL)
  ),
  constructor = function(atoms, weights, n = NULL) {
    if (!is.matrix(atoms)) {
      stop("`atoms` must be a matrix")
    }
    if (length(weights) == 0L) {
      stop("`weights` must be non-empty")
    }
    if (ncol(atoms) != length(weights)) {
      stop("`weights` must have one entry per column of `atoms`")
    }
    mean_val <- as.vector(atoms %*% weights)
    mode_val <- atoms[, which.max(weights)]
    new_object(
      S7_object(),
      atoms = atoms,
      weights = weights,
      n = n,
      mean = mean_val,
      mode = mode_val
    )
  },
  validator = function(self) {
    if (!is.matrix(self@atoms)) {
      return("`atoms` must be a matrix")
    }
    if (
      any(self@atoms < 0) ||
        any(self@atoms > 1) ||
        any(abs(colSums(self@atoms) - 1) > 1e-9)
    ) {
      return(
        "`atoms` must represent valid probability vectors (columns must be in [0,1] and sum to 1)"
      )
    }
    if (ncol(self@atoms) != length(self@weights)) {
      return("`weights` must have one entry per column of `atoms`")
    }
    if (abs(sum(self@weights) - 1) > 1e-9) {
      return("`weights` must sum to 1")
    }
    if (
      !is.null(self@n) &&
        (!is.numeric(self@n) || length(self@n) != 1L || self@n < 1)
    ) {
      return("`n` must be a positive scalar or NULL")
    }
    NULL
  }
)

method(log_pmf, discrete_simplex_mixture) <- function(
  simplex_mixture,
  X,
  warn = TRUE
) {
  n <- X[1, ]$sum()$item()
  log_atoms_t <- torch_tensor(
    log(simplex_mixture@atoms),
    device = device,
    dtype = dtype
  )
  log_wts <- torch_tensor(
    log(simplex_mixture@weights),
    device = device,
    dtype = dtype
  )
  # (N, M) + (M,) broadcast -> logsumexp over M -> (N,)
  torch_logsumexp(mnom_logpmf(X, log_atoms_t, n) + log_wts, dim = 2L)
}

method(n_categories, discrete_simplex_mixture) <- function(simplex_mixture) {
  nrow(simplex_mixture@atoms)
}

# Log of the unnormalised integral
# I(gamma) = int_{H_1} prod_i theta_i^{gamma_i - 1} d theta.
log_I <- function(gamma) {
  sum(lgamma(gamma)) -
    lgamma(sum(gamma)) +
    log(dirichlet_exceedance(gamma, j = 1))
}

trunc_dirichlet_mean <- function(alpha, log_I_alpha) {
  # E[theta_l] = I(alpha + e_l) / I(alpha) via the same exceedance trick.
  # K extra exceedance calls.
  K <- length(alpha)
  log_I_denom <- log_I_alpha
  result <- numeric(K)
  for (l in seq_len(K)) {
    gamma <- alpha
    gamma[l] <- gamma[l] + 1
    result[l] <- exp(log_I(gamma) - log_I_denom)
  }
  result
}

trunc_dirichlet_mode <- function(alpha) {
  # Unconstrained Dirichlet mode is (alpha - 1) / (sum(alpha) - K), valid
  # when all alpha > 1. If it lies in H_1, return it. Otherwise return NA.
  K <- length(alpha)
  mode_unconstr <- (alpha - 1) / (sum(alpha) - K)
  if (all(alpha > 1) && all(mode_unconstr[1] > mode_unconstr[-1])) {
    return(mode_unconstr)
  } else {
    rep(NA_real_, K)
  }
}

#' Dirichlet on H_1 (for plurality)
#'
#' Represents `Dirichlet(alpha)` truncated to the open polytope
#' `H_1 = {theta in Delta^{K-1} : theta_1 > theta_k for all k != 1}`.
#' The induced multinomial-mixture PMF is computed analytically up to a 1D
#' numerical integral (the Dirichlet exceedance probability), via:
#'
#'   log P(x) = log multinom_coef(n, x) + log I(alpha + x) - log I(alpha)
#'
#' where `I(gamma) = int_{H_1} prod theta_i^{gamma_i - 1} dtheta`.
#'
#' The log of the truncation normaliser `log I(alpha)` is cached at
#' construction since it doesn't depend on x.
#'
#' @slot alpha Numeric vector of Dirichlet concentration parameters.
#' @slot log_I_alpha Cached log-normaliser. Computed at construction.
#' @export
truncated_dirichlet <- new_class(
  "truncated_dirichlet",
  parent = simplex_mixture,
  properties = list(
    alpha = class_numeric,
    log_I_alpha = class_numeric # cached at construction
  ),
  constructor = function(alpha) {
    if (any(alpha <= 0)) {
      stop("`alpha` must be strictly positive")
    }
    log_I_alpha <- log_I(alpha)
    mean_val <- trunc_dirichlet_mean(alpha, log_I_alpha)
    mode_val <- trunc_dirichlet_mode(alpha)
    new_object(
      S7_object(),
      alpha = alpha,
      log_I_alpha = log_I_alpha,
      mean = mean_val,
      mode = mode_val
    )
  },
  validator = function(self) {
    if (any(self@alpha <= 0)) {
      return("`alpha` must be strictly positive")
    }
    NULL
  }
)

method(log_pmf, truncated_dirichlet) <- function(
  simplex_mixture,
  X
) {
  n <- X[1, ]$sum()$item()

  alpha <- simplex_mixture@alpha
  log_I_denom <- simplex_mixture@log_I_alpha

  # Multinomial coefficient on the GPU.
  log_multinom_coef <- torch_lgamma(torch_tensor(
    n + 1,
    device = device,
    dtype = dtype
  )) -
    torch_lgamma(X + 1)$sum(dim = 2L)

  # Numerator: log_I(alpha + x) per row. Exceedance isn't vectorised,
  # so loop. Move X to CPU once.
  X_cpu <- as.matrix(as.array(X$cpu()))
  N <- nrow(X_cpu)
  log_I_num <- numeric(N)
  for (i in seq_len(N)) {
    log_I_num[i] <- log_I(alpha + X_cpu[i, ])
  }
  log_I_num_t <- torch_tensor(log_I_num, device = device, dtype = dtype)

  log_multinom_coef + log_I_num_t - log_I_denom
}

method(n_categories, truncated_dirichlet) <- function(simplex_mixture) {
  length(simplex_mixture@alpha)
}


#' Compute the expected likelihood ratio of two mixtures under a multinomial DGP
#' with sample size n, i.e. E_{X ~ Multinomial(n, theta)}[P(X|Q) / P(X|P)].
#' @param theta Numeric vector of length K with the multinomial parameter theta.
#' @param n Integer sample size of the multinomial DGP.
#' @param Q A `simplex_mixture` representing the alternative distribution Q.
#' @param P A `simplex_mixture` representing the null distribution P.
#' @return Numeric. The expected likelihood ratio E_{X ~ Multinomial(n, theta)}[P(X|Q) / P(X|P)].
#' @export
expected_likelihood_ratio <- function(theta, n, Q, P) {
  K <- n_categories(P)
  if (n_categories(Q) != K) {
    stop("P and Q must have the same number of categories")
  }
  likelihood <- make_multinomial_likelihood(n, K)
  log_pmf_Q <- log_pmf(Q, likelihood$support_tensor)
  log_pmf_P <- log_pmf(P, likelihood$support_tensor)
  log_diff <- log_pmf_Q - log_pmf_P # length-M tensor

  if (is.matrix(theta)) {
    # log_pmf_batch returns (M, N) — log multinomial PMF for each theta column
    log_pmf_theta <- likelihood$log_pmf_batch(theta)
    # Broadcast log_diff over the N theta columns
    log_terms <- (log_pmf_theta + log_diff$unsqueeze(2L))$nan_to_num(nan = -Inf)
    as.numeric(torch_logsumexp(log_terms, dim = 1L)$exp()$cpu())
  } else {
    log_pmf_theta <- likelihood$log_pmf(theta)
    log_terms <- (log_pmf_theta + log_diff)$nan_to_num(nan = -Inf)
    torch_logsumexp(log_terms, dim = 1L)$exp()$item()
  }
}
