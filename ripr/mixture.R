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
  torch[torch_tensor, torch_logsumexp],
  ripr / torch_settings[device, dtype],
  ripr /
    multinomial[mnom_logpmf, make_multinomial_likelihood, log_multinom_coef],
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

as_count_tensor <- function(X) {
  if (inherits(X, "torch_tensor")) {
    return(X)
  }
  X <- if (is.null(dim(X))) matrix(X, nrow = 1L) else as.matrix(X)
  torch_tensor(X, device = device, dtype = dtype)
}

method(log_pmf, discrete_simplex_mixture) <- function(
  simplex_mixture,
  X,
  warn = TRUE
) {
  # Convert to a torch tensor if needed
  X <- as_count_tensor(X)
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
  log_I_num <- vapply(
    seq_along(alpha),
    function(l) {
      gamma <- alpha
      gamma[l] <- gamma[l] + 1
      log_I(gamma)
    },
    numeric(1L)
  )
  exp(log_I_num - log_I_alpha)
}

trunc_dirichlet_mode <- function(alpha) {
  # Unconstrained Dirichlet mode is (alpha - 1) / (sum(alpha) - K), valid
  # when all alpha > 1. If it lies in H_1, return it. Otherwise return NA.
  K <- length(alpha)
  mode_unconstr <- (alpha - 1) / (sum(alpha) - K)
  if (all(alpha > 1) && all(mode_unconstr[1L] > mode_unconstr[-1L])) {
    return(mode_unconstr)
  }
  rep(NA_real_, K)
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
  # Convert to a torch tensor if needed
  X <- as_count_tensor(X)

  n <- X[1, ]$sum()$item()

  alpha <- simplex_mixture@alpha
  log_I_denom <- simplex_mixture@log_I_alpha

  # Multinomial coefficient on the GPU.
  log_multinom_coef <- log_multinom_coef(X, n)

  # Numerator: log_I(alpha + x) per row. Exceedance isn't vectorised,
  # so loop. Move X to CPU once.
  X_cpu <- as.matrix(as.array(X$cpu()))
  N <- nrow(X_cpu)
  log_I_num <- vapply(
    seq_len(N),
    \(i) log_I(alpha + X_cpu[i, ]),
    numeric(1L)
  )
  log_I_num_t <- torch_tensor(log_I_num, device = device, dtype = dtype)

  log_multinom_coef + log_I_num_t - log_I_denom
}

method(n_categories, truncated_dirichlet) <- function(simplex_mixture) {
  length(simplex_mixture@alpha)
}


#' Pairwise-null reverse information projection of a simplex mixture
#'
#' Represents the RIPr P* of the numerator `Q` onto the pairwise plurality
#' null `H_0^{(j)} = {theta_1 <= theta_j}`. Let `pi_j` replace coordinates 1
#' and `j` of theta by their average and fix the rest; for any mixing measure
#' W_1 with `W_1({theta_1 >= theta_j}) = 1` (true for any prior on the
#' plurality alternative H_1; not checked here), the RIPr of `Q = P_{W_1}`
#' onto `H_0^{(j)}` is the pushforward `P* = P_{pi_j # W_1}`:
#' `E_phi[Q / P*] = 1` on the tie facet `{theta_1 = theta_j}` and `< 1`
#' strictly inside the null, which is the variational characterisation of
#' the projection over the convex null.
#'
#' The pmf needs no new integrals for any Q. Factoring the multinomial
#' through the sufficient split `N = x_1 + x_j` and expanding
#' `(theta_1 + theta_j)^N` binomially gives the finite-sum identity
#'
#'   P*(x) = Bin(x_1 | N, 1/2) * sum_{m=0}^{N} Q(x with (x_1, x_j) -> (m, N-m)),
#'
#' i.e. P* keeps Q's marginal over `(N, x_{-1,-j})` and freezes the split
#' conditional at `Binomial(N, 1/2)`. `log_pmf` implements this via one
#' stacked `log_pmf(Q, .)` call plus a grouped logsumexp, so it works
#' uniformly for point masses, discrete mixtures, and [truncated_dirichlet()]
#' numerators.
#'
#' @slot Q The numerator `simplex_mixture` being projected.
#' @slot j Integer in `2:K`. The competing candidate defining the null.
#' @slot n Optional integer multinomial sample size (informational, as in
#'   [discrete_simplex_mixture()]).
#' @export
pairwise_projection <- new_class(
  "pairwise_projection",
  parent = simplex_mixture,
  properties = list(
    Q = simplex_mixture,
    j = class_numeric,
    n = new_property(default = NULL)
  ),
  constructor = function(Q, j, n = NULL) {
    K <- n_categories(Q)
    j <- as.integer(j)
    if (length(j) != 1L || j < 2L || j > K) {
      stop("`j` must be a single integer in 2:K")
    }
    # pi_j is linear, so the pushforward mean is pi_j applied to Q's mean.
    mean_val <- Q@mean
    mean_val[c(1L, j)] <- (mean_val[1L] + mean_val[j]) / 2
    new_object(
      S7_object(),
      Q = Q,
      j = j,
      n = n,
      mean = mean_val,
      mode = rep(NA_real_, K)
    )
  }
)

logsumexp_num <- function(v) {
  m <- max(v)
  if (!is.finite(m)) {
    return(m)
  }
  m + log(sum(exp(v - m)))
}

method(log_pmf, pairwise_projection) <- function(simplex_mixture, X) {
  Q <- simplex_mixture@Q
  j <- simplex_mixture@j
  X_cpu <- if (inherits(X, "torch_tensor")) {
    as.matrix(as.array(X$cpu()))
  } else if (is.null(dim(X))) {
    matrix(X, nrow = 1L)
  } else {
    as.matrix(X)
  }

  x1 <- X_cpu[, 1L]
  N <- x1 + X_cpu[, j]
  merged <- X_cpu
  merged[, 1L] <- N
  merged[, j] <- 0
  key <- apply(merged, 1L, paste, collapse = "_")
  grp <- match(key, key) # index of first row in each (N, x_-) group

  # One stacked Q evaluation over every split (m, N - m) of every unique group.
  rep_idx <- which(!duplicated(key))
  split_rows <- lapply(rep_idx, function(i) {
    m <- 0:N[i]
    rows <- matrix(
      X_cpu[i, ],
      nrow = N[i] + 1L,
      ncol = ncol(X_cpu),
      byrow = TRUE
    )
    rows[, 1L] <- m
    rows[, j] <- N[i] - m
    rows
  })
  stacked <- do.call(rbind, split_rows)
  stacked_grp <- rep(rep_idx, times = N[rep_idx] + 1L)
  log_q_stacked <- as.numeric(log_pmf(Q, stacked)$cpu())

  # log of Q's marginal over the merged outcome (N, x_-), per group.
  log_marg <- vapply(
    split(log_q_stacked, stacked_grp),
    logsumexp_num,
    numeric(1L)
  )[as.character(grp)]

  result <- log_marg + lchoose(N, x1) - N * log(2)
  torch_tensor(result, device = device, dtype = dtype)
}

method(n_categories, pairwise_projection) <- function(simplex_mixture) {
  n_categories(simplex_mixture@Q)
}


#' Compute the expected likelihood ratio of two mixtures under a multinomial DGP
#' with sample size n, i.e. E_{X ~ Multinomial(n, theta)}[P(X|Q) / P(X|P)].
#' @param theta Numeric vector of length K or matrix with dimension (K, .) with
#' the multinomial parameter(s) theta.
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
