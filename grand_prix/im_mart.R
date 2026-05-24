box::use(
  ripr / mixture[simplex_mixture],
  S7[new_class, new_object, method, `method<-`, class_numeric, S7_object],
  seqan[
    Test,
    update,
    reset,
    is_stopped,
    decision,
    n_obs,
    stopping_time,
    value
  ]
)

# ---------------------------------------------------------------------------
# PairwiseMartingale — product martingale for candidate 1 vs competitor j
# ---------------------------------------------------------------------------
#
# M_t = ∏ᵢ q(Xᵢ) / θ*(Xᵢ)
#
# θ* is the I-projection of point q onto the pairwise null boundary {θ₁ = θⱼ}:
#   θ*₁ = θ*ⱼ = (q₁ + qⱼ) / 2,   θ*ₖ = qₖ  for k ≠ 1, j
#
# Note: only supports point alternatives q. For mixture alternatives, the
# I-projection requires solving a constrained optimisation (not yet implemented).

#' Pairwise product martingale for candidate 1 vs competitor j
#'
#' M_t = prod_i q(X_i) / theta*(X_i), where theta* is the I-projection of
#' point q onto the pairwise null boundary {theta_1 = theta_j}:
#' theta*_1 = theta*_j = (q_1 + q_j) / 2, theta*_k = q_k for k != 1, j.
#' Supports point alternatives only.
#'
#' @param q Numeric vector of length K summing to 1, with q[1] > q[competitor]
#'   (strictly in H_1).
#' @param competitor Integer. Index j of the challenging candidate, in 2:K.
#' @param alpha Numeric. Significance level. Default: 0.05.
#' @param stream Optional stream object.
#' @return A `PairwiseMartingale` object.
#' @export
PairwiseMartingale <- new_class(
  "PairwiseMartingale",
  parent = Test,
  properties = list(
    q = class_numeric,
    competitor = class_numeric,
    theta = class_numeric,
    alpha = class_numeric
  ),
  constructor = function(q, competitor, alpha = 0.05, stream = NULL) {
    K <- length(q)
    if (K < 2L || any(q <= 0) || abs(sum(q) - 1) > 1e-10) {
      stop("q must be a probability vector of length >= 2 summing to 1")
    }
    j <- as.integer(competitor)
    if (!(j %in% 2:K)) {
      stop(sprintf(
        "competitor must be in 2:%d (index of the challenging candidate)",
        K
      ))
    }
    if (alpha <= 0 || alpha >= 1) {
      stop("alpha must be in (0, 1)")
    }

    theta <- q
    theta[1L] <- (q[1L] + q[j]) / 2
    theta[j] <- (q[1L] + q[j]) / 2

    state <- new.env(parent = emptyenv())
    state$log_m <- 0
    state$history <- 1
    state$n <- 0L
    state$stop_time <- NA_integer_
    state$decision <- "continue"

    new_object(
      S7_object(),
      q = q,
      competitor = j,
      theta = theta,
      alpha = alpha,
      stream = stream,
      description = sprintf("Pairwise RIPr martingale (candidate 1 vs %d)", j),
      state = state
    )
  }
)

method(update, PairwiseMartingale) <- function(stat, new_x = NULL, ...) {
  if (is_stopped(stat)) {
    message("PairwiseMartingale has already stopped. Use reset() to restart.")
  }
  if (is.null(new_x)) {
    if (is.null(stat@stream)) {
      stop("Provide new_x or attach a stream.")
    }
    new_x <- fetch(stat@stream)
  }
  if (length(new_x) == 0) {
    return(invisible(stat))
  }

  log_incr <- log(stat@q) - log(stat@theta)
  threshold <- 1 / stat@alpha
  old_n <- stat@state$n

  new_e <- numeric(length(new_x))
  for (i in seq_along(new_x)) {
    stat@state$log_m <- stat@state$log_m + log_incr[new_x[i]]
    new_e[i] <- exp(stat@state$log_m)
  }

  stat@state$history <- c(stat@state$history, new_e)
  stat@state$n <- old_n + length(new_x)

  if (!is_stopped(stat)) {
    crossed <- which(new_e >= threshold)
    if (length(crossed) > 0L) {
      stat@state$decision <- "reject_H0"
      stat@state$stop_time <- old_n + crossed[1L]
      message("PairwiseMartingale: rejection threshold reached.")
    }
  }

  invisible(stat)
}

method(reset, PairwiseMartingale) <- function(object, ...) {
  object@state$log_m <- 0
  object@state$history <- 1
  object@state$n <- 0L
  object@state$stop_time <- NA_integer_
  object@state$decision <- "continue"
  if (!is.null(object@stream)) {
    reset(object@stream)
  }
  invisible(object)
}

method(is_stopped, PairwiseMartingale) <- function(stat, ...) {
  decision(stat) != "continue"
}
method(decision, PairwiseMartingale) <- function(test, ...) test@state$decision
method(n_obs, PairwiseMartingale) <- function(stat, ...) stat@state$n
method(stopping_time, PairwiseMartingale) <- function(test, ...) {
  test@state$stop_time
}

method(value, PairwiseMartingale) <- function(stat, n = 1L, ...) {
  tail(stat@state$history, n = n)
}

method(print, PairwiseMartingale) <- function(x, ...) {
  cat(sprintf("Pairwise RIPr martingale (candidate 1 vs %d)\n", x@competitor))
  cat("q:    ", x@q, "\n")
  cat("theta:", x@theta, "\n")
  cat("Observations:", x@state$n, "\n")
  if (is_stopped(x)) {
    cat("Stopping time:", x@state$stop_time, "\n")
  }
  cat("Current E_t:", round(exp(x@state$log_m), 4), "\n")
  cat("Rejection threshold:", round(1 / x@alpha, 4), "\n")
  cat("Decision:", decision(x), "\n")
  invisible(x)
}

# ---------------------------------------------------------------------------
# InfimumMartingale — pointwise inf of K-1 mixture pairwise sub-tests
# ---------------------------------------------------------------------------
#
# For a mixture alternative Q = Σ_j w_j δ_{θ_j}, the test martingale against
# competitor k is the weighted sum of per-atom pairwise likelihood ratios:
#
#   M_t^(k) = Σ_j w_j ∏_i θ_j(X_i) / θ*_{j,k}(X_i)
#
# where θ*_{j,k} is the I-projection of atom θ_j onto face k (the pairwise
# null boundary {θ_1 = θ_k}):
#   θ*_{j,k,1} = θ*_{j,k,k} = (θ_j,1 + θ_j,k) / 2,   others unchanged.
#
# M_t^(k) is a valid test martingale for H_0^(k) provided all atoms lie in
# H_1 (i.e. θ_j,1 > θ_j,k for all j), which holds for the simplex-lattice
# Dirichlet mixtures used here. InfimumMartingale = min_k M_t^(k).
#
# For a point alternative (single-atom Q), this reduces exactly to the
# standard pairwise I-projection martingale.

#' Infimum of mixture pairwise martingales for K-candidate plurality
#'
#' For each competitor k, constructs a mixture pairwise martingale
#' M_t^(k) = sum_j w_j prod_i theta_j(X_i) / theta*_{j,k}(X_i), where
#' theta*_{j,k} is the I-projection of atom theta_j onto the pairwise null
#' boundary {theta_1 = theta_k}. The test value is min_k M_t^(k), which is
#' valid against H_0 = union_k H_0^(k). For a single-atom Q this reduces to
#' the standard pairwise I-projection martingale.
#'
#' All atoms of Q must lie strictly in H_1 (Q@atoms[1,j] > Q@atoms[k,j] for
#' all k > 1 and j). This is satisfied by the filtered simplex-lattice
#' Dirichlet mixtures constructed in `_targets.R`.
#'
#' @param Q A `simplex_mixture`. Use [point_mnom()] or [dirichlet_mnom()].
#' @param alpha Numeric. Significance level. Default: 0.05.
#' @param stream Optional stream object.
#' @return An `InfimumMartingale` object.
#' @export
InfimumMartingale <- new_class(
  "InfimumMartingale",
  parent = Test,
  properties = list(
    Q = simplex_mixture,
    alpha = class_numeric
  ),
  constructor = function(Q, alpha = 0.05, stream = NULL) {
    K <- nrow(Q@atoms)
    A <- ncol(Q@atoms)
    if (alpha <= 0 || alpha >= 1) {
      stop("alpha must be in (0, 1)")
    }

    # Precompute log-increment array: log_incr[ki, x, j] =
    #   log θ_j[x] - log θ*_{j,k}[x], where k = ki + 1.
    # Only categories 1 and k differ from 0; the rest cancel.
    log_incr <- array(0, dim = c(K - 1L, K, A))
    for (ki in seq_len(K - 1L)) {
      k <- ki + 1L
      pj_vec <- (Q@atoms[1L, ] + Q@atoms[k, ]) / 2 # length A
      log_incr[ki, 1L, ] <- log(Q@atoms[1L, ]) - log(pj_vec)
      log_incr[ki, k, ] <- ifelse(
        Q@atoms[k, ] > 0,
        log(Q@atoms[k, ]) - log(pj_vec),
        -Inf # atom assigns zero prob to k → impossible after seeing k
      )
    }

    log_w <- log(Q@weights) # length A

    state <- new.env(parent = emptyenv())
    state$log_incr <- log_incr # (K-1) x K x A — constant
    state$log_w <- log_w # length A — constant
    state$log_m <- matrix(0, nrow = K - 1L, ncol = A) # running log LRs
    state$history <- 1
    state$n <- 0L
    state$stop_time <- NA_integer_
    state$decision <- "continue"

    new_object(
      S7_object(),
      Q = Q,
      alpha = alpha,
      stream = stream,
      description = sprintf(
        "Infimum mixture pairwise martingale (%d-candidate plurality, %d atoms)",
        K,
        A
      ),
      state = state
    )
  }
)

method(update, InfimumMartingale) <- function(stat, new_x = NULL, ...) {
  if (is_stopped(stat)) {
    message("InfimumMartingale has already stopped. Use reset() to restart.")
  }
  if (is.null(new_x)) {
    if (is.null(stat@stream)) {
      stop("Provide new_x or attach a stream.")
    }
    new_x <- fetch(stat@stream)
  }
  if (length(new_x) == 0) {
    return(invisible(stat))
  }

  threshold <- 1 / stat@alpha
  old_n <- stat@state$n
  new_e <- numeric(length(new_x))

  for (i in seq_along(new_x)) {
    x <- new_x[i]
    # Update all (competitor, atom) log LRs simultaneously.
    stat@state$log_m <- stat@state$log_m + stat@state$log_incr[, x, ]
    # M_t^(k) = logsumexp(log_w + log_m[ki, ]) for each competitor ki.
    lv_mat <- stat@state$log_w + stat@state$log_m # (K-1) x A, log_w broadcast
    m_vals <- apply(lv_mat, 1L, function(lv) {
      fin <- is.finite(lv)
      if (!any(fin)) {
        return(0)
      }
      m <- max(lv[fin])
      exp(log(sum(exp(lv[fin] - m))) + m)
    })
    new_e[i] <- min(m_vals)
  }

  stat@state$history <- c(stat@state$history, new_e)
  stat@state$n <- old_n + length(new_x)

  if (!is_stopped(stat)) {
    crossed <- which(new_e >= threshold)
    if (length(crossed) > 0L) {
      stat@state$decision <- "reject_H0"
      stat@state$stop_time <- old_n + crossed[1L]
      message("InfimumMartingale: rejection threshold reached.")
    }
  }

  invisible(stat)
}

method(reset, InfimumMartingale) <- function(object, ...) {
  d <- dim(object@state$log_incr)
  object@state$log_m <- matrix(0, nrow = d[1L], ncol = d[3L])
  object@state$history <- 1
  object@state$n <- 0L
  object@state$stop_time <- NA_integer_
  object@state$decision <- "continue"
  if (!is.null(object@stream)) {
    reset(object@stream)
  }
  invisible(object)
}

method(is_stopped, InfimumMartingale) <- function(stat, ...) {
  decision(stat) != "continue"
}
method(decision, InfimumMartingale) <- function(test, ...) test@state$decision
method(n_obs, InfimumMartingale) <- function(stat, ...) stat@state$n
method(stopping_time, InfimumMartingale) <- function(test, ...) {
  test@state$stop_time
}

method(value, InfimumMartingale) <- function(stat, n = 1L, ...) {
  tail(stat@state$history, n = n)
}

method(print, InfimumMartingale) <- function(x, ...) {
  K <- nrow(x@Q@atoms)
  A <- ncol(x@Q@atoms)
  cat(sprintf(
    "Infimum mixture pairwise martingale (%d-candidate plurality, %d atoms)\n",
    K,
    A
  ))
  cat("Observations:", x@state$n, "\n")
  if (is_stopped(x)) {
    cat("Stopping time:", x@state$stop_time, "\n")
  }
  cat("Current E_t^inf:", round(tail(x@state$history, 1L), 4), "\n")
  cat("Rejection threshold:", round(1 / x@alpha, 4), "\n")
  cat("Decision:", decision(x), "\n")
  invisible(x)
}
