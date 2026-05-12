library(S7)
library(seqan)
box::use(ripr / boundary_iterative[run_boundary_ripr])

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

# Restricted-MLE log-likelihood for K-candidate plurality.
#
# Returns max_{θ ∈ H_0} Σ_k c_k log θ_k  where H_0 = {θ_1 ≤ max(θ_2,...,θ_K)}.
#
# Two cases:
#   A) Empirical c/t is already in H_0: unrestricted MLE equals c/t.
#   B) c_1 > max(c_2,...,c_K): RMLE is on the null boundary. For each competitor
#      j, the MLE on face j (θ_1=θ_j) has θ_1=θ_j=(c_1+c_j)/(2t), θ_k=c_k/t.
#      We take the max log-likelihood over all competitors.
.rmle_log_lik <- function(counts) {
  tot <- sum(counts)
  if (tot == 0L) return(0)
  K <- length(counts)
  safe_loglik <- function(theta) {
    sum(ifelse(counts == 0L | theta <= 0, 0, counts * log(theta)))
  }

  if (counts[1L] <= max(counts[-1L])) {
    safe_loglik(counts / tot)
  } else {
    max(vapply(2:K, function(j) {
      theta     <- counts / tot
      pj        <- (counts[1L] + counts[j]) / (2 * tot)
      theta[1L] <- pj
      theta[j]  <- pj
      safe_loglik(theta)
    }, numeric(1L)))
  }
}

# ---------------------------------------------------------------------------
# UITest — Universal Inference e-process (K-candidate plurality)
# ---------------------------------------------------------------------------
#
# E_t^UI = q̄(X^t) / sup_{θ ∈ H_0} p_θ(X^t)
#
# With the oracle numerator q̄(X^t) = ∏ q(X_i) and denominator equal to
# the restricted-MLE likelihood over the joint sample.
#
# See the review paper for reference: Ramdas et al., "Game-Theoretic Statistics and Safe
# Anytime-Valid Inference".

UITest <- new_class(
  "UITest",
  parent = Test,
  properties = list(
    q     = class_numeric,
    alpha = class_numeric
  ),
  constructor = function(q, alpha = 0.05, stream = NULL) {
    K <- length(q)
    if (K < 2L || any(q <= 0) || abs(sum(q) - 1) > 1e-10)
      stop("q must be a probability vector of length >= 2 summing to 1")
    if (q[1L] <= max(q[-1L]))
      stop("q[1] must exceed max(q[-1]): candidate 1 is the announced winner")
    if (alpha <= 0 || alpha >= 1)
      stop("alpha must be in (0, 1)")

    state <- new.env(parent = emptyenv())
    state$counts   <- integer(K)
    state$log_num  <- 0
    state$history  <- 1
    state$n        <- 0L
    state$stop_time <- NA_integer_
    state$decision <- "continue"

    new_object(
      S7_object(),
      q           = q,
      alpha       = alpha,
      stream      = stream,
      description = sprintf("Universal Inference e-process (%d-candidate plurality)", K),
      state       = state
    )
  }
)

method(update, UITest) <- function(stat, new_x = NULL, ...) {
  if (is_stopped(stat)) {
    message("UITest has already stopped. Use reset() to restart.")
  }
  if (is.null(new_x)) {
    if (is.null(stat@stream)) stop("Provide new_x or attach a stream.")
    new_x <- fetch(stat@stream)
  }
  if (length(new_x) == 0) return(invisible(stat))

  threshold <- 1 / stat@alpha
  old_n     <- stat@state$n
  log_q     <- log(stat@q)

  new_e <- numeric(length(new_x))
  for (i in seq_along(new_x)) {
    x <- new_x[i]
    stat@state$counts[x]  <- stat@state$counts[x] + 1L
    stat@state$log_num    <- stat@state$log_num + log_q[x]
    log_denom             <- .rmle_log_lik(stat@state$counts)
    new_e[i]              <- exp(stat@state$log_num - log_denom)
  }

  stat@state$history  <- c(stat@state$history, new_e)
  stat@state$n        <- old_n + length(new_x)

  if (!is_stopped(stat)) {
    crossed <- which(new_e >= threshold)
    if (length(crossed) > 0L) {
      stat@state$decision  <- "reject_H0"
      stat@state$stop_time <- old_n + crossed[1L]
      message("UITest: rejection threshold reached.")
    }
  }

  invisible(stat)
}

method(reset, UITest) <- function(object, ...) {
  object@state$counts    <- integer(length(object@q))
  object@state$log_num   <- 0
  object@state$history   <- 1
  object@state$n         <- 0L
  object@state$stop_time <- NA_integer_
  object@state$decision  <- "continue"
  if (!is.null(object@stream)) reset(object@stream)
  invisible(object)
}

method(is_stopped, UITest)    <- function(stat, ...) decision(stat) != "continue"
method(decision, UITest)      <- function(test, ...) test@state$decision
method(n_obs, UITest)         <- function(stat, ...) stat@state$n
method(stopping_time, UITest) <- function(test, ...) test@state$stop_time

method(value, UITest) <- function(stat, n = 1L, ...) {
  tail(stat@state$history, n = n)
}

method(print, UITest) <- function(x, ...) {
  K <- length(x@q)
  cat(sprintf("Universal Inference e-process (%d-candidate plurality)\n", K))
  cat("q:", x@q, "\n")
  cat("Observations:", x@state$n, "\n")
  cat("Counts:", x@state$counts, "\n")
  if (is_stopped(x)) cat("Stopping time:", x@state$stop_time, "\n")
  cat("Current E_t:", round(tail(x@state$history, 1L), 4), "\n")
  cat("Rejection threshold:", round(1 / x@alpha, 4), "\n")
  cat("Decision:", decision(x), "\n")
  invisible(x)
}

# ---------------------------------------------------------------------------
# PairwiseMartingale — product martingale for candidate 1 vs competitor j
# ---------------------------------------------------------------------------
#
# M_t = ∏ᵢ q(Xᵢ) / θ*(Xᵢ)
#
# θ* is the I-projection of Q onto the pairwise null boundary {θ₁ = θⱼ}:
#   θ*₁ = θ*ⱼ = (q₁ + qⱼ) / 2,   θ*ₖ = qₖ  for k ≠ 1, j
#
# This ensures E_θ[q(X)/θ*(X)] ≤ 1 for all θ with θ₁ ≤ θⱼ, making M_t a
# valid test martingale for H₀^(j): candidate 1 does not beat candidate j.

PairwiseMartingale <- new_class(
  "PairwiseMartingale",
  parent = Test,
  properties = list(
    q           = class_numeric,
    competitor  = class_numeric,
    theta       = class_numeric,
    alpha       = class_numeric
  ),
  constructor = function(q, competitor, alpha = 0.05, stream = NULL) {
    K <- length(q)
    if (K < 2L || any(q <= 0) || abs(sum(q) - 1) > 1e-10)
      stop("q must be a probability vector of length >= 2 summing to 1")
    j <- as.integer(competitor)
    if (!(j %in% 2:K))
      stop(sprintf("competitor must be in 2:%d (index of the challenging candidate)", K))
    if (alpha <= 0 || alpha >= 1)
      stop("alpha must be in (0, 1)")

    theta <- q
    theta[1L] <- (q[1L] + q[j]) / 2
    theta[j]  <- (q[1L] + q[j]) / 2

    state <- new.env(parent = emptyenv())
    state$log_m    <- 0
    state$history  <- 1
    state$n        <- 0L
    state$stop_time <- NA_integer_
    state$decision <- "continue"

    new_object(
      S7_object(),
      q           = q,
      competitor  = j,
      theta       = theta,
      alpha       = alpha,
      stream      = stream,
      description = sprintf("Pairwise RIPr martingale (candidate 1 vs %d)", j),
      state       = state
    )
  }
)

method(update, PairwiseMartingale) <- function(stat, new_x = NULL, ...) {
  if (is_stopped(stat)) {
    message("PairwiseMartingale has already stopped. Use reset() to restart.")
  }
  if (is.null(new_x)) {
    if (is.null(stat@stream)) stop("Provide new_x or attach a stream.")
    new_x <- fetch(stat@stream)
  }
  if (length(new_x) == 0) return(invisible(stat))

  log_incr  <- log(stat@q) - log(stat@theta)
  threshold <- 1 / stat@alpha
  old_n     <- stat@state$n

  new_e <- numeric(length(new_x))
  for (i in seq_along(new_x)) {
    stat@state$log_m <- stat@state$log_m + log_incr[new_x[i]]
    new_e[i]         <- exp(stat@state$log_m)
  }

  stat@state$history  <- c(stat@state$history, new_e)
  stat@state$n        <- old_n + length(new_x)

  if (!is_stopped(stat)) {
    crossed <- which(new_e >= threshold)
    if (length(crossed) > 0L) {
      stat@state$decision  <- "reject_H0"
      stat@state$stop_time <- old_n + crossed[1L]
      message("PairwiseMartingale: rejection threshold reached.")
    }
  }

  invisible(stat)
}

method(reset, PairwiseMartingale) <- function(object, ...) {
  object@state$log_m    <- 0
  object@state$history  <- 1
  object@state$n        <- 0L
  object@state$stop_time <- NA_integer_
  object@state$decision <- "continue"
  if (!is.null(object@stream)) reset(object@stream)
  invisible(object)
}

method(is_stopped, PairwiseMartingale)    <- function(stat, ...) decision(stat) != "continue"
method(decision, PairwiseMartingale)      <- function(test, ...) test@state$decision
method(n_obs, PairwiseMartingale)         <- function(stat, ...) stat@state$n
method(stopping_time, PairwiseMartingale) <- function(test, ...) test@state$stop_time

method(value, PairwiseMartingale) <- function(stat, n = 1L, ...) {
  tail(stat@state$history, n = n)
}

method(print, PairwiseMartingale) <- function(x, ...) {
  cat(sprintf("Pairwise RIPr martingale (candidate 1 vs %d)\n", x@competitor))
  cat("q:    ", x@q, "\n")
  cat("theta:", x@theta, "\n")
  cat("Observations:", x@state$n, "\n")
  if (is_stopped(x)) cat("Stopping time:", x@state$stop_time, "\n")
  cat("Current E_t:", round(exp(x@state$log_m), 4), "\n")
  cat("Rejection threshold:", round(1 / x@alpha, 4), "\n")
  cat("Decision:", decision(x), "\n")
  invisible(x)
}

# ---------------------------------------------------------------------------
# InfimumMartingale — pointwise inf of K-1 pairwise sub-tests
# ---------------------------------------------------------------------------
#
# Constructs K-1 PairwiseMartingale sub-tests (candidate 1 vs each competitor
# j = 2,...,K) and tracks E_t^inf = min(E_t^(2), ..., E_t^(K)).
#
# Valid for the composite null H₀ = ∪_j {θ₁ ≤ θⱼ}: for any θ ∈ H₀ at least
# one sub-test has E_θ[E_t^(j)] ≤ 1, so E_θ[E_t^inf] ≤ 1.

InfimumMartingale <- new_class(
  "InfimumMartingale",
  parent = Test,
  properties = list(
    q     = class_numeric,
    alpha = class_numeric
  ),
  constructor = function(q, alpha = 0.05, stream = NULL) {
    K <- length(q)
    if (K < 2L || any(q <= 0) || abs(sum(q) - 1) > 1e-10)
      stop("q must be a probability vector of length >= 2 summing to 1")
    if (q[1L] <= max(q[-1L]))
      stop("q[1] must exceed max(q[-1]): candidate 1 is the announced winner")
    if (alpha <= 0 || alpha >= 1)
      stop("alpha must be in (0, 1)")

    state <- new.env(parent = emptyenv())
    state$tests    <- lapply(2:K, function(j) PairwiseMartingale(q, j, alpha))
    state$history  <- 1
    state$n        <- 0L
    state$stop_time <- NA_integer_
    state$decision <- "continue"

    new_object(
      S7_object(),
      q           = q,
      alpha       = alpha,
      stream      = stream,
      description = sprintf("Infimum (pairwise RIPr) test martingale (%d-candidate plurality)", K),
      state       = state
    )
  }
)

method(update, InfimumMartingale) <- function(stat, new_x = NULL, ...) {
  if (is_stopped(stat)) {
    message("InfimumMartingale has already stopped. Use reset() to restart.")
  }
  if (is.null(new_x)) {
    if (is.null(stat@stream)) stop("Provide new_x or attach a stream.")
    new_x <- fetch(stat@stream)
  }
  if (length(new_x) == 0) return(invisible(stat))

  threshold <- 1 / stat@alpha
  old_n     <- stat@state$n
  new_e     <- numeric(length(new_x))

  for (i in seq_along(new_x)) {
    xi <- new_x[i]
    for (tst in stat@state$tests) update(tst, new_x = xi)
    new_e[i] <- min(vapply(stat@state$tests, value, numeric(1L)))
  }

  stat@state$history  <- c(stat@state$history, new_e)
  stat@state$n        <- old_n + length(new_x)

  if (!is_stopped(stat)) {
    crossed <- which(new_e >= threshold)
    if (length(crossed) > 0L) {
      stat@state$decision  <- "reject_H0"
      stat@state$stop_time <- old_n + crossed[1L]
      message("InfimumMartingale: rejection threshold reached.")
    }
  }

  invisible(stat)
}

method(reset, InfimumMartingale) <- function(object, ...) {
  lapply(object@state$tests, reset)
  object@state$history   <- 1
  object@state$n         <- 0L
  object@state$stop_time <- NA_integer_
  object@state$decision  <- "continue"
  if (!is.null(object@stream)) reset(object@stream)
  invisible(object)
}

method(is_stopped, InfimumMartingale)    <- function(stat, ...) decision(stat) != "continue"
method(decision, InfimumMartingale)      <- function(test, ...) test@state$decision
method(n_obs, InfimumMartingale)         <- function(stat, ...) stat@state$n
method(stopping_time, InfimumMartingale) <- function(test, ...) test@state$stop_time

method(value, InfimumMartingale) <- function(stat, n = 1L, ...) {
  tail(stat@state$history, n = n)
}

method(print, InfimumMartingale) <- function(x, ...) {
  K <- length(x@q)
  cat(sprintf("Infimum (pairwise RIPr) test martingale (%d-candidate plurality)\n", K))
  cat("q:", x@q, "\n")
  for (tst in x@state$tests) {
    cat(sprintf("  theta_%d: ", tst@competitor), tst@theta, "\n")
  }
  cat("Observations:", x@state$n, "\n")
  if (is_stopped(x)) cat("Stopping time:", x@state$stop_time, "\n")
  e_subs <- vapply(x@state$tests, \(t) value(t), numeric(1L))
  cat("Current E_t^inf:", round(tail(x@state$history, 1L), 4), "\n")
  cat("Sub-test E_t:   ", round(e_subs, 4), "\n")
  cat("Rejection threshold:", round(1 / x@alpha, 4), "\n")
  cat("Decision:", decision(x), "\n")
  invisible(x)
}

# ---------------------------------------------------------------------------
# DirectRIPrSequenceTest — fixed-t RIPr e-variables corrected to a
# supermartingale (K-candidate plurality)
# ---------------------------------------------------------------------------
#
# At time t with cumulative counts c_t:
#   Y_t = Mult(t, q)(c_t) / P_w^(t)(c_t)
# where P_w^(t) is the boundary-RIPr optimal mixture from run_boundary_ripr(n=t).
#
# The PDF construction gives the supermartingale:
#   Z_t = a_t * Y_t,   a_0 = 1,
#   log a_t = log a_{t-1} + log Y_{t-1} − log sup_{θ on boundary} E_θ[Y_t | F_{t-1}]
#
# Since E_θ[Y_t | c_{t-1}] = Σ_k θ_k * Y_t(c_{t-1}+e_k) is linear in θ, the sup
# over the null boundary is attained at one of its vertices. The null boundary for
# K candidates has two types of vertices for each face j (θ_1=θ_j):
#   - Pair vertex:   1/2 at positions {1,j}, 0 elsewhere
#   - Triple vertex: 1/3 at positions {1,j,r} for each r ∉ {1,j}, 0 elsewhere
# The sup is the max E over all such vertices across all faces.
#
# RIPr results are cached in results_dir as boundary_ripr_n{n:04d}.rds.

# log Y_t(counts) given a boundary-RIPr result and log(q).
.log_Y <- function(counts, ripr, log_q) {
  n        <- sum(counts)
  log_base <- lgamma(n + 1L) - sum(lgamma(counts + 1L))
  log_q_mass <- log_base + sum(counts * log_q)
  log_comps  <- vapply(ripr$atoms, function(theta) {
    log_base + sum(counts * ifelse(theta > 0, log(theta), -Inf))
  }, numeric(1L))
  log_comps_w <- log_comps + log(ripr$weights)
  m           <- max(log_comps_w[is.finite(log_comps_w)])
  log_Pw      <- log(sum(exp(log_comps_w - m))) + m
  log_q_mass - log_Pw
}

# sup_{θ on null boundary} E_θ[Y_{t+1} | c_t].
# log_Y_next[k] = log Y_{t+1}(c_t + e_k) for k = 1,...,K.
# Null boundary vertices: pair (1/2 at {1,j}) and triple (1/3 at {1,j,r}).
.sup_boundary <- function(log_Y_next) {
  K <- length(log_Y_next)
  Y <- exp(log_Y_next)
  competitors <- 2:K

  pair_vals <- vapply(competitors, function(j) (Y[1L] + Y[j]) / 2, numeric(1L))

  triple_vals <- if (K >= 3L) {
    combs <- combn(competitors, 2L)
    apply(combs, 2L, function(jk) (Y[1L] + Y[jk[1L]] + Y[jk[2L]]) / 3)
  } else {
    numeric(0L)
  }

  max(c(pair_vals, triple_vals))
}

# Load cached RIPr result or compute and cache it.
.load_or_compute_ripr <- function(n, q, results_dir) {
  path <- file.path(results_dir, sprintf("boundary_ripr_n%04d.rds", n))
  if (file.exists(path)) return(readRDS(path))
  result <- run_boundary_ripr(n = n, q = q)
  saveRDS(result, file = path)
  result
}

DirectRIPrSequenceTest <- new_class(
  "DirectRIPrSequenceTest",
  parent = Test,
  properties = list(
    q           = class_numeric,
    alpha       = class_numeric,
    results_dir = class_character
  ),
  constructor = function(q, alpha = 0.05, results_dir = "results", stream = NULL) {
    K <- length(q)
    if (K < 2L || any(q <= 0) || abs(sum(q) - 1) > 1e-10)
      stop("q must be a probability vector of length >= 2 summing to 1")
    if (q[1L] <= max(q[-1L]))
      stop("q[1] must exceed max(q[-1]): candidate 1 is the announced winner")
    if (alpha <= 0 || alpha >= 1)
      stop("alpha must be in (0, 1)")
    if (!dir.exists(results_dir))
      dir.create(results_dir, recursive = TRUE)

    state <- new.env(parent = emptyenv())
    state$counts    <- integer(K)
    state$log_a     <- 0
    state$log_Y     <- 0
    state$history_Z <- 1
    state$history_Y <- 1
    state$history_a <- 1
    state$n         <- 0L
    state$stop_time <- NA_integer_
    state$decision <- "continue"

    new_object(
      S7_object(),
      q           = q,
      alpha       = alpha,
      results_dir = results_dir,
      stream      = stream,
      description = sprintf("Direct RIPr sequence test (supermartingale, %d-candidate plurality)", K),
      state       = state
    )
  }
)

method(update, DirectRIPrSequenceTest) <- function(stat, new_x = NULL, ...) {
  if (is.null(new_x)) {
    if (is.null(stat@stream)) stop("Provide new_x or attach a stream.")
    new_x <- fetch(stat@stream)
  }
  if (length(new_x) == 0) return(invisible(stat))

  threshold <- 1 / stat@alpha
  old_n     <- stat@state$n
  log_q     <- log(stat@q)
  K         <- length(stat@q)
  m         <- length(new_x)
  new_z     <- numeric(m)
  new_y     <- numeric(m)
  new_a     <- numeric(m)

  for (i in seq_along(new_x)) {
    t_next <- old_n + i

    ripr <- .load_or_compute_ripr(t_next, stat@q, stat@results_dir)

    log_Y_next <- vapply(seq_len(K), function(k) {
      cnext    <- stat@state$counts
      cnext[k] <- cnext[k] + 1L
      .log_Y(cnext, ripr, log_q)
    }, numeric(1L))

    sup_val          <- .sup_boundary(log_Y_next)
    stat@state$log_a <- stat@state$log_a + stat@state$log_Y - log(sup_val)

    x                        <- new_x[i]
    stat@state$counts[x]     <- stat@state$counts[x] + 1L
    stat@state$log_Y         <- log_Y_next[x]

    new_z[i] <- exp(stat@state$log_a + stat@state$log_Y)
    new_y[i] <- exp(stat@state$log_Y)
    new_a[i] <- exp(stat@state$log_a)
  }

  stat@state$history_Z <- c(stat@state$history_Z, new_z)
  stat@state$history_Y <- c(stat@state$history_Y, new_y)
  stat@state$history_a <- c(stat@state$history_a, new_a)
  stat@state$n         <- old_n + m

  if (!is_stopped(stat)) {
    crossed <- which(new_z >= threshold)
    if (length(crossed) > 0L) {
      stat@state$decision  <- "reject_H0"
      stat@state$stop_time <- old_n + crossed[1L]
      message("DirectRIPrSequenceTest: rejection threshold reached.")
    }
  }

  invisible(stat)
}

method(reset, DirectRIPrSequenceTest) <- function(object, ...) {
  object@state$counts    <- integer(length(object@q))
  object@state$log_a     <- 0
  object@state$log_Y     <- 0
  object@state$history_Z <- 1
  object@state$history_Y <- 1
  object@state$history_a <- 1
  object@state$n         <- 0L
  object@state$stop_time <- NA_integer_
  object@state$decision  <- "continue"
  if (!is.null(object@stream)) reset(object@stream)
  invisible(object)
}

method(is_stopped, DirectRIPrSequenceTest)    <- function(stat, ...) decision(stat) != "continue"
method(decision, DirectRIPrSequenceTest)      <- function(test, ...) test@state$decision
method(n_obs, DirectRIPrSequenceTest)         <- function(stat, ...) stat@state$n
method(stopping_time, DirectRIPrSequenceTest) <- function(test, ...) test@state$stop_time

method(value, DirectRIPrSequenceTest) <- function(stat, n = 1L, ...) {
  tail(stat@state$history_Z, n = n)
}

method(print, DirectRIPrSequenceTest) <- function(x, ...) {
  K <- length(x@q)
  cat(sprintf("Direct RIPr sequence test (supermartingale, %d-candidate plurality)\n", K))
  cat("q:", x@q, "\n")
  cat("Observations:", x@state$n, "\n")
  cat("Counts:", x@state$counts, "\n")
  if (is_stopped(x)) cat("Stopping time:", x@state$stop_time, "\n")
  cat("Current Z_t:", round(tail(x@state$history_Z, 1L), 4), "\n")
  cat("Current Y_t:", round(tail(x@state$history_Y, 1L), 4), "\n")
  cat("Current a_t:", round(tail(x@state$history_a, 1L), 4), "\n")
  cat("Rejection threshold:", round(1 / x@alpha, 4), "\n")
  cat("Decision:", decision(x), "\n")
  invisible(x)
}
