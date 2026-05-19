box::use(
  ripr / mixture[mixture_mnom],
  grand_prix / martingale_utils[log_mixture_mass],
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

# Restricted-MLE log-likelihood for K-candidate plurality.
#
# Returns max_{θ ∈ H_0} Σ_k c_k log θ_k  where H_0 = {θ_1 ≤ max(θ_2,...,θ_K)}.
.rmle_log_lik <- function(counts) {
  tot <- sum(counts)
  if (tot == 0L) {
    return(0)
  }
  K <- length(counts)
  safe_loglik <- function(theta) {
    sum(ifelse(counts == 0L | theta <= 0, 0, counts * log(theta)))
  }

  if (counts[1L] <= max(counts[-1L])) {
    safe_loglik(counts / tot)
  } else {
    max(vapply(
      2:K,
      function(j) {
        theta <- counts / tot
        pj <- (counts[1L] + counts[j]) / (2 * tot)
        theta[1L] <- pj
        theta[j] <- pj
        safe_loglik(theta)
      },
      numeric(1L)
    ))
  }
}

# ---------------------------------------------------------------------------
# UITest — Universal Inference e-process (K-candidate plurality)
# ---------------------------------------------------------------------------
#
# E_t^UI = Q(X^t) / sup_{θ ∈ H_0} p_θ(X^t)
#
# Q is a mixture_mnom (use point_mnom for the standard point-alternative case).
# The numerator Q(X^t) is the mixture-multinomial mass at the cumulative counts.
# The denominator is the restricted-MLE likelihood over the joint sample.

#' Universal Inference e-process for K-candidate plurality
#'
#' Sequential test based on the e-process E_t = Q(X^t) / sup_{θ ∈ H_0} p_θ(X^t),
#' where Q is a mixture-multinomial alternative and the denominator is the
#' restricted-MLE likelihood. Valid for any stopping time.
#'
#' @param Q A `mixture_mnom` alternative. Use [point_mnom()] for a single
#'   point alternative or [dirichlet_mnom()] for a grid-weighted Dirichlet
#'   prior.
#' @param alpha Numeric. Significance level; rejection when E_t >= 1/alpha.
#'   Must be in (0, 1). Default: 0.05.
#' @param stream Optional stream object; if supplied, `new_x` is fetched from
#'   it when `update()` is called without `new_x`.
#' @return A `UITest` object.
#' @export
UITest <- new_class(
  "UITest",
  parent = Test,
  properties = list(
    Q = mixture_mnom,
    alpha = class_numeric
  ),
  constructor = function(Q, alpha = 0.05, stream = NULL) {
    K <- nrow(Q@atoms)
    if (alpha <= 0 || alpha >= 1) {
      stop("alpha must be in (0, 1)")
    }

    state <- new.env(parent = emptyenv())
    state$counts <- integer(K)
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
        "Universal Inference e-process (%d-candidate plurality)",
        K
      ),
      state = state
    )
  }
)

#' @export
method(update, UITest) <- function(stat, new_x = NULL, ...) {
  if (is_stopped(stat)) {
    message("UITest has already stopped. Use reset() to restart.")
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
    stat@state$counts[x] <- stat@state$counts[x] + 1L
    log_Q <- log_mixture_mass(stat@state$counts, stat@Q)
    log_rmle <- .rmle_log_lik(stat@state$counts)
    new_e[i] <- exp(log_Q - log_rmle)
  }

  stat@state$history <- c(stat@state$history, new_e)
  stat@state$n <- old_n + length(new_x)

  if (!is_stopped(stat)) {
    crossed <- which(new_e >= threshold)
    if (length(crossed) > 0L) {
      stat@state$decision <- "reject_H0"
      stat@state$stop_time <- old_n + crossed[1L]
      message("UITest: rejection threshold reached.")
    }
  }

  invisible(stat)
}

#' @export
method(reset, UITest) <- function(object, ...) {
  object@state$counts <- integer(nrow(object@Q@atoms))
  object@state$history <- 1
  object@state$n <- 0L
  object@state$stop_time <- NA_integer_
  object@state$decision <- "continue"
  if (!is.null(object@stream)) {
    reset(object@stream)
  }
  invisible(object)
}

#' @export
method(is_stopped, UITest) <- function(stat, ...) decision(stat) != "continue"
#' @export
method(decision, UITest) <- function(test, ...) test@state$decision
#' @export
method(n_obs, UITest) <- function(stat, ...) stat@state$n
#' @export
method(stopping_time, UITest) <- function(test, ...) test@state$stop_time

#' @export
method(value, UITest) <- function(stat, n = 1L, ...) {
  tail(stat@state$history, n = n)
}

#' @export
method(print, UITest) <- function(x, ...) {
  K <- nrow(x@Q@atoms)
  cat(sprintf("Universal Inference e-process (%d-candidate plurality)\n", K))
  cat("Atoms:", ncol(x@Q@atoms), "\n")
  cat("Observations:", x@state$n, "\n")
  cat("Counts:", x@state$counts, "\n")
  if (is_stopped(x)) {
    cat("Stopping time:", x@state$stop_time, "\n")
  }
  cat("Current E_t:", round(tail(x@state$history, 1L), 4), "\n")
  cat("Rejection threshold:", round(1 / x@alpha, 4), "\n")
  cat("Decision:", decision(x), "\n")
  invisible(x)
}
