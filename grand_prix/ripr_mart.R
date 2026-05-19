box::use(
  ripr / mixture[mixture_mnom],
  grand_prix / martingale_utils[log_mixture_mass],
  S7[
    new_class,
    new_object,
    method,
    `method<-`,
    class_numeric,
    class_any,
    class_integer,
    S7_object
  ],
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

# log Y_t(counts) = log Q(counts) - log P_W(counts) for two mixture_mnom objects.
.log_Y <- function(counts, Q, P_W) {
  log_mixture_mass(counts, Q) - log_mixture_mass(counts, P_W)
}

# sup_{θ on null boundary} E_θ[Y_{t+1} | c_t].
# log_Y_next[k] = log Y_{t+1}(c_t + e_k) for k = 1,...,K.
# Boundary vertex for subset S ⊆ {2,...,K} has θ[1] = θ[j] = 1/(|S|+1) for
# j ∈ S, giving E_θ[Y] = (Y[1] + Σ_{j∈S} Y[j]) / (|S|+1). The optimal S is
# always a prefix of the competitors sorted by Y descending (swapping any
# element for a larger one can only raise the average), so we check K-1
# prefix averages rather than all 2^{K-1} subsets.
.sup_boundary <- function(log_Y_next) {
  K <- length(log_Y_next)
  Y <- exp(log_Y_next)
  cum_sums <- Y[1L] + cumsum(sort(Y[-1L], decreasing = TRUE))
  max(cum_sums / (seq_len(K - 1L) + 1L))
}

# ---------------------------------------------------------------------------
# DirectRIPrSequenceTest — fixed-t RIPr e-variables corrected to a
# supermartingale (K-candidate plurality)
# ---------------------------------------------------------------------------
#
# ripr_fn is a user-supplied function(n) that returns list(mixture, e_ratio)
# where mixture is the boundary-optimal P_W (a mixture_mnom) for sample size n.
# Q is the alternative distribution (a mixture_mnom; use point_mnom for point).

#' Direct RIPr sequence test (supermartingale)
#'
#' At each step t, evaluates the RIPr e-value Y_t = Q(X^t) / P_W^t(X^t) for
#' the boundary-optimal mixture P_W^t returned by `ripr_fn(t)`. A running
#' correction factor a_t enforces the supermartingale property; the test
#' statistic is Z_t = a_t * Y_t.
#'
#' @param Q A `mixture_mnom` alternative. Use [point_mnom()] or
#'   [dirichlet_mnom()].
#' @param ripr_fn Function of signature `function(n)` returning
#'   `list(mixture = mixture_mnom, e_ratio = numeric)` — the boundary-optimal
#'   P_W and its duality gap for sample size n.
#' @param alpha Numeric. Significance level. Default: 0.05.
#' @param stream Optional stream object.
#' @return A `DirectRIPrSequenceTest` object.
#' @export
DirectRIPrSequenceTest <- new_class(
  "DirectRIPrSequenceTest",
  parent = Test,
  properties = list(
    Q = mixture_mnom,
    alpha = class_numeric,
    ripr_fn = class_any # function(n) -> list(mixture = mixture_mnom, e_ratio)
  ),
  constructor = function(Q, ripr_fn, alpha = 0.05, stream = NULL) {
    K <- nrow(Q@atoms)
    if (alpha <= 0 || alpha >= 1) {
      stop("alpha must be in (0, 1)")
    }
    if (!is.function(ripr_fn)) {
      stop("ripr_fn must be a function(n) returning list(mixture, e_ratio)")
    }

    state <- new.env(parent = emptyenv())
    state$counts <- integer(K)
    state$log_a <- 0
    state$log_Y <- 0
    state$history_Z <- 1
    state$history_Y <- 1
    state$history_a <- 1
    state$n <- 0L
    state$stop_time <- NA_integer_
    state$decision <- "continue"

    new_object(
      S7_object(),
      Q = Q,
      alpha = alpha,
      ripr_fn = ripr_fn,
      stream = stream,
      description = sprintf(
        "Direct RIPr sequence test (supermartingale, %d-candidate plurality)",
        K
      ),
      state = state
    )
  }
)

#' @export
method(update, DirectRIPrSequenceTest) <- function(stat, new_x = NULL, ...) {
  if (is_stopped(stat)) {
    message(
      "DirectRIPrSequenceTest has already stopped. Use reset() to restart."
    )
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
  K <- nrow(stat@Q@atoms)
  m <- length(new_x)
  new_z <- numeric(m)
  new_y <- numeric(m)
  new_a <- numeric(m)

  for (i in seq_along(new_x)) {
    t_next <- old_n + i
    ripr <- stat@ripr_fn(t_next)

    log_Y_next <- vapply(
      seq_len(K),
      function(k) {
        cnext <- stat@state$counts
        cnext[k] <- cnext[k] + 1L
        .log_Y(cnext, stat@Q, ripr$mixture)
      },
      numeric(1L)
    )

    sup_val <- .sup_boundary(log_Y_next)
    stat@state$log_a <- stat@state$log_a + stat@state$log_Y - log(sup_val)

    x <- new_x[i]
    stat@state$counts[x] <- stat@state$counts[x] + 1L
    stat@state$log_Y <- log_Y_next[x]

    new_z[i] <- exp(stat@state$log_a + stat@state$log_Y)
    new_y[i] <- exp(stat@state$log_Y)
    new_a[i] <- exp(stat@state$log_a)
  }

  stat@state$history_Z <- c(stat@state$history_Z, new_z)
  stat@state$history_Y <- c(stat@state$history_Y, new_y)
  stat@state$history_a <- c(stat@state$history_a, new_a)
  stat@state$n <- old_n + m

  if (!is_stopped(stat)) {
    crossed <- which(new_z >= threshold)
    if (length(crossed) > 0L) {
      stat@state$decision <- "reject_H0"
      stat@state$stop_time <- old_n + crossed[1L]
      message("DirectRIPrSequenceTest: rejection threshold reached.")
    }
  }

  invisible(stat)
}

#' @export
method(reset, DirectRIPrSequenceTest) <- function(object, ...) {
  object@state$counts <- integer(nrow(object@Q@atoms))
  object@state$log_a <- 0
  object@state$log_Y <- 0
  object@state$history_Z <- 1
  object@state$history_Y <- 1
  object@state$history_a <- 1
  object@state$n <- 0L
  object@state$stop_time <- NA_integer_
  object@state$decision <- "continue"
  if (!is.null(object@stream)) {
    reset(object@stream)
  }
  invisible(object)
}

#' @export
method(is_stopped, DirectRIPrSequenceTest) <- function(stat, ...) {
  decision(stat) != "continue"
}
#' @export
method(decision, DirectRIPrSequenceTest) <- function(test, ...) {
  test@state$decision
}
#' @export
method(n_obs, DirectRIPrSequenceTest) <- function(stat, ...) stat@state$n
#' @export
method(stopping_time, DirectRIPrSequenceTest) <- function(test, ...) {
  test@state$stop_time
}

#' @export
method(value, DirectRIPrSequenceTest) <- function(stat, n = 1L, ...) {
  tail(stat@state$history_Z, n = n)
}

#' @export
method(print, DirectRIPrSequenceTest) <- function(x, ...) {
  K <- nrow(x@Q@atoms)
  cat(sprintf(
    "Direct RIPr sequence test (supermartingale, %d-candidate plurality)\n",
    K
  ))
  cat("Atoms:", ncol(x@Q@atoms), "\n")
  cat("Observations:", x@state$n, "\n")
  cat("Counts:", x@state$counts, "\n")
  if (is_stopped(x)) {
    cat("Stopping time:", x@state$stop_time, "\n")
  }
  cat("Current Z_t:", round(tail(x@state$history_Z, 1L), 4), "\n")
  cat("Current Y_t:", round(tail(x@state$history_Y, 1L), 4), "\n")
  cat("Current a_t:", round(tail(x@state$history_a, 1L), 4), "\n")
  cat("Rejection threshold:", round(1 / x@alpha, 4), "\n")
  cat("Decision:", decision(x), "\n")
  invisible(x)
}


# ---------------------------------------------------------------------------
# BatchRIPr — product of fixed-n RIPr e-values over non-overlapping batches
# ---------------------------------------------------------------------------
#
# E_k = Q(c_k) / P_W(c_k) / e_ratio
# where Q is the alternative mixture_mnom (numerator), P_W is the boundary-
# optimal mixture_mnom from run_plurality_ripr (denominator), and e_ratio =
# max_theta E_theta[Q/P_W] guarantees E_theta[E_k] <= 1 for all theta in H_0.
#
# Both point and Dirichlet mixture alternatives are supported via mixture_mnom.

#' Batch RIPr test martingale for K-candidate plurality
#'
#' Product of corrected RIPr e-values over non-overlapping batches of size
#' `P_W@n`. Each batch contributes E_k = Q(c_k) / P_W(c_k) / e_ratio, where
#' c_k is the count vector for batch k. The correction by `e_ratio` (the
#' duality gap from [run_plurality_ripr()]) ensures E_theta[E_k] <= 1 for all
#' theta in H_0. Supports both point and Dirichlet mixture alternatives.
#'
#' @param Q A `mixture_mnom` alternative (numerator). Use [point_mnom()] or
#'   [dirichlet_mnom()].
#' @param P_W A `mixture_mnom` with `@n` set to the batch size (denominator).
#'   Typically the `mixture` component of [run_plurality_ripr()] output.
#' @param e_ratio Numeric. Duality gap (`E_star`) from [run_plurality_ripr()].
#' @param alpha Numeric. Significance level. Default: 0.05.
#' @param stream Optional stream object.
#' @return A `BatchRIPr` object.
#' @export
BatchRIPr <- new_class(
  "BatchRIPr",
  parent = Test,
  properties = list(
    Q = mixture_mnom,
    P_W = mixture_mnom,
    batch_size = class_integer,
    alpha = class_numeric
  ),
  constructor = function(Q, P_W, e_ratio, alpha = 0.05, stream = NULL) {
    K <- nrow(Q@atoms)
    batch_size <- as.integer(P_W@n)
    if (alpha <= 0 || alpha >= 1) {
      stop("alpha must be in (0, 1)")
    }
    if (is.null(batch_size) || batch_size < 1L) {
      stop("P_W@n must be a positive integer (the batch size)")
    }

    state <- new.env(parent = emptyenv())
    state$correction <- log(e_ratio)
    state$batch_counts <- integer(K)
    state$log_m <- 0
    state$history <- 1
    state$n <- 0L
    state$stop_time <- NA_integer_
    state$decision <- "continue"

    new_object(
      S7_object(),
      Q = Q,
      P_W = P_W,
      batch_size = batch_size,
      alpha = alpha,
      stream = stream,
      description = sprintf(
        "Batch RIPr test martingale (batch=%d, %d-candidate plurality)",
        batch_size,
        K
      ),
      state = state
    )
  }
)

#' @export
method(update, BatchRIPr) <- function(stat, new_x = NULL, ...) {
  if (is_stopped(stat)) {
    message("BatchRIPr has already stopped. Use reset() to restart.")
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
  K <- nrow(stat@Q@atoms)
  new_vals <- numeric(length(new_x))

  for (i in seq_along(new_x)) {
    x <- new_x[i]
    stat@state$batch_counts[x] <- stat@state$batch_counts[x] + 1L

    if ((old_n + i) %% stat@batch_size == 0L) {
      log_Y <- .log_Y(stat@state$batch_counts, stat@Q, stat@P_W)
      stat@state$log_m <- stat@state$log_m + log_Y - stat@state$correction
      stat@state$batch_counts <- integer(K)
    }

    new_vals[i] <- exp(stat@state$log_m)
  }

  stat@state$history <- c(stat@state$history, new_vals)
  stat@state$n <- old_n + length(new_x)

  if (!is_stopped(stat)) {
    crossed <- which(new_vals >= threshold)
    if (length(crossed) > 0L) {
      stat@state$decision <- "reject_H0"
      stat@state$stop_time <- old_n + crossed[1L]
      message("BatchRIPr: rejection threshold reached.")
    }
  }

  invisible(stat)
}

#' @export
method(reset, BatchRIPr) <- function(object, ...) {
  object@state$batch_counts <- integer(nrow(object@Q@atoms))
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

#' @export
method(is_stopped, BatchRIPr) <- function(stat, ...) {
  decision(stat) != "continue"
}
#' @export
method(decision, BatchRIPr) <- function(test, ...) test@state$decision
#' @export
method(n_obs, BatchRIPr) <- function(stat, ...) stat@state$n
#' @export
method(stopping_time, BatchRIPr) <- function(test, ...) test@state$stop_time

#' @export
method(value, BatchRIPr) <- function(stat, n = 1L, ...) {
  tail(stat@state$history, n = n)
}

#' @export
method(print, BatchRIPr) <- function(x, ...) {
  K <- nrow(x@Q@atoms)
  cat(sprintf(
    "Batch RIPr test martingale (batch=%d, %d-candidate plurality)\n",
    x@batch_size,
    K
  ))
  cat("Q atoms:", ncol(x@Q@atoms), "\n")
  cat("Observations:", x@state$n, "\n")
  cat("Completed batches:", x@state$n %/% x@batch_size, "\n")
  cat("Batch counts so far:", x@state$batch_counts, "\n")
  if (is_stopped(x)) {
    cat("Stopping time:", x@state$stop_time, "\n")
  }
  cat("Current M_t:", round(tail(x@state$history, 1L), 4), "\n")
  cat("Rejection threshold:", round(1 / x@alpha, 4), "\n")
  cat("Decision:", decision(x), "\n")
  invisible(x)
}
