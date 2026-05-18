# Utilities for optimisation over the probability simplex.
# Nothing in this file is specific to RIPr, the multinomial, or plurality audits.

#' Softmax reparametrisation of the simplex
#'
#' Maps an unconstrained vector `v` of length `d-1` to a point `alpha` on the
#' standard `(d-1)`-simplex via `softmax(c(0, v))`. The leading zero pins the
#' first coordinate, giving a bijection from `R^{d-1}` to the interior of
#' `Delta^{d-1}`. Suitable as an unconstrained parameterisation for BFGS.
#'
#' @param v Numeric vector of length `d-1`.
#' @return Numeric vector of length `d` summing to 1, with all entries > 0.
#' @seealso [v_from_alpha()] for the inverse, [softmax_jacobian()] for the
#'   derivative.
#' @export
alpha_from_v <- function(v) {
  u <- c(0, v)
  e <- exp(u - max(u))
  e / sum(e)
}

#' Inverse softmax reparametrisation
#'
#' Maps a point `alpha` in the interior of `Delta^{d-1}` back to the
#' unconstrained vector `v` such that `alpha_from_v(v) == alpha`. Uses
#' log-ratios relative to the first coordinate, with an `eps` guard against
#' exact zeros.
#'
#' @param alpha Numeric vector summing to 1, with all entries >= 0.
#' @param eps Small constant added before taking logs to avoid `-Inf`.
#'   Default: `1e-12`.
#' @return Numeric vector of length `length(alpha) - 1`.
#' @seealso [alpha_from_v()]
#' @export
v_from_alpha <- function(alpha, eps = 1e-12) {
  log(alpha[-1L] + eps) - log(alpha[1L] + eps)
}

#' Jacobian of the softmax reparametrisation
#'
#' Computes `d(alpha)/d(v)` at the point corresponding to `alpha`, where
#' `v = v_from_alpha(alpha)`. The result is the `d x (d-1)` matrix obtained
#' by dropping the first column of the full `d x d` softmax Jacobian, using
#' the chain rule through `d(u)/d(v) = rbind(0, I_{d-1})`.
#'
#' @param alpha Numeric vector of length `d` summing to 1 (a simplex point).
#' @return Numeric matrix of dimension `d x (d-1)`.
#' @seealso [alpha_from_v()], [v_from_alpha()]
#' @export
softmax_jacobian <- function(alpha) {
  (outer(alpha, alpha, function(a, b) -a * b) + diag(alpha))[,
    -1L,
    drop = FALSE
  ]
}

#' Mirror descent on the probability simplex
#'
#' Minimises a convex loss over the probability simplex using multiplicative
#' weights (entropic mirror descent) with a backtracking line search. The
#' update at each step is `w_new = w * exp(-lr * grad)`, renormalised to sum
#' to 1. The step size is halved up to 50 times until the loss decreases.
#' Stops early when the maximum weight change drops below `tol`.
#'
#' @param w_init Numeric vector of initial weights (need not sum to 1;
#'   normalised internally).
#' @param loss_and_grad Function taking a weight vector and returning a list
#'   with components `loss` (scalar) and `grad` (numeric vector of the same
#'   length as `w_init`).
#' @param max_iter Maximum number of gradient steps. Default: `1000`.
#' @param tol Convergence tolerance on `max(|w_new - w|)`. Default: `1e-12`.
#' @return Numeric vector of optimal weights summing to 1.
#' @export
mirror_descent <- function(
  w_init,
  loss_and_grad,
  max_iter = 1000L,
  tol = 1e-12
) {
  w <- w_init / sum(w_init)
  for (i in seq_len(max_iter)) {
    lr <- 1.0
    lg <- loss_and_grad(w)
    loss <- lg$loss
    grad_w <- lg$grad
    for (j in seq_len(50L)) {
      w_new <- w * exp(-lr * grad_w)
      w_new <- w_new / sum(w_new)
      if (loss_and_grad(w_new)$loss <= loss) {
        break
      }
      lr <- lr * 0.5
    }
    if (max(abs(w_new - w)) < tol) {
      break
    }
    w <- w_new
  }
  w
}

#' Simplex lattice (integer compositions)
#'
#' Generates all non-negative integer vectors of length `d` that sum to `n`.
#' Each row of the returned matrix is one such vector. Dividing by `n` gives a
#' uniform grid over the standard `(d-1)`-simplex with spacing `1/n`.
#'
#' @param d Integer. Number of coordinates (dimension of the ambient space).
#' @param n Integer. Target sum for each row.
#' @return Integer matrix with `choose(n + d - 1, d - 1)` rows and `d` columns.
#' @export
simplex_lattice <- function(d, n) {
  if (d == 1L) {
    return(matrix(n, nrow = 1L, ncol = 1L))
  }
  do.call(
    rbind,
    lapply(0L:n, function(k) cbind(k, simplex_lattice(d - 1L, n - k)))
  )
}
