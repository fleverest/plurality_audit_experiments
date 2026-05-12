box::use(
  ripr / multinomial[build_counts_tensor]
)

#' KL divergence D(q ‖ P_w) for a discrete mixture P_w
#'
#' Computes the loss minimised by [run_boundary_ripr()]:
#'   D(q ‖ P_w) = Σ_x q(x) log(q(x) / P_w(x))
#' where P_w(x) = Σ_k weights[k] · Multinom(x; n, ws_list[[k]]).
#'
#' @param n Total number of trials per observation.
#' @param q Numeric vector — the numerator distribution Q (length m).
#' @param ws_list List of numeric vectors — mixture atom distributions (each length m).
#' @param weights Numeric vector — mixture weights summing to 1 (length K).
#' @return Scalar KL divergence.
#' @export
ripr_loss <- function(n, q, ws_list, weights) {
  X_tensor <- build_counts_tensor(n, length(q))
  X_mat    <- matrix(as.integer(as.array(X_tensor$cpu())), ncol = length(q))
  log_base <- lgamma(n + 1) - rowSums(lgamma(X_mat + 1))

  log_multinom <- function(theta) {
    log_theta <- ifelse(theta > 0, log(theta), -Inf)
    log_base + as.vector(X_mat %*% log_theta)
  }

  log_q_mass  <- log_multinom(q)
  q_mass      <- exp(log_q_mass)
  finite      <- is.finite(log_q_mass) & q_mass > 0

  log_comps   <- do.call(cbind, lapply(ws_list, log_multinom))
  log_comps_w <- sweep(log_comps, 2L, log(weights), "+")
  row_max     <- apply(log_comps_w, 1L, max)
  log_Pw      <- log(rowSums(exp(log_comps_w - row_max))) + row_max

  sum(q_mass[finite] * (log_q_mass[finite] - log_Pw[finite]))
}

#' E_theta[Q(X)/P_w(X)] for each theta in a grid
#'
#' @param n Total number of trials per observation.
#' @param q Numeric vector — the numerator distribution Q (length m).
#' @param ws_list List of numeric vectors — mixture atom distributions (each length m).
#' @param weights Numeric vector — mixture weights summing to 1 (length K).
#' @param thetas List of numeric vectors — DGP grid (each length m).
#' @return Numeric vector of length T: E_theta[Q(X)/P_w(X)] for each theta.
#' @export
ripr_expectations <- function(n, q, ws_list, weights, thetas) {
  X_tensor <- build_counts_tensor(n, length(q))
  X_mat    <- matrix(as.integer(as.array(X_tensor$cpu())), ncol = length(q))
  log_base <- lgamma(n + 1) - rowSums(lgamma(X_mat + 1))

  log_multinom <- function(theta) {
    log_theta <- ifelse(theta > 0, log(theta), -Inf)
    log_base + as.vector(X_mat %*% log_theta)
  }

  log_q_mass  <- log_multinom(q)
  log_comps   <- do.call(cbind, lapply(ws_list, log_multinom))
  log_comps_w <- sweep(log_comps, 2L, log(weights), "+")
  row_max     <- apply(log_comps_w, 1L, max)
  log_Pw      <- log(rowSums(exp(log_comps_w - row_max))) + row_max
  log_llr     <- log_q_mass - log_Pw

  vapply(thetas, function(theta) {
    log_terms <- log_multinom(theta) + log_llr
    finite    <- is.finite(log_terms)
    if (!any(finite)) return(0)
    m <- max(log_terms[finite])
    exp(m) * sum(exp(log_terms[finite] - m))
  }, numeric(1L))
}

# Parametrise the null boundary for a 3-candidate plurality audit.
# The boundary is the set where the winner (category 1) ties with their best
# competitor: (1/2,1/2,0) -> (1/3,1/3,1/3) -> (1/2,0,1/2).
# s in [0,1] traces the full path.
null_boundary_3 <- function(s) {
  if (s <= 0.5) {
    t <- 0.5 - s / 3
    c(t, t, 1 - 2 * t)
  } else {
    t <- 1 / 3 + (s - 0.5) / 3
    c(t, 1 - 2 * t, t)
  }
}

#' Find the optimal two-atom boundary mixture
#'
#' Jointly optimises the positions of one atom on each boundary segment
#' (s1 in [0, 0.5], s2 in [0.5, 1]) and their mixture weight, minimising
#' D(q || w1*P_theta1 + w2*P_theta2).
#'
#' A coarse grid search over (s1, s2) pairs seeds the optimisation, with the
#' weight solved via 1D search at each grid pair. The best grid point is then
#' refined with `optim()`.
#'
#' @param n Total ballot count.
#' @param q Numeric vector of length 3 — the numerator distribution Q.
#' @param grid Number of grid points per segment for the coarse search. Default: 50.
#' @return List with:
#'   - `ws_list`: list of two atom probability vectors.
#'   - `weights`: numeric vector of length 2 summing to 1.
#'   - `s1`, `s2`: boundary parameters of the two atoms.
#'   - `max_E_ratio`: max_theta E_theta[Q/P_w] at the optimum.
#' @export
optimise_two_atoms <- function(n, q, grid = 50L) {
  m <- 3L

  X_tensor <- build_counts_tensor(n, m)
  X_mat    <- matrix(as.integer(as.array(X_tensor$cpu())), ncol = m)

  log_base <- lgamma(n + 1) - rowSums(lgamma(X_mat + 1))
  log_multinom_all <- function(theta) {
    log_base + as.vector(X_mat %*% ifelse(theta > 0, log(theta), -Inf))
  }

  log_q_mass <- log_multinom_all(q)
  q_mass     <- ifelse(is.finite(log_q_mass), exp(log_q_mass), 0)
  finite_q   <- q_mass > 0

  kl_loss <- function(log_comps, w1) {
    w       <- c(w1, 1 - w1)
    lc_w    <- sweep(log_comps, 2L, log(w), "+")
    row_max <- apply(lc_w, 1L, max)
    log_Pw  <- ifelse(is.finite(row_max),
                      log(rowSums(exp(lc_w - row_max))) + row_max,
                      -Inf)
    -sum(q_mass[finite_q] * log_Pw[finite_q])
  }

  # For fixed atom positions, find the optimal w1 in (0, 1).
  best_w1 <- function(log_comps) {
    stats::optimize(function(w1) kl_loss(log_comps, w1), c(1e-6, 1 - 1e-6))$minimum
  }

  # Unconstrained parametrisation for optim():
  #   s1 = 0.5 * plogis(u1), s2 = 0.5 + 0.5 * plogis(u2), w1 = plogis(u3)
  objective <- function(u) {
    s1 <- 0.5  * stats::plogis(u[1L])
    s2 <- 0.5  + 0.5 * stats::plogis(u[2L])
    w1 <- stats::plogis(u[3L])
    log_comps <- cbind(log_multinom_all(null_boundary_3(s1)),
                       log_multinom_all(null_boundary_3(s2)))
    kl_loss(log_comps, w1)
  }

  # Coarse grid search over (s1, s2) to seed optim()
  s1_grid   <- seq(0, 0.5,  length.out = grid + 1L)[-(grid + 1L)]  # exclude 0.5 (shared point)
  s2_grid   <- seq(0.5, 1,  length.out = grid + 1L)[-1L]           # exclude 0.5
  best_loss <- Inf
  best_u    <- c(0, 0, 0)

  for (s1 in s1_grid) {
    lc1 <- log_multinom_all(null_boundary_3(s1))
    for (s2 in s2_grid) {
      lc  <- cbind(lc1, log_multinom_all(null_boundary_3(s2)))
      w1  <- best_w1(lc)
      val <- kl_loss(lc, w1)
      if (val < best_loss) {
        best_loss <- val
        best_u    <- c(stats::qlogis(s1 / 0.5),
                       stats::qlogis((s2 - 0.5) / 0.5),
                       stats::qlogis(w1))
      }
    }
  }

  res <- stats::optim(best_u, objective, method = "BFGS",
                      control = list(maxit = 10000L))

  s1    <- 0.5 * stats::plogis(res$par[1L])
  s2    <- 0.5 + 0.5 * stats::plogis(res$par[2L])
  w1    <- stats::plogis(res$par[3L])
  atom1 <- null_boundary_3(s1)
  atom2 <- null_boundary_3(s2)

  log_comps <- cbind(log_multinom_all(atom1), log_multinom_all(atom2))
  row_max   <- apply(sweep(log_comps, 2L, log(c(w1, 1 - w1)), "+"), 1L, max)
  log_Pw    <- log(rowSums(exp(sweep(log_comps, 2L, log(c(w1, 1 - w1)), "+") - row_max))) + row_max

  # Oracle: max_theta E_theta[Q/P_w] over full boundary
  s_grid  <- seq(0, 1, length.out = 1000L)
  E_vals  <- vapply(s_grid, function(s) {
    lm <- log_multinom_all(null_boundary_3(s))
    lt <- lm + log_q_mass - log_Pw
    fi <- is.finite(lt)
    if (!any(fi)) return(0)
    mv <- max(lt[fi]); exp(mv) * sum(exp(lt[fi] - mv))
  }, numeric(1L))

  list(
    ws_list     = list(atom1, atom2),
    weights     = c(w1, 1 - w1),
    s1          = s1,
    s2          = s2,
    max_E_ratio = max(E_vals)
  )
}
