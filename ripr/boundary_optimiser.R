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

#' Iterative boundary atom optimiser for RIPr (3-candidate plurality)
#'
#' Grows a mixture P_w supported on the null hypothesis boundary by repeatedly
#' adding the most adversarial atom:
#'   theta* = argmax_{theta on null boundary} E_theta[Q(X) / P_w(X)]
#' then reweighting all atoms to minimise D(q || P_w).  Stops when
#' max_theta E_theta[Q / P_w] <= 1 + tol (the RIPr validity condition).
#'
#' The null boundary is traced by [null_boundary_3()]: distributions where the
#' announced winner ties with their best competitor, identical to the path
#' produced by [make_simplex_grid()].
#'
#' Runs on CPU in base R (no GPU required).
#'
#' @param n Total ballot count.
#' @param q Numeric vector of length 3 — the numerator distribution Q.
#' @param n_atoms Maximum atoms to add. Default: 50.
#' @param oracle_grid Number of grid points for the boundary oracle search.
#'   A coarse grid finds the basin, then `optimize()` refines within it.
#'   Default: 200.
#' @param reweight_maxit Max L-BFGS iterations for the reweighting step.
#'   Default: 1000.
#' @param tol Convergence tolerance: stop when max E_theta <= 1 + tol.
#'   Default: 1e-4.
#' @param emit_fn Progress message callback. Default: message.
#' @param monitor_fn Optional callback called each iteration with a named list:
#'   `atom`, `max_E_ratio`, `theta_star`, `weights`, `ws_list`. Default: NULL.
#' @return List with:
#'   - `ws_list`: list of atom probability vectors (all on null boundary).
#'   - `weights`: final mixture weights (sums to 1).
#'   - `n_atoms`: number of atoms in the final mixture.
#'   - `atom_history`: list of per-iteration info (theta_star, E_ratio, weights).
#'   - `converged`: TRUE if the validity condition was met.
#' @export
run_boundary_ripr <- function(
  n,
  q,
  n_atoms        = 50L,
  oracle_grid    = 200L,
  reweight_maxit = 1000L,
  tol            = 1e-4,
  min_dist       = NULL,
  emit_fn        = message,
  monitor_fn     = NULL
) {
  m <- 3L

  # --- Enumerate all multinomial outcomes (M x m integer matrix) ---
  X_tensor <- build_counts_tensor(n, m)
  X_mat    <- matrix(as.integer(as.array(X_tensor$cpu())), ncol = m)
  M        <- nrow(X_mat)

  # log Multinom(x; n, theta) for all x: vectorised via X %*% log(theta).
  log_base <- lgamma(n + 1) - rowSums(lgamma(X_mat + 1))
  log_multinom_all <- function(theta) {
    log_theta <- ifelse(theta > 0, log(theta), -Inf)
    log_base + as.vector(X_mat %*% log_theta)
  }

  # log P_w(x) = logsumexp_k( log(w_k) + log_comp_k(x) )
  compute_log_Pw <- function(log_comps, weights) {
    log_comps_w <- sweep(log_comps, 2L, log(weights), "+")
    row_max     <- apply(log_comps_w, 1L, max)
    log(rowSums(exp(log_comps_w - row_max))) + row_max
  }

  # E_theta[Q(X)/P_w(X)] = sum_x Multinom(x;n,theta) * Q(x)/P_w(x)
  log_q_mass <- log_multinom_all(q)
  q_mass     <- ifelse(is.finite(log_q_mass), exp(log_q_mass), 0)
  finite_q   <- q_mass > 0

  compute_E_ratio <- function(log_theta_mass, log_Pw) {
    log_terms <- log_theta_mass + log_q_mass - log_Pw
    finite    <- is.finite(log_terms)
    if (!any(finite)) return(0)
    max_val   <- max(log_terms[finite])
    exp(max_val) * sum(exp(log_terms[finite] - max_val))
  }

  # Invert null_boundary_3 to recover s from an atom.
  atom_to_s <- function(theta) {
    if (theta[2L] >= theta[3L]) 1.5 - 3 * theta[1L]
    else                        3 * theta[1L] - 0.5
  }

  # --- Oracle: argmax_{s in [0,1]} E_{null_boundary_3(s)}[Q/P_w] ---
  # Coarse grid identifies the best basin; optimize() refines within it.
  # existing_s: s-coordinates of atoms already in the mixture.
  # Points within min_dist of any existing atom are excluded.
  find_best_theta <- function(log_Pw, existing_s = numeric(0L)) {
    s_grid    <- seq(0, 1, length.out = oracle_grid)
    too_close <- if (!is.null(min_dist) && length(existing_s) > 0L)
      vapply(s_grid, function(s) any(abs(s - existing_s) < min_dist), logical(1L))
    else
      rep(FALSE, oracle_grid)

    E_grid          <- vapply(s_grid, function(s) {
      compute_E_ratio(log_multinom_all(null_boundary_3(s)), log_Pw)
    }, numeric(1L))
    E_grid[too_close] <- -Inf

    if (all(!is.finite(E_grid))) return(list(theta = NULL, E_ratio = -Inf))

    best_i <- which.max(E_grid)
    s_best <- s_grid[best_i]
    lo     <- if (best_i > 1L)          s_grid[best_i - 1L] else 0
    hi     <- if (best_i < oracle_grid) s_grid[best_i + 1L] else 1

    # Clip optimize() interval away from excluded zones
    if (!is.null(min_dist) && length(existing_s) > 0L) {
      left_atoms  <- existing_s[existing_s < s_best]
      right_atoms <- existing_s[existing_s > s_best]
      if (length(left_atoms)  > 0L) lo <- max(lo, max(left_atoms)  + min_dist)
      if (length(right_atoms) > 0L) hi <- min(hi, min(right_atoms) - min_dist)
    }

    if (lo >= hi) return(list(theta = null_boundary_3(s_best), E_ratio = E_grid[best_i]))

    res <- stats::optimize(
      function(s) -compute_E_ratio(log_multinom_all(null_boundary_3(s)), log_Pw),
      interval = c(lo, hi)
    )
    list(theta = null_boundary_3(res$minimum), E_ratio = -res$objective)
  }

  # --- Reweighting: minimise D(q || P_w) = -E_q[log P_w(X)] over weights ---
  # Mirror descent (multiplicative weights): w <- w * exp(-lr * grad) / Z.
  # Natural for simplex problems; handles near-zero weights without the
  # conditioning issues that BFGS+softmax develops as K grows.
  reweight_mirror <- function(log_comps, w_init, max_iter = reweight_maxit, tol = 1e-12) {
    w <- w_init / sum(w_init)
    lr <- 1.0
    for (i in seq_len(max_iter)) {
      log_Pw <- compute_log_Pw(log_comps, w)
      lrs    <- sweep(log_comps[finite_q, , drop = FALSE], 1L, log_Pw[finite_q], "-")
      grad_w <- -colSums(q_mass[finite_q] * exp(lrs))
      loss   <- -sum(q_mass[finite_q] * log_Pw[finite_q])
      # Backtracking line search: halve lr until loss decreases
      for (j in seq_len(50L)) {
        w_new     <- w * exp(-lr * grad_w); w_new <- w_new / sum(w_new)
        log_Pw_new <- compute_log_Pw(log_comps, w_new)
        if (-sum(q_mass[finite_q] * log_Pw_new[finite_q]) <= loss) break
        lr <- lr * 0.5
      }
      if (max(abs(w_new - w)) < tol) break
      w  <- w_new
      lr <- min(lr * 1.2, 10)  # cautiously grow step after successful step
    }
    w
  }

  # --- Initialise: one atom on each boundary segment, closest to q ---
  atom1   <- c((q[1L] + q[2L]) / 2, (q[1L] + q[2L]) / 2, q[3L])
  atom2   <- c((q[1L] + q[3L]) / 2, q[2L],                (q[1L] + q[3L]) / 2)
  ws_list   <- list(atom1, atom2)
  log_comps <- cbind(log_multinom_all(atom1), log_multinom_all(atom2))
  weights   <- reweight_mirror(log_comps, c(0.5, 0.5))
  converged <- FALSE

  atom_history <- vector("list", n_atoms)
  prev_kl      <- Inf

  for (atom_idx in seq_len(n_atoms)) {
    existing_s  <- vapply(ws_list, atom_to_s, numeric(1L))
    log_Pw      <- compute_log_Pw(log_comps, weights)
    E_star      <- find_best_theta(log_Pw)$E_ratio          # unconstrained: for convergence
    oracle_res  <- find_best_theta(log_Pw, existing_s)      # constrained: for atom selection
    theta_star  <- oracle_res$theta

    if (is.null(theta_star)) {
      emit_fn("All candidate atoms excluded by min_dist; stopping.")
      break
    }

    kl      <- sum(q_mass[finite_q] * (log_q_mass[finite_q] - log_Pw[finite_q]))
    kl_diff <- prev_kl - kl
    prev_kl <- kl
    emit_fn(sprintf(
      "Atom %d/%d: max_E_ratio - 1 = %e  delta_kl = %e",
      atom_idx, n_atoms, E_star - 1, kl_diff
    ))

    atom_history[[atom_idx]] <- list(
      theta_star = theta_star,
      E_ratio    = E_star,
      weights    = weights
    )

    if (!is.null(monitor_fn)) monitor_fn(list(
      atom        = atom_idx,
      max_E_ratio = E_star,
      theta_star  = theta_star,
      weights     = weights,
      ws_list     = ws_list
    ))

    if (E_star <= 1 + tol) {
      emit_fn(sprintf("Converged after %d atoms (max_E_ratio - 1 = %e).", length(ws_list), E_star - 1))
      converged <- TRUE
      break
    }

    if (length(ws_list) >= n_atoms) break

    # Add theta* as new mixture component and reweight
    ws_list   <- c(ws_list, list(theta_star))
    log_comps <- cbind(log_comps, log_multinom_all(theta_star))
    K         <- ncol(log_comps)

    w_init  <- c(weights * (K - 1L) / K, 1 / K)
    weights <- reweight_mirror(log_comps, w_init)
  }

  list(
    ws_list      = ws_list,
    weights      = weights,
    n_atoms      = length(ws_list),
    atom_history = atom_history[!sapply(atom_history, is.null)],
    converged    = converged
  )
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
