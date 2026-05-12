box::use(
  ripr / multinomial[build_counts_tensor]
)

# All non-negative integer vectors of length d summing to n (rows of result sum to n).
# Used to build a lattice grid over the standard (d-1)-simplex.
simplex_lattice <- function(d, n) {
  if (d == 1L) return(matrix(n, nrow = 1L, ncol = 1L))
  do.call(rbind, lapply(0L:n, function(k) cbind(k, simplex_lattice(d - 1L, n - k))))
}

# Point on null-boundary face j for K-candidate plurality audit.
# alpha: length-(K-1) vector in the standard (K-2)-simplex (sums to 1, all >= 0).
#
# Face j is the set where theta_1 = theta_j >= all others. Its K-1 vertices are:
#   v_0          : pure-pair  (1/2 at pos 1, 1/2 at pos j, 0 elsewhere)
#   v_r (r>=1)   : triple     (1/3 at pos 1, 1/3 at pos j, 1/3 at pos others[r])
# where others = sort({2,...,K} \ {j}).
#
# theta_1 = theta_j = alpha[1]/6 + 1/3,  theta[others[r]] = alpha[r+1]/3.
null_boundary_face <- function(j, alpha, K) {
  others    <- setdiff(seq_len(K)[-1L], j)
  theta     <- numeric(K)
  theta[1L] <- alpha[1L] / 6 + 1 / 3
  theta[j]  <- theta[1L]
  for (r in seq_along(others)) theta[others[r]] <- alpha[r + 1L] / 3
  theta
}

# Jacobian of null_boundary_face(j, alpha, K) w.r.t. alpha.
# theta[1] = theta[j] = alpha[1]/6 + 1/3,  theta[others[r]] = alpha[r+1]/3.
# d theta / d alpha is a K x d matrix.
face_jacobian <- function(K, d, j) {
  others <- setdiff(seq_len(K), c(1L, j))
  J <- matrix(0, K, K - 1L)
  J[1L,  1L] <- 1 / 6
  J[j,   1L] <- 1 / 6
  for (r in seq_along(others)) J[others[r], r + 1L] <- 1 / 3
  J
}

#' Iterative boundary atom optimiser for RIPr (K-candidate plurality)
#'
#' Grows a mixture P_w supported on the null hypothesis boundary by repeatedly
#' adding the most adversarial atom on each boundary face:
#'   theta*_j = argmax_{theta on face j} E_theta[Q(X) / P_w(X)]
#' then reweighting all atoms to minimise D(q || P_w).  Stops when
#' max_theta E_theta[Q / P_w] <= 1 + tol (the RIPr validity condition).
#'
#' K is inferred from length(q).  The null boundary has K-1 faces (indexed by
#' which competitor ties with the announced winner).  Each face is a
#' (K-2)-simplex parameterised by [null_boundary_face()].  One atom per face
#' is added in each iteration (K-1 atoms total per round).
#'
#' Runs on CPU in base R (no GPU required).
#'
#' @param n Total ballot count.
#' @param q Numeric vector of length K — the numerator distribution Q.
#' @param n_atoms Maximum total atoms (including the initial K-1). Default: 50.
#' @param oracle_grid Grid density per simplex dimension for the boundary oracle.
#'   Total lattice points per face ~ oracle_grid^(K-2); reduce for K >= 4.
#'   Default: 200.
#' @param reweight_maxit Max mirror descent iterations for the reweighting step.
#'   Default: 1000.
#' @param tol Convergence tolerance: stop when max E_theta <= 1 + tol.
#'   Default: 1e-4.
#' @return List with:
#'   - `atoms`: list of atom probability vectors (all on null boundary).
#'   - `weights`: final mixture weights (sums to 1).
#'   - `atom_history`: list of per-iteration info (theta_stars, E_ratio, weights).
#'   - `converged`: TRUE if the validity condition was met.
#' @export
run_boundary_ripr <- function(
  n,
  q,
  n_atoms        = 50L,
  oracle_grid    = 200L,
  reweight_maxit = 1000L,
  tol            = 1e-4
) {
  K <- length(q)

  X_tensor <- build_counts_tensor(n, K)
  X_mat    <- matrix(as.integer(as.array(X_tensor$cpu())), ncol = K)

  log_base <- lgamma(n + 1) - rowSums(lgamma(X_mat + 1))
  log_multinom_all <- function(theta) {
    log_theta <- ifelse(theta > 0, log(theta), -Inf)
    log_base + as.vector(X_mat %*% log_theta)
  }

  compute_log_Pw <- function(log_comps, weights) {
    log_comps_w <- sweep(log_comps, 2L, log(weights), "+")
    row_max     <- apply(log_comps_w, 1L, max)
    result      <- rep(-Inf, nrow(log_comps))
    finite      <- is.finite(row_max)
    if (any(finite)) {
      lc <- log_comps_w[finite, , drop = FALSE]
      rm <- row_max[finite]
      result[finite] <- log(rowSums(exp(lc - rm))) + rm
    }
    result
  }

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

  # Gradient of E_theta[Q/P_w] with respect to theta via the score function.
  compute_E_ratio_grad_theta <- function(log_theta_mass, log_Pw, theta) {
    log_terms <- log_theta_mass + log_q_mass - log_Pw
    finite    <- is.finite(log_terms) & finite_q
    if (!any(finite)) return(rep(0, length(theta)))
    weights_x <- exp(log_terms[finite])
    score <- sweep(X_mat[finite, , drop = FALSE], 2L, theta, "/") - n
    as.vector(weights_x %*% score)
  }

  # Oracle for face j with analytic gradient via chain rule through
  # null_boundary_face and the softmax reparametrisation.
  find_best_on_face <- function(j, log_Pw) {
    d      <- K - 1L
    J      <- face_jacobian(K, d, j)

    from_v <- function(v) { u <- c(0, v); e <- exp(u - max(u)); e / sum(e) }
    to_v   <- function(alpha) log(alpha[-1L] + 1e-12) - log(alpha[1L] + 1e-12)

    softmax_jacobian <- function(alpha) {
      outer(alpha, alpha, function(a, b) -a * b) + diag(alpha)
    }[-1L, , drop = FALSE] |>
      (\(M) rbind(-colSums(M), M))()

    obj_and_grad <- function(v) {
      alpha      <- from_v(v)
      theta      <- null_boundary_face(j, alpha, K)
      log_tm     <- log_multinom_all(theta)
      E          <- compute_E_ratio(log_tm, log_Pw)
      grad_theta <- compute_E_ratio_grad_theta(log_tm, log_Pw, theta)
      grad_v     <- as.vector(grad_theta %*% J %*% softmax_jacobian(alpha))
      list(value = -E, gradient = -grad_v)
    }

    lat_density <- max(3L, round(oracle_grid ^ (1 / max(1L, K - 2L))))
    alpha_mat   <- simplex_lattice(d, lat_density) / lat_density
    E_grid      <- apply(alpha_mat, 1L, function(alpha)
      compute_E_ratio(log_multinom_all(null_boundary_face(j, alpha, K)), log_Pw))

    best_alpha <- pmax(alpha_mat[which.max(E_grid), ], 1e-8)
    best_alpha <- best_alpha / sum(best_alpha)

    res <- tryCatch(
      stats::optim(
        to_v(best_alpha),
        fn  = function(v) obj_and_grad(v)$value,
        gr  = function(v) obj_and_grad(v)$gradient,
        method = "BFGS"
      ),
      error = function(e) list(par = to_v(best_alpha), value = -max(E_grid, na.rm = TRUE))
    )

    alpha_star <- from_v(res$par)
    list(theta = null_boundary_face(j, alpha_star, K), E_ratio = -res$value)
  }

  # Mirror descent (multiplicative weights): w <- w * exp(-lr * grad) / Z.
  reweight_mirror <- function(log_comps, w_init, max_iter = reweight_maxit, tol = 1e-12) {
    w <- w_init / sum(w_init)
    for (i in seq_len(max_iter)) {
      lr     <- 1.0
      log_Pw <- compute_log_Pw(log_comps, w)
      lrs    <- sweep(log_comps[finite_q, , drop = FALSE], 1L, log_Pw[finite_q], "-")
      lrs[is.nan(lrs)] <- -Inf   # -Inf - (-Inf): atom zero where mixture zero => no contribution
      grad_w <- -colSums(q_mass[finite_q] * exp(lrs))
      loss   <- -sum(q_mass[finite_q] * log_Pw[finite_q])
      for (jj in seq_len(50L)) {
        w_new      <- w * exp(-lr * grad_w); w_new <- w_new / sum(w_new)
        log_Pw_new <- compute_log_Pw(log_comps, w_new)
        if (-sum(q_mass[finite_q] * log_Pw_new[finite_q]) <= loss) break
        lr <- lr * 0.5
      }
      if (max(abs(w_new - w)) < tol) break
      w <- w_new
    }
    w
  }

  # Initialise: centroid of each face (uniform alpha), giving full positive support.
  faces <- 2L:K
  atoms <- lapply(faces, function(j) {
    null_boundary_face(j, rep(1 / (K - 1L), K - 1L), K)
  })
  log_comps <- do.call(cbind, lapply(atoms, log_multinom_all))
  weights   <- reweight_mirror(log_comps, rep(1 / (K - 1L), K - 1L))
  converged <- FALSE
  history   <- vector("list", n_atoms)

  for (atom_idx in seq_len(n_atoms)) {
    log_Pw       <- compute_log_Pw(log_comps, weights)
    face_results <- lapply(faces, function(j) find_best_on_face(j, log_Pw))
    E_star       <- max(vapply(face_results, `[[`, "E_ratio", FUN.VALUE = numeric(1L)))

    message(sprintf("Atom %d/%d: max_E_ratio - 1 = %e", atom_idx, n_atoms, E_star - 1))

    history[[atom_idx]] <- list(
      theta_stars = lapply(face_results, `[[`, "theta"),
      E_ratio     = E_star,
      weights     = weights
    )

    if (E_star <= 1 + tol) {
      message(sprintf("Converged after %d atoms (max_E_ratio - 1 = %e).", length(atoms), E_star - 1))
      converged <- TRUE
      break
    }

    if (length(atoms) + (K - 1L) > n_atoms) break

    new_atoms <- lapply(face_results, `[[`, "theta")
    atoms     <- c(atoms, new_atoms)
    log_comps <- cbind(log_comps, do.call(cbind, lapply(new_atoms, log_multinom_all)))
    n_curr    <- ncol(log_comps)
    n_new     <- K - 1L
    w_init    <- c(weights * (n_curr - n_new) / n_curr, rep(1 / n_curr, n_new))
    weights   <- reweight_mirror(log_comps, w_init)
  }

  list(
    atoms        = atoms,
    weights      = weights,
    atom_history = history[!sapply(history, is.null)],
    converged    = converged
  )
}
