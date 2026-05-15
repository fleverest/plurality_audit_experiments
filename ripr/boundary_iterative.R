box::use(
  torch[
    torch_tensor,
    torch_full,
    torch_logsumexp
  ],
  ripr / torch_settings[device, dtype],
  ripr / tensor_ops[matmul_0_ninf, add_ninf_any_],
  ripr / multinomial[build_counts_tensor],
  ripr / mixture[mixture_mnom, log_pmf]
)

# Initial atom on face j: project q onto face j by projection along the line from q to e_j.
init_atom_face <- function(j, q) {
  lambda    <- (q[1L] - q[j]) / (1 + q[1L] - q[j])
  theta     <- (1 - lambda) * q
  theta[j]  <- theta[j] + lambda
  theta
}

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

# Vectorised null_boundary_face over all rows of alpha_mat.
# alpha_mat: (N, K-1); returns (K, N) matrix of theta vectors.
compute_face_thetas <- function(j, alpha_mat, K) {
  others    <- setdiff(seq_len(K)[-1L], j)
  N         <- nrow(alpha_mat)
  theta_mat <- matrix(0, K, N)
  theta_mat[1L, ] <- alpha_mat[, 1L] / 6 + 1 / 3
  theta_mat[j,  ] <- theta_mat[1L, ]
  for (r in seq_along(others)) theta_mat[others[r], ] <- alpha_mat[, r + 1L] / 3
  theta_mat
}

# Jacobian of null_boundary_face(j, alpha, K) w.r.t. alpha.
# theta[1] = theta[j] = alpha[1]/6 + 1/3,  theta[others[r]] = alpha[r+1]/3.
# d theta / d alpha is a K x (K-1) matrix.
face_jacobian <- function(K, j) {
  others <- setdiff(seq_len(K), c(1L, j))
  J <- matrix(0, K, K - 1L)
  J[1L,  1L] <- 1 / 6
  J[j,   1L] <- 1 / 6
  for (r in seq_along(others)) J[others[r], r + 1L] <- 1 / 3
  J
}

# Softmax reparametrisation: maps v ∈ R^{d-1} to α ∈ Δ^{d-1} via softmax(c(0, v)).
# Used in the BFGS oracle to give an unconstrained parameterisation of each face.
alpha_from_v <- function(v) {
  u <- c(0, v); e <- exp(u - max(u)); e / sum(e)
}

# Inverse of alpha_from_v: log-ratios of α relative to α[1], with eps guard.
v_from_alpha <- function(alpha, eps = 1e-12) {
  log(alpha[-1L] + eps) - log(alpha[1L] + eps)
}

# Jacobian of alpha_from_v(v): d(alpha)/d(v), a d × (d-1) matrix.
# Chain rule: full d×d softmax Jacobian times d(u)/d(v) = rbind(0, I_{d-1}),
# which is equivalent to dropping the first column of the full Jacobian.
softmax_jacobian <- function(alpha) {
  (outer(alpha, alpha, function(a, b) -a * b) + diag(alpha))[, -1L, drop = FALSE]
}

# Stable in-place logsumexp along dim. buf is used as scratch and must not be read after.
# Avoids allocating a full (M, A) intermediate — only creates (M, 1) and (M,) tensors.
logsumexp_inplace_ <- function(buf, dim) {
  mx      <- buf$amax(dim = dim, keepdim = TRUE)
  safe_mx <- mx$masked_fill_(mx$isneginf(), 0)
  buf$sub_(safe_mx)$exp_()
  buf$sum(dim = dim)$log_()$add_(safe_mx$squeeze(dim))
}

# Mirror descent (multiplicative weights) on the probability simplex.
# loss_and_grad: function(w) returning list(loss, grad) at the current weights.
reweight_mirror <- function(w_init, loss_and_grad, max_iter = 1000L, tol = 1e-12) {
  w <- w_init / sum(w_init)
  for (i in seq_len(max_iter)) {
    lr     <- 1.0
    lg     <- loss_and_grad(w)
    loss   <- lg$loss
    grad_w <- lg$grad
    for (j in seq_len(50L)) {
      w_new <- w * exp(-lr * grad_w); w_new <- w_new / sum(w_new)
      if (loss_and_grad(w_new)$loss <= loss) break
      lr <- lr * 0.5
    }
    if (max(abs(w_new - w)) < tol) break
    w <- w_new
  }
  w
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
#' Heavy computation (log-PMF evaluation, KL gradient) runs on the torch device
#' from [ripr/torch_settings]; the BFGS oracle and mirror descent remain in R.
#' GPU memory is bounded: atom log-PMFs are stored in fixed-size pre-allocated
#' buffers filled in-place, and the KL hot path avoids large intermediate tensors
#' via [logsumexp_inplace_()].
#'
#' @param n Total ballot count.
#' @param q A `mixture_mnom` — the numerator distribution Q. Use [point_mnom()]
#'   for a single point alternative or [dirichlet_mnom()] for a grid-weighted
#'   Dirichlet prior over H_1. K is inferred from `nrow(q@atoms)`.
#' @param atoms_per_face Number of atoms to add per boundary face. Default: 50.
#' @param oracle_grid Grid density per simplex dimension for the boundary oracle.
#'   Total lattice points per face ~ oracle_grid^(K-2); reduce for K >= 4.
#'   Default: 200.
#' @param reweight_maxit Max mirror descent iterations for the reweighting step.
#'   Default: 1000.
#' @param tol Convergence tolerance: stop when max E_theta <= 1 + tol.
#'   Default: 1e-4.
#' @return List with:
#'   - `mixture`: a `mixture_mnom` with atoms on the null boundary and final weights.
#'   - `atom_history`: list of per-iteration info (theta_stars, E_ratio, weights).
#'   - `converged`: TRUE if the validity condition was met.
#' @export
run_boundary_ripr <- function(
  n,
  q,
  atoms_per_face = 50L,
  oracle_grid    = 200L,
  reweight_maxit = 1000L,
  tol            = 1e-4
) {
  K         <- nrow(q@atoms)
  n_faces   <- K - 1L
  max_atoms <- atoms_per_face * n_faces

  X_tensor  <- build_counts_tensor(n, K)
  X_mat_gpu <- X_tensor$to(device = device, dtype = dtype)
  X_mat_r   <- matrix(as.integer(as.array(X_tensor$cpu())), ncol = K)
  M         <- X_mat_gpu$size(1L)

  log_base_gpu <- torch_tensor(
    lgamma(n + 1) - rowSums(lgamma(X_mat_r + 1)),
    device = device, dtype = dtype
  )

  # Single-theta log-PMF: R K-vector → (M,) GPU tensor.
  log_multinom_gpu <- function(theta) {
    lt <- torch_tensor(ifelse(theta > 0, log(theta), -Inf), device = device, dtype = dtype)
    matmul_0_ninf(X_mat_gpu, lt$unsqueeze(2L))$squeeze(2L) + log_base_gpu
  }

  # Batched log-PMF: (K, N) R matrix → (M, N) GPU tensor.
  log_multinom_batch_gpu <- function(theta_mat) {
    lt <- torch_tensor(ifelse(theta_mat > 0, log(theta_mat), -Inf), device = device, dtype = dtype)
    matmul_0_ninf(X_mat_gpu, lt) + log_base_gpu$unsqueeze(2L)
  }

  log_q_mass_gpu <- log_pmf(q, X_mat_gpu)
  q_mass_gpu     <- log_q_mass_gpu$exp()
  finite_q_gpu   <- q_mass_gpu > 0
  q_mass_f_gpu   <- q_mass_gpu[finite_q_gpu]
  M_f            <- q_mass_f_gpu$numel()

  # Pre-allocate fixed-size buffers for atom log-PMFs (full and finite-q slice).
  # Columns 1..n_live contain computed values; remaining columns stay -Inf.
  # buf_MA / buf_Mf are scratch space overwritten by compute_log_Pw_gpu / kl_loss_and_grad.
  lc     <- torch_full(c(M,   max_atoms), -Inf, device = device, dtype = dtype)
  lc_f   <- torch_full(c(M_f, max_atoms), -Inf, device = device, dtype = dtype)
  buf_MA <- torch_full(c(M,   max_atoms), -Inf, device = device, dtype = dtype)
  buf_Mf <- torch_full(c(M_f, max_atoms), -Inf, device = device, dtype = dtype)
  # Precomputed -Inf masks for add_ninf_any_ (avoids recomputing torch_isneginf each call).
  lc_ninf   <- lc$isneginf()    # (M,   max_atoms) bool, all TRUE initially
  lc_f_ninf <- lc_f$isneginf()  # (M_f, max_atoms) bool, all TRUE initially
  n_live    <- 0L

  # Append one atom in-place: fills the next column of lc/lc_f and their ninf masks.
  add_atom_col <- function(theta) {
    lm     <- log_multinom_gpu(theta)
    n_live <<- n_live + 1L
    lc[,     n_live] <- lm
    lc_ninf[, n_live] <- lm$isneginf()
    lm_f              <- lm[finite_q_gpu]
    lc_f[,    n_live] <- lm_f
    lc_f_ninf[, n_live] <- lm_f$isneginf()
  }

  # log P_w(x) for all outcomes. Uses buf_MA as scratch — caller must not read buf_MA after.
  # Creates only (M, 1) and (M,) tensors; avoids the (M, A) temporary of naive logsumexp.
  compute_log_Pw_gpu <- function(w) {
    w_log    <- torch_tensor(log(pmax(w, 1e-300)), device = device, dtype = dtype)
    buf_live <- buf_MA$narrow(2L, 1L, n_live)
    add_ninf_any_(buf_live, lc$narrow(2L, 1L, n_live), w_log$unsqueeze(1L),
                  lc_ninf$narrow(2L, 1L, n_live))
    logsumexp_inplace_(buf_live, dim = 2L)
  }

  # E_theta[Q/P_w]: R scalar.
  compute_E_ratio <- function(log_tm_gpu, log_Pw_gpu) {
    log_terms <- (log_tm_gpu + log_q_mass_gpu - log_Pw_gpu)$nan_to_num(nan = -Inf)
    torch_logsumexp(log_terms, dim = 1L)$exp()$item()
  }

  # Gradient of E_theta[Q/P_w] w.r.t. theta: R K-vector.
  compute_E_ratio_grad_theta <- function(log_tm_gpu, log_Pw_gpu, theta) {
    log_terms <- (log_tm_gpu + log_q_mass_gpu - log_Pw_gpu)$nan_to_num(nan = -Inf)
    weights_x <- log_terms$exp()
    theta_gpu <- torch_tensor(theta, device = device, dtype = dtype)
    score_gpu <- X_mat_gpu / theta_gpu$unsqueeze(1L) - n
    as.numeric(
      (weights_x$unsqueeze(2L) * score_gpu)$nan_to_num(nan = 0.0)$sum(dim = 1L)$cpu()
    )
  }

  # KL loss and gradient w.r.t. weights.
  # Uses buf_Mf as scratch via logsumexp_inplace_ — creates only (M_f, 1) and (M_f,) tensors.
  kl_loss_and_grad <- function(w) {
    log_Pw   <- compute_log_Pw_gpu(w)
    log_Pw_f <- log_Pw[finite_q_gpu]

    buf_f_live <- buf_Mf$narrow(2L, 1L, n_live)
    add_ninf_any_(buf_f_live, lc_f$narrow(2L, 1L, n_live), (-log_Pw_f)$unsqueeze(2L),
                  lc_f_ninf$narrow(2L, 1L, n_live))
    buf_f_live$exp_()
    buf_f_live$mul_(q_mass_f_gpu$unsqueeze(2L))

    list(
      loss = -(q_mass_f_gpu * log_Pw_f)$nan_to_num_(nan = 0.0)$sum()$item(),
      grad = as.numeric(-buf_f_live$sum(dim = 1L)$cpu())
    )
  }

  # Oracle for face j with analytic gradient via chain rule through
  # null_boundary_face and the softmax reparametrisation.
  find_best_on_face <- function(j, log_Pw_gpu) {
    J <- face_jacobian(K, j)

    obj_and_grad <- function(v) {
      alpha      <- alpha_from_v(v)
      theta      <- null_boundary_face(j, alpha, K)
      log_tm     <- log_multinom_gpu(theta)
      E          <- compute_E_ratio(log_tm, log_Pw_gpu)
      grad_theta <- compute_E_ratio_grad_theta(log_tm, log_Pw_gpu, theta)
      grad_v     <- as.vector(grad_theta %*% J %*% softmax_jacobian(alpha))
      list(value = -E, gradient = -grad_v)
    }

    lat_density  <- max(3L, round(oracle_grid ^ (1 / max(1L, K - 2L))))
    alpha_mat    <- simplex_lattice(K - 1L, lat_density) / lat_density

    # Batch-evaluate E_ratio for all grid points in one GPU call.
    log_tm_batch <- log_multinom_batch_gpu(compute_face_thetas(j, alpha_mat, K))
    log_terms    <- (log_tm_batch + log_q_mass_gpu$unsqueeze(2L) - log_Pw_gpu$unsqueeze(2L))$nan_to_num(nan = -Inf)
    E_grid       <- as.numeric(torch_logsumexp(log_terms, dim = 1L)$exp()$cpu())
    rm(log_tm_batch, log_terms)

    best_alpha <- pmax(alpha_mat[which.max(E_grid), ], 1e-8)
    best_alpha <- best_alpha / sum(best_alpha)

    res <- tryCatch(
      stats::optim(
        v_from_alpha(best_alpha),
        fn  = function(v) obj_and_grad(v)$value,
        gr  = function(v) obj_and_grad(v)$gradient,
        method = "BFGS"
      ),
      error = function(e) list(par = v_from_alpha(best_alpha), value = -max(E_grid, na.rm = TRUE))
    )

    alpha_star <- alpha_from_v(res$par)
    list(theta = null_boundary_face(j, alpha_star, K), E_ratio = -res$value)
  }

  faces  <- 2L:K
  q_mean <- as.vector(q@atoms %*% q@weights)
  atoms  <- lapply(faces, function(j) init_atom_face(j, q_mean))
  for (th in atoms) add_atom_col(th)

  weights   <- reweight_mirror(rep(1 / n_faces, n_faces), kl_loss_and_grad,
                               max_iter = reweight_maxit)
  converged <- FALSE
  history   <- vector("list", atoms_per_face)

  for (atom_idx in seq_len(atoms_per_face - 1L)) {
    log_Pw_gpu   <- compute_log_Pw_gpu(weights)
    face_results <- lapply(faces, function(j) find_best_on_face(j, log_Pw_gpu))
    E_star       <- max(vapply(face_results, `[[`, "E_ratio", FUN.VALUE = numeric(1L)))

    message(sprintf("Atom %d/%d: max_E_ratio - 1 = %e", atom_idx, atoms_per_face, E_star - 1))

    history[[atom_idx]] <- list(
      theta_stars = lapply(face_results, `[[`, "theta"),
      E_ratio     = E_star,
      weights     = weights
    )

    if (E_star <= 1 + tol) {
      message(sprintf("Converged after %d atoms (max_E_ratio - 1 = %e).", n_live, E_star - 1))
      converged <- TRUE
      break
    }

    new_atoms <- lapply(face_results, `[[`, "theta")
    atoms     <- c(atoms, new_atoms)
    for (th in new_atoms) add_atom_col(th)

    n_curr <- n_live
    w_init <- c(weights * (n_curr - n_faces) / n_curr, rep(1 / n_curr, n_faces))
    weights <- reweight_mirror(w_init, kl_loss_and_grad, max_iter = reweight_maxit)
  }

  list(
    mixture = mixture_mnom(
      atoms = do.call(cbind, atoms),
      weights = weights
    ),
    history = history[!sapply(history, is.null)],
    converged = converged
  )
}
