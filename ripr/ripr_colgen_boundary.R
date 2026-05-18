box::use(
  torch[
    torch_tensor,
    torch_full,
    torch_logsumexp,
    torch_lgamma
  ],
  ripr / torch_settings[device, dtype],
  ripr / tensor_ops[matmul_0_ninf, add_ninf_any_],
  ripr / multinomial[build_counts_tensor],
  ripr / mixture[mixture_mnom, log_pmf],
  ripr / grids[simplex_lattice]
)

# Initial atom on face j: project q onto face j by projection along the line from q to e_j.
init_atom_face <- function(j, q) {
  lambda <- (q[1L] - q[j]) / (1 + q[1L] - q[j])
  theta <- (1 - lambda) * q
  theta[j] <- theta[j] + lambda
  theta
}

# Enumerate vertices of face j: for each non-empty subset S of {2,...,K}\{j},
# the vertex has theta_1 = theta_j = theta_k = 1/(|S|+2) for k in S, others 0.
# Plus the pure-pair vertex (S = empty): theta_1 = theta_j = 1/2.
face_vertices <- function(j, K) {
  others <- setdiff(seq_len(K)[-1L], j)
  n_others <- length(others)
  # All 2^n_others subsets of others
  subsets <- lapply(0L:(2L^n_others - 1L), function(mask) {
    others[as.logical(intToBits(mask)[seq_len(n_others)])]
  })
  vertices <- lapply(subsets, function(S) {
    v <- numeric(K)
    s_size <- length(S)
    val <- 1 / (s_size + 2L)
    v[1L] <- val
    v[j] <- val
    for (k in S) {
      v[k] <- val
    }
    v
  })
  do.call(cbind, vertices) # K x 2^(K-2) matrix
}

# Softmax reparametrisation: maps v ∈ R^{d-1} to α ∈ Δ^{d-1} via softmax(c(0, v)).
# Used in the BFGS oracle to give an unconstrained parameterisation of each face.
alpha_from_v <- function(v) {
  u <- c(0, v)
  e <- exp(u - max(u))
  e / sum(e)
}

# Inverse of alpha_from_v: log-ratios of α relative to α[1], with eps guard.
v_from_alpha <- function(alpha, eps = 1e-12) {
  log(alpha[-1L] + eps) - log(alpha[1L] + eps)
}

# Jacobian of alpha_from_v(v): d(alpha)/d(v), a d × (d-1) matrix.
# Chain rule: full d×d softmax Jacobian times d(u)/d(v) = rbind(0, I_{d-1}),
# which is equivalent to dropping the first column of the full Jacobian.
softmax_jacobian <- function(alpha) {
  (outer(alpha, alpha, function(a, b) -a * b) + diag(alpha))[,
    -1L,
    drop = FALSE
  ]
}

# Stable in-place logsumexp along dim. buf is used as scratch and must not be read after.
# Avoids allocating a full (M, A) intermediate — only creates (M, 1) and (M,) tensors.
logsumexp_inplace_ <- function(buf, dim) {
  mx <- buf$amax(dim = dim, keepdim = TRUE)
  safe_mx <- mx$masked_fill_(mx$isneginf(), 0)
  buf$sub_(safe_mx)$exp_()
  buf$sum(dim = dim)$log_()$add_(safe_mx$squeeze(dim))
}


# Mirror descent (multiplicative weights) on the probability simplex.
# loss_and_grad: function(w) returning list(loss, grad) at the current weights.
reweight_mirror <- function(
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
#' @param n_em_iter Maximum EM refinement iterations after each Frank-Wolfe step.
#'   EM stops early when the KL decrease per iteration drops below `tol`.
#'   Set to 0 to disable EM. Default: 3.
#' @param tol Convergence tolerance: stop when max E_theta <= 1 + tol.
#'   Default: 1e-4.
#' @return List with:
#'   - `mixture`: a `mixture_mnom` with atoms on the null boundary and final weights.
#'   - `history`: list of per-iteration info (theta_stars, E_ratio, KL-divergence).
#'   - `converged`: TRUE if the validity condition was met.
#' @export
run_boundary_ripr <- function(
  n,
  q,
  atoms_per_face = 50L,
  oracle_grid = 200L,
  reweight_maxit = 1000L,
  n_em_iter = 3L,
  tol = 1e-4,
  verbose = TRUE
) {
  K <- nrow(q@atoms)
  n_faces <- K - 1L
  max_atoms <- atoms_per_face * n_faces

  # Precompute vertices for every face once
  face_vertices_cache <- lapply(2L:K, function(j) face_vertices(j, K))
  names(face_vertices_cache) <- as.character(2L:K)

  # Helper accessors
  get_vertices <- function(j) face_vertices_cache[[as.character(j)]]

  null_boundary_face <- function(j, alpha) {
    as.vector(get_vertices(j) %*% alpha)
  }

  compute_face_thetas <- function(j, alpha_mat) {
    get_vertices(j) %*% t(alpha_mat)
  }

  face_jacobian <- function(j) get_vertices(j)

  X_tensor <- build_counts_tensor(n, K)
  X_mat_gpu <- X_tensor$to(device = device, dtype = dtype)
  M <- X_mat_gpu$size(1L)

  log_base_gpu <- torch_lgamma(torch_tensor(
    n + 1L,
    device = device,
    dtype = dtype
  )) -
    torch_lgamma(X_mat_gpu + 1)$sum(dim = 2L) # (M,)

  # Single-theta log-PMF: R K-vector → (M,) GPU tensor.
  log_multinom_gpu <- function(theta) {
    lt <- torch_tensor(log(theta), device = device, dtype = dtype)
    matmul_0_ninf(X_mat_gpu, lt) + log_base_gpu
  }

  # Batched log-PMF: (K, N) R matrix → (M, N) GPU tensor.
  log_multinom_batch_gpu <- function(theta_mat) {
    lt <- torch_tensor(log(theta_mat), device = device, dtype = dtype)
    matmul_0_ninf(X_mat_gpu, lt) + log_base_gpu$unsqueeze(2L)
  }

  log_q_mass_gpu <- log_pmf(q, X_mat_gpu)
  q_mass_gpu <- log_q_mass_gpu$exp()
  finite_q_gpu <- q_mass_gpu > 0
  q_mass_f_gpu <- q_mass_gpu[finite_q_gpu]
  M_f <- q_mass_f_gpu$numel()

  # Pre-allocate fixed-size buffers for atom log-PMFs (full and finite-q slice).
  # Columns 1..n_live contain computed values; remaining columns stay -Inf.
  # buf_MA / buf_Mf are scratch space overwritten by compute_log_Pw_gpu / kl_loss_and_grad.
  lc <- torch_full(c(M, max_atoms), -Inf, device = device, dtype = dtype)
  lc_f <- torch_full(c(M_f, max_atoms), -Inf, device = device, dtype = dtype)
  buf_MA <- torch_full(c(M, max_atoms), -Inf, device = device, dtype = dtype)
  buf_Mf <- torch_full(c(M_f, max_atoms), -Inf, device = device, dtype = dtype)
  # Precomputed -Inf masks for add_ninf_any_ (avoids recomputing torch_isneginf each call).
  lc_ninf <- lc$isneginf() # (M,   max_atoms) bool, all TRUE initially
  lc_f_ninf <- lc_f$isneginf() # (M_f, max_atoms) bool, all TRUE initially
  n_live <- 0L

  write_atom_col <- function(k, theta) {
    lm <- log_multinom_gpu(theta)
    lc[, k] <<- lm
    lc_ninf[, k] <<- lm$isneginf()
    lm_f <- lm[finite_q_gpu]
    lc_f[, k] <<- lm_f
    lc_f_ninf[, k] <<- lm_f$isneginf()
  }

  # Append one atom in-place: fills the next column of lc/lc_f and their ninf masks.
  add_atom_col <- function(theta) {
    n_live <<- n_live + 1L
    write_atom_col(n_live, theta)
  }

  # log P_w(x) for all outcomes. Uses buf_MA as scratch — caller must not read buf_MA after.
  # Creates only (M, 1) and (M,) tensors; avoids the (M, A) temporary of naive logsumexp.
  compute_log_Pw_gpu <- function(w) {
    w_log <- torch_tensor(log(pmax(w, 1e-300)), device = device, dtype = dtype)
    buf_live <- buf_MA$narrow(2L, 1L, n_live)
    add_ninf_any_(
      buf_live,
      lc$narrow(2L, 1L, n_live),
      w_log$unsqueeze(1L),
      lc_ninf$narrow(2L, 1L, n_live)
    )
    logsumexp_inplace_(buf_live, dim = 2L)
  }

  # E_theta[Q/P_w]: R scalar.
  compute_E_ratio <- function(log_tm_gpu, log_Pw_gpu) {
    log_terms <- (log_tm_gpu + log_q_mass_gpu - log_Pw_gpu)$nan_to_num(
      nan = -Inf
    )
    torch_logsumexp(log_terms, dim = 1L)$exp()$item()
  }

  # Gradient of E_theta[Q/P_w] w.r.t. theta: R K-vector.
  compute_E_ratio_grad_theta <- function(log_tm_gpu, log_Pw_gpu, theta) {
    log_terms <- (log_tm_gpu + log_q_mass_gpu - log_Pw_gpu)$nan_to_num(
      nan = -Inf
    )
    weights_x <- log_terms$exp()
    theta_gpu <- torch_tensor(theta, device = device, dtype = dtype)
    score_gpu <- X_mat_gpu / theta_gpu$unsqueeze(1L) - n
    as.numeric(
      (weights_x$unsqueeze(2L) * score_gpu)$nan_to_num(nan = 0.0)$sum(
        dim = 1L
      )$cpu()
    )
  }

  # KL loss and gradient w.r.t. weights.
  # Uses buf_Mf as scratch via logsumexp_inplace_ — creates only (M_f, 1) and (M_f,) tensors.
  kl_loss_and_grad <- function(w) {
    log_Pw <- compute_log_Pw_gpu(w)
    log_Pw_f <- log_Pw[finite_q_gpu]

    buf_f_live <- buf_Mf$narrow(2L, 1L, n_live)
    add_ninf_any_(
      buf_f_live,
      lc_f$narrow(2L, 1L, n_live),
      (-log_Pw_f)$unsqueeze(2L),
      lc_f_ninf$narrow(2L, 1L, n_live)
    )
    buf_f_live$exp_()
    buf_f_live$mul_(q_mass_f_gpu$unsqueeze(2L))

    list(
      loss = -(q_mass_f_gpu * log_Pw_f)$nan_to_num_(nan = 0.0)$sum()$item(),
      grad = as.numeric(-buf_f_live$sum(dim = 1L)$cpu())
    )
  }

  # Run a coarse grid search, and refine with BFGS on a convex polytope.
  # obj_and_grad: function(v) returning list(value, gradient) where value is the objective function to be minimised.
  # obj_grid_eval: function(alpha_mat) returning numeric vector of negative objectives over grid rows.
  # Returns list(alpha_star, neg_value).
  optimise_on_face <- function(j, obj_and_grad, obj_grid_eval) {
    J <- face_jacobian(j)
    n_vertices <- ncol(J)
    lat_density <- max(3L, round(oracle_grid^(1 / max(1L, n_vertices - 1L))))
    alpha_mat <- simplex_lattice(n_vertices, lat_density) / lat_density

    neg_obj_grid <- obj_grid_eval(alpha_mat)
    best_alpha <- pmax(alpha_mat[which.max(neg_obj_grid), ], 1e-8)
    best_alpha <- best_alpha / sum(best_alpha)

    res <- tryCatch(
      stats::optim(
        v_from_alpha(best_alpha),
        fn = function(v) obj_and_grad(v)$value,
        gr = function(v) obj_and_grad(v)$gradient,
        method = "BFGS"
      ),
      error = function(e) {
        list(
          par = v_from_alpha(best_alpha),
          value = -max(neg_obj_grid, na.rm = TRUE)
        )
      }
    )

    list(alpha_star = alpha_from_v(res$par), neg_value = res$value)
  }

  # Oracle for face j with analytic gradient via chain rule through
  # null_boundary_face and the softmax reparametrisation.
  find_best_on_face <- function(j, log_Pw_gpu) {
    J <- face_jacobian(j)

    obj_and_grad <- function(v) {
      alpha <- alpha_from_v(v)
      theta <- null_boundary_face(j, alpha)
      log_tm <- log_multinom_gpu(theta)
      E <- compute_E_ratio(log_tm, log_Pw_gpu)
      grad_theta <- compute_E_ratio_grad_theta(log_tm, log_Pw_gpu, theta)
      grad_v <- as.vector(grad_theta %*% J %*% softmax_jacobian(alpha))
      list(value = -E, gradient = -grad_v)
    }

    obj_grid_eval <- function(alpha_mat) {
      log_tm_batch <- log_multinom_batch_gpu(compute_face_thetas(j, alpha_mat))
      log_terms <- (log_tm_batch +
        log_q_mass_gpu$unsqueeze(2L) -
        log_Pw_gpu$unsqueeze(2L))$nan_to_num(nan = -Inf)
      as.numeric(torch_logsumexp(log_terms, dim = 1L)$exp()$cpu())
    }

    res <- optimise_on_face(j, obj_and_grad, obj_grid_eval)
    list(
      theta = null_boundary_face(j, res$alpha_star),
      E_ratio = -res$neg_value,
      face = j
    )
  }

  # Weighted M-step location update: maximise sum_x q(x) r_k(x) log p_theta(x)
  # over theta on face j.  This is the EM M-step, NOT a weighted Frank-Wolfe oracle.
  find_best_on_face_weighted <- function(j, log_r_k) {
    J <- face_jacobian(j)
    log_weights <- log_q_mass_gpu + log_r_k
    weights_x <- log_weights$exp()$nan_to_num(nan = 0.0) # (M,)

    obj_and_grad <- function(v) {
      alpha <- alpha_from_v(v)
      theta <- null_boundary_face(j, alpha)
      log_tm <- log_multinom_gpu(theta)
      obj_val <- (weights_x * log_tm)$nan_to_num(nan = 0.0)$sum()$item()
      theta_gpu <- torch_tensor(theta, device = device, dtype = dtype)
      score_gpu <- X_mat_gpu / theta_gpu$unsqueeze(1L)
      grad_theta <- as.numeric(
        (weights_x$unsqueeze(2L) * score_gpu)$nan_to_num(nan = 0.0)$sum(
          dim = 1L
        )$cpu()
      )
      grad_v <- as.vector(grad_theta %*% J %*% softmax_jacobian(alpha))
      list(value = -obj_val, gradient = -grad_v)
    }

    obj_grid_eval <- function(alpha_mat) {
      log_tm_batch <- log_multinom_batch_gpu(compute_face_thetas(j, alpha_mat))
      as.numeric(
        (weights_x$unsqueeze(2L) * log_tm_batch)$nan_to_num(nan = 0.0)$sum(
          dim = 1L
        )$cpu()
      )
    }

    res <- optimise_on_face(j, obj_and_grad, obj_grid_eval)
    list(theta = null_boundary_face(j, res$alpha_star))
  }

  # E-step: log-responsibilities log r_k(x) = log w_k + log p_{theta_k}(x) - log P_w(x).
  # Returns (M, n_live) tensor of normalised log-responsibilities.
  compute_responsibilities <- function(log_Pw_gpu, weights) {
    w_log <- torch_tensor(
      log(pmax(weights, 1e-300)),
      device = device,
      dtype = dtype
    )
    lc_live <- lc$narrow(2L, 1L, n_live)
    log_r <- lc_live + w_log$unsqueeze(1L) - log_Pw_gpu$unsqueeze(2L)
    log_r$nan_to_num_(nan = -Inf)
    log_r - torch_logsumexp(log_r, dim = 2L, keepdim = TRUE)
  }

  frank_wolfe_oracle <- function(weights) {
    log_Pw_gpu <- compute_log_Pw_gpu(weights)
    face_results <- lapply(faces, function(j) find_best_on_face(j, log_Pw_gpu))
    list(
      face_results = face_results,
      E_star = max(vapply(
        face_results,
        `[[`,
        "E_ratio",
        FUN.VALUE = numeric(1L)
      ))
    )
  }

  run_em_step <- function(weights, n_em_iter, kl_init) {
    kl <- kl_init
    for (em_idx in seq_len(n_em_iter)) {
      log_Pw_gpu <- compute_log_Pw_gpu(weights)
      log_r <- compute_responsibilities(log_Pw_gpu, weights)

      new_weights <- (log_r$exp() * q_mass_gpu$unsqueeze(2L))$sum(dim = 1L)
      new_weights <- pmax(as.numeric(new_weights$cpu()), 1e-300)
      new_weights <- new_weights / sum(new_weights)

      for (k in seq_len(n_live)) {
        result <- find_best_on_face_weighted(atom_faces[[k]], log_r[, k])
        write_atom_col(k, result$theta)
        atoms[[k]] <<- result$theta
      }

      weights <- new_weights
      kl_new <- kl_loss_and_grad(weights)$loss
      if (kl - kl_new < tol) {
        break
      }
      kl <- kl_new
    }
    list(weights = weights, kl = kl)
  }

  if (verbose) {
    message(sprintf(
      "run_boundary_ripr: n=%d, K=%d, M=%d outcomes, %d atoms/face, %d faces, tol=%g",
      n,
      K,
      M,
      atoms_per_face,
      n_faces,
      tol
    ))
  }

  faces <- 2L:K
  # Initialise by projecting the mode of q onto each face.
  q_max <- q@atoms[, which.max(q@weights)]
  atoms <- lapply(faces, function(j) init_atom_face(j, q_max))
  atom_faces <- as.list(faces)
  for (th in atoms) {
    add_atom_col(th)
  }

  weights <- rep(1 / n_faces, n_faces)
  converged <- FALSE
  history <- vector("list", atoms_per_face)
  kl_prev <- NULL

  for (atom_idx in seq_len(atoms_per_face)) {
    # Frank-Wolfe step: add new approximate best atom, then reweight all atoms via mirror descent to minimise KL.
    weights <- reweight_mirror(
      weights,
      kl_loss_and_grad,
      max_iter = reweight_maxit
    )
    em_result <- run_em_step(weights, n_em_iter, kl_loss_and_grad(weights)$loss)
    weights <- em_result$weights
    kl <- em_result$kl

    fw <- frank_wolfe_oracle(weights)
    face_results <- fw$face_results
    E_star <- fw$E_star

    history[[atom_idx]] <- list(
      theta_stars = lapply(face_results, `[[`, "theta"),
      E_ratio = E_star,
      kl = kl
    )

    if (verbose) {
      if (is.null(kl_prev)) {
        message(sprintf(
          "Atom %d/%d: max_E_ratio - 1 = %e, KL = %e",
          atom_idx,
          atoms_per_face,
          E_star - 1,
          kl
        ))
      } else {
        message(sprintf(
          "Atom %d/%d: max_E_ratio - 1 = %e, KL = %e, delta_KL = %e",
          atom_idx,
          atoms_per_face,
          E_star - 1,
          kl,
          kl - kl_prev
        ))
      }
    }
    kl_prev <- kl

    if (E_star <= 1 + tol) {
      converged <- TRUE
      break
    }

    # Prepare new atoms for next iteration
    if (atom_idx < atoms_per_face) {
      new_atoms <- lapply(face_results, `[[`, "theta")
      new_faces <- as.list(faces)
      atoms <- c(atoms, new_atoms)
      atom_faces <- c(atom_faces, new_faces)
      for (th in new_atoms) {
        add_atom_col(th)
      }
      n_curr <- n_live
      weights <- c(
        weights * (n_curr - n_faces) / n_curr,
        rep(1 / n_curr, n_faces)
      )
    }
  }

  if (converged && verbose) {
    message(sprintf(
      "Converged after %d atoms (max_E_ratio - 1 = %e).",
      n_live,
      E_star - 1
    ))
  }

  list(
    mixture = mixture_mnom(
      atoms = do.call(cbind, atoms),
      weights = weights,
      n = n
    ),
    history = history[!sapply(history, is.null)],
    converged = converged
  )
}
