box::use(
  torch[
    torch_tensor,
    torch_full,
    torch_logsumexp,
    torch_maximum
  ],
  ripr / torch_settings[device, dtype],
  ripr / tensor_ops[add_ninf_any_, logsumexp_inplace_],
  ripr / mixture[log_pmf, n_categories],
  ripr /
    simplex_utils[
      alpha_from_v,
      v_from_alpha,
      softmax_jacobian,
      simplex_lattice
    ]
)

# =============================================================================
# Pure helpers (no closure state)
# =============================================================================

#' Weighted score sum: sum_x w(x) * score_theta(x)
#'
#' Used by both the Frank-Wolfe oracle gradient and the EM M-step gradient.
#' `weights_x` and `score_gpu` may contain NaNs (e.g. 0 * -Inf); they are
#' zeroed before summing.
weighted_score_sum <- function(weights_x, score_gpu) {
  as.numeric(
    (weights_x$unsqueeze(2L) * score_gpu)$nan_to_num(nan = 0.0)$sum(
      dim = 1L
    )$cpu()
  )
}

#' E_theta[Q / P_w] (the Frank-Wolfe objective value)
#' @export
e_ratio <- function(log_tm_gpu, log_q_mass_gpu, log_Pw_gpu) {
  log_terms <- (log_tm_gpu + log_q_mass_gpu - log_Pw_gpu)$nan_to_num(nan = -Inf)
  torch_logsumexp(log_terms, dim = 1L)$exp()$item()
}

#' Gradient of E_theta[Q / P_w] with respect to theta
e_ratio_grad_theta <- function(
  log_tm_gpu,
  log_q_mass_gpu,
  log_Pw_gpu,
  theta,
  likelihood
) {
  log_terms <- (log_tm_gpu + log_q_mass_gpu - log_Pw_gpu)$nan_to_num(nan = -Inf)
  weights_x <- log_terms$exp()
  weighted_score_sum(weights_x, likelihood$score(theta))
}

#' Generic per-face optimiser: simplex-lattice grid search + BFGS refinement.
#'
#' `obj_and_grad(v)` is the BFGS objective in the unconstrained softmax
#' parametrisation `v`. `obj_grid_eval(alpha_mat)` evaluates the negative
#' objective at a batch of simplex points (used to pick BFGS seeds).
#' `oracle_grid` controls the density of the simplex lattice used for seed selection.
#' `seed_alpha` can be set to skip the global search and just run BFGS from a
#' single initial point.
optimise_on_face <- function(
  fd,
  obj_and_grad,
  obj_grid_eval,
  oracle_grid = NULL,
  n_restarts = 25L,
  seed_alpha = NULL
) {
  n_vertices <- fd$n_vertices

  # Local-mode early return
  if (!is.null(seed_alpha)) {
    init_alpha <- pmax(seed_alpha, 1e-8)
    init_alpha <- init_alpha / sum(init_alpha)
    res <- tryCatch(
      stats::optim(
        v_from_alpha(init_alpha),
        fn = function(v) obj_and_grad(v)$value,
        gr = function(v) obj_and_grad(v)$gradient,
        method = "BFGS"
      ),
      error = function(e) {
        list(par = v_from_alpha(init_alpha), value = Inf)
      }
    )
    return(list(alpha_star = alpha_from_v(res$par), neg_value = res$value))
  }

  # Original global-mode body, unchanged:
  lat_density <- max(3L, round(oracle_grid^(1 / max(1L, n_vertices - 1L))))
  alpha_mat <- simplex_lattice(n_vertices, lat_density) / lat_density

  neg_obj_grid <- obj_grid_eval(alpha_mat)
  top_idx <- order(neg_obj_grid, decreasing = TRUE)[seq_len(min(
    n_restarts,
    length(neg_obj_grid)
  ))]

  best <- list(par = NULL, value = Inf)
  for (idx in top_idx) {
    init_alpha <- pmax(alpha_mat[idx, ], 1e-8)
    init_alpha <- init_alpha / sum(init_alpha)
    res <- tryCatch(
      stats::optim(
        v_from_alpha(init_alpha),
        fn = function(v) obj_and_grad(v)$value,
        gr = function(v) obj_and_grad(v)$gradient,
        method = "BFGS"
      ),
      error = function(e) {
        list(par = v_from_alpha(init_alpha), value = -neg_obj_grid[idx])
      }
    )
    if (res$value < best$value) best <- res
  }
  list(alpha_star = alpha_from_v(best$par), neg_value = best$value)
}

#' Wrap a (value, grad_theta) pair into the BFGS objective expected by
#' `optimise_on_face`.
#'
#' Handles the softmax + face-parametrisation chain rule so callers only
#' supply the pointwise value and theta-gradient.
make_face_objective <- function(fd, value_fn, grad_theta_fn) {
  J <- fd$jacobian

  # Cache last input output pair for potential reuse since optim calls fn and gr separately.
  last_v <- NULL
  last_result <- NULL
  function(v) {
    if (!is.null(last_v) && identical(v, last_v)) {
      return(last_result)
    }
    alpha <- alpha_from_v(v)
    theta <- fd$parametrise(alpha)
    val <- value_fn(theta)
    grad_theta <- grad_theta_fn(theta)
    grad_v <- as.vector(grad_theta %*% J %*% softmax_jacobian(alpha))

    last_v <<- v
    last_result <<- list(value = -val, gradient = -grad_v)
    last_result
  }
}

#' Frank-Wolfe oracle on one face: argmax_theta E_theta[Q / P_w].
fw_oracle_face <- function(
  fd,
  likelihood,
  log_q_mass_gpu,
  log_Pw_gpu,
  oracle_grid
) {
  value_fn <- function(theta) {
    e_ratio(likelihood$log_pmf(theta), log_q_mass_gpu, log_Pw_gpu)
  }
  grad_theta_fn <- function(theta) {
    e_ratio_grad_theta(
      likelihood$log_pmf(theta),
      log_q_mass_gpu,
      log_Pw_gpu,
      theta,
      likelihood
    )
  }
  obj_and_grad <- make_face_objective(fd, value_fn, grad_theta_fn)

  obj_grid_eval <- function(alpha_mat) {
    log_tm_batch <- likelihood$log_pmf_batch(fd$parametrise_batch(alpha_mat))
    log_terms <- (log_tm_batch +
      log_q_mass_gpu$unsqueeze(2L) -
      log_Pw_gpu$unsqueeze(2L))$nan_to_num(nan = -Inf)
    as.numeric(torch_logsumexp(log_terms, dim = 1L)$exp()$cpu())
  }

  res <- optimise_on_face(
    fd,
    obj_and_grad,
    obj_grid_eval,
    oracle_grid = oracle_grid
  )
  list(theta = fd$parametrise(res$alpha_star), E_ratio = -res$neg_value)
}

#' Frank-Wolfe oracle over all faces: returns the most adversarial atom and
#' the precomputed log P_w (reused downstream for line search).
fw_oracle <- function(
  face_descriptors,
  likelihood,
  log_q_mass_gpu,
  log_Pw_gpu,
  oracle_grid
) {
  face_results <- lapply(face_descriptors, function(fd) {
    fw_oracle_face(fd, likelihood, log_q_mass_gpu, log_Pw_gpu, oracle_grid)
  })
  E_ratios <- vapply(face_results, `[[`, "E_ratio", FUN.VALUE = numeric(1L))
  best_fi <- which.max(E_ratios)
  list(
    face_results = face_results,
    E_star = E_ratios[best_fi],
    best_theta = face_results[[best_fi]]$theta,
    best_fi = best_fi
  )
}

#' EM M-step location update on one face:
#'   argmax_theta sum_x q(x) r_k(x) log p_theta(x).
em_mstep_face <- function(
  fd,
  log_r_k,
  likelihood,
  log_q_mass_gpu,
  seed_alpha
) {
  log_weights <- log_q_mass_gpu + log_r_k
  weights_x <- log_weights$exp()$nan_to_num(nan = 0.0)

  value_fn <- function(theta) {
    (weights_x * likelihood$log_pmf(theta))$nan_to_num(nan = 0.0)$sum()$item()
  }
  grad_theta_fn <- function(theta) {
    # score = x_l / theta_l - n; the constant n drops out under the weighted
    # sum only if sum(weights_x) is constant, which it isn't, so we use the
    # full per-outcome score including +n.
    weighted_score_sum(weights_x, likelihood$score(theta) + likelihood$n)
  }
  obj_and_grad <- make_face_objective(fd, value_fn, grad_theta_fn)

  obj_grid_eval <- function(alpha_mat) {
    log_tm_batch <- likelihood$log_pmf_batch(fd$parametrise_batch(alpha_mat))
    as.numeric(
      (weights_x$unsqueeze(2L) * log_tm_batch)$nan_to_num(nan = 0.0)$sum(
        dim = 1L
      )$cpu()
    )
  }

  res <- optimise_on_face(
    fd,
    obj_and_grad,
    obj_grid_eval,
    seed_alpha = seed_alpha
  )
  list(
    theta = fd$parametrise(res$alpha_star),
    obj = -res$neg_value,
    weights_x = weights_x
  )
}

#' Optimal mixing weight eps in [0, 1] for adding a new atom to P_w.
line_search <- function(log_Pw_gpu, log_tm_new, q_mass_gpu) {
  m <- torch_maximum(log_Pw_gpu, log_tm_new)
  exp_a <- (log_Pw_gpu - m)$exp()
  exp_b <- (log_tm_new - m)$exp()
  g <- function(eps) {
    log_mix <- m + ((1 - eps) * exp_a + eps * exp_b)$log()
    -(q_mass_gpu * log_mix)$nan_to_num(nan = 0.0)$sum()$item()
  }
  stats::optimize(g, interval = c(1e-10, 1 - 1e-10))$minimum
}

# =============================================================================
# Workspace: closures over the pre-allocated atom-log-PMF buffers.
# =============================================================================

#' Construct the buffer workspace used by `run_ripr`.
#'
#' Pre-allocates the (M, max_atoms) and (M_f, max_atoms) tensors that store
#' per-atom log-PMFs and their -Inf masks, plus matching scratch buffers used
#' by the in-place tensor ops. The returned list of closures shares mutable
#' state via `<<-`; passing the workspace around is how we avoid leaking that
#' state into the main optimiser body.
make_ripr_workspace <- function(likelihood, q, max_atoms) {
  M <- likelihood$M
  X_mat_gpu <- likelihood$support_tensor

  log_q_mass_gpu <- log_pmf(q, X_mat_gpu)
  q_mass_gpu <- log_q_mass_gpu$exp()
  finite_q_gpu <- q_mass_gpu > 0
  q_mass_f_gpu <- q_mass_gpu[finite_q_gpu]
  M_f <- q_mass_f_gpu$numel()
  H_Q <- -(q_mass_f_gpu * log_q_mass_gpu[finite_q_gpu])$nan_to_num(
    nan = 0
  )$sum()$item()

  lc <- torch_full(c(M, max_atoms), -Inf, device = device, dtype = dtype)
  lc_f <- torch_full(c(M_f, max_atoms), -Inf, device = device, dtype = dtype)
  buf_MA <- torch_full(c(M, max_atoms), -Inf, device = device, dtype = dtype)
  buf_Mf <- torch_full(c(M_f, max_atoms), -Inf, device = device, dtype = dtype)
  lc_ninf <- lc$isneginf()
  lc_f_ninf <- lc_f$isneginf()
  n_live <- 0L

  write_atom_col <- function(k, theta) {
    lm <- likelihood$log_pmf(theta)
    lc[, k] <<- lm
    lc_ninf[, k] <<- lm$isneginf()
    lm_f <- lm[finite_q_gpu]
    lc_f[, k] <<- lm_f
    lc_f_ninf[, k] <<- lm_f$isneginf()
  }

  add_atom_col <- function(theta) {
    n_live <<- n_live + 1L
    write_atom_col(n_live, theta)
  }

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
      loss = -H_Q -
        (q_mass_f_gpu * log_Pw_f)$nan_to_num_(nan = 0.0)$sum()$item(),
      grad = as.numeric(-buf_f_live$sum(dim = 1L)$cpu())
    )
  }

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

  list(
    log_q_mass_gpu = log_q_mass_gpu,
    q_mass_gpu = q_mass_gpu,
    n_live = function() n_live,
    write_atom_col = write_atom_col,
    add_atom_col = add_atom_col,
    compute_log_Pw_gpu = compute_log_Pw_gpu,
    kl_loss_and_grad = kl_loss_and_grad,
    compute_responsibilities = compute_responsibilities,
    lc_col = function(k) lc[, k]
  )
}

# =============================================================================
# EM refinement
# =============================================================================

run_em_step <- function(
  workspace,
  face_descriptors,
  likelihood,
  weights,
  atoms,
  atom_face_idx,
  oracle_grid,
  em_iters,
  kl_atol,
  kl_rtol,
  kl_init
) {
  kl <- kl_init
  n_live <- workspace$n_live()
  kl_trace <- numeric(0L) # KL after each EM iteration actually run

  for (em_idx in seq_len(em_iters)) {
    log_Pw_gpu <- workspace$compute_log_Pw_gpu(weights)
    log_r <- workspace$compute_responsibilities(log_Pw_gpu, weights)

    new_weights <- (log_r$exp() * workspace$q_mass_gpu$unsqueeze(2L))$sum(
      dim = 1L
    )
    new_weights <- pmax(as.numeric(new_weights$cpu()), 1e-300)
    new_weights <- new_weights / sum(new_weights)

    for (k in seq_len(n_live)) {
      fd_k <- face_descriptors[[atom_face_idx[k]]]
      # Recover current atom's alpha via pseudo-inverse, clip + renormalise for FP safety
      seed_alpha_k <- as.vector(fd_k$pinv %*% atoms[[k]])
      seed_alpha_k <- pmax(seed_alpha_k, 0)
      seed_alpha_k <- seed_alpha_k / sum(seed_alpha_k)
      result <- em_mstep_face(
        fd_k,
        log_r[, k],
        likelihood,
        workspace$log_q_mass_gpu,
        seed_alpha_k
      )

      old_ll <- (result$weights_x * workspace$lc_col(k))$nan_to_num(
        nan = 0.0
      )$sum()$item()
      if (result$obj >= old_ll) {
        workspace$write_atom_col(k, result$theta)
        atoms[[k]] <- result$theta
      }
    }

    weights <- new_weights
    kl_new <- workspace$kl_loss_and_grad(weights)$loss
    kl_trace <- c(kl_trace, kl_new)
    if (kl - kl_new < kl_atol + kl_rtol * abs(kl)) {
      break
    }
    kl <- kl_new
  }
  list(weights = weights, atoms = atoms, kl = kl, kl_trace = kl_trace)
}

# =============================================================================
# Main optimiser
# =============================================================================

#' Generic RIPr optimiser via Frank-Wolfe + EM hybrid.
#'
#' Minimises D(Q || P_W) over mixtures W supported on a piecewise-convex null
#' hypothesis described by face descriptors. The optimisation begins by
#' initialising one atom per face (projected from q_max) and running EM to
#' refine that initial mixture. Each subsequent outer iteration runs the FW
#' oracle to add the most adversarial atom, then refines via EM.
#'
#' @param face_descriptors List of face descriptors. See
#'   [plurality_face_descriptors()].
#' @param likelihood Likelihood interface. See [make_multinomial_likelihood()].
#' @param q A `simplex_mixture` representing the numerator distribution Q.
#' @param max_atoms Total budget of atoms (including the K-1 initial atoms).
#'   Must be at least K-1. Default 50.
#' @param oracle_grid Grid density for per-face oracle. Default 200.
#' @param em_iters Max EM iterations per outer step. Default 3.
#' @param kl_atol Absolute tolerance for EM convergence. Default 1e-12.
#' @param kl_rtol Relative tolerance for EM convergence. Default 1e-6
#' @param gap_tol Convergence tolerance on the FW gap. Default 1e-6.
#' @param verbose Print progress. Default TRUE.
#' @return List with:
#'   - `atoms`: list of atom theta vectors.
#'   - `atom_face_idx`: integer vector of face indices for each atom.
#'   - `weights`: numeric vector of mixture weights.
#'   - `kl_trace`: data.frame with one row per recorded step. Columns:
#'       `iter` (outer iteration; 0 for init), `step_type`
#'       (`"init"`, `"em"`, `"fw"`), `n_atoms`, `kl`.
#'   - `outer_history`: data.frame with one row per outer iteration. Columns:
#'       `iter`, `face_idx`, `E_ratio`, `eps_star`, `kl_after_fw`, `kl_after_em`.
#'   - `E_star`: terminal FW gap value.
#'   - `kl`: terminal KL divergence.
#'   - `converged`: TRUE if convergence criteria were met.
#' @export
run_ripr <- function(
  face_descriptors,
  likelihood,
  q,
  max_atoms = 50L,
  oracle_grid = 200L,
  em_iters = 3L,
  kl_atol = 1e-12,
  kl_rtol = 1e-6,
  gap_tol = 1e-6,
  verbose = TRUE
) {
  n_faces <- length(face_descriptors)
  if (max_atoms < n_faces) {
    stop(sprintf(
      "max_atoms (%d) must be at least K-1 = %d (one atom per face for initialisation).",
      max_atoms,
      n_faces
    ))
  }
  max_fw_atoms <- max_atoms - n_faces
  workspace <- make_ripr_workspace(likelihood, q, max_atoms)

  # --- History accumulators ---
  # Pre-allocate generously; trim at the end.
  max_rows <- (max_fw_atoms + 1L) * (em_iters + 1L) + 1L
  trace_iter <- integer(max_rows)
  trace_type <- character(max_rows)
  trace_n_atoms <- integer(max_rows)
  trace_kl <- numeric(max_rows)
  trace_row <- 0L

  record_trace <- function(iter, type, n_atoms, kl_val) {
    trace_row <<- trace_row + 1L
    trace_iter[trace_row] <<- iter
    trace_type[trace_row] <<- type
    trace_n_atoms[trace_row] <<- n_atoms
    trace_kl[trace_row] <<- kl_val
  }

  outer_iter <- integer(max_fw_atoms + 1L)
  outer_face_idx <- integer(max_fw_atoms + 1L)
  outer_E_ratio <- numeric(max_fw_atoms + 1L)
  outer_eps_star <- numeric(max_fw_atoms + 1L)
  outer_kl_after_fw <- numeric(max_fw_atoms + 1L)
  outer_kl_after_em <- numeric(max_fw_atoms + 1L)
  outer_row <- 0L

  if (verbose) {
    message(sprintf(
      "run_ripr: K=%d, M=%d outcomes, max_atoms=%d, %d faces, kl_atol=%g, kl_rtol=%g, gap_tol=%g",
      likelihood$K,
      likelihood$M,
      max_atoms,
      n_faces,
      kl_atol,
      kl_rtol,
      gap_tol
    ))
  }

  # --- Main loop: each iter adds one FW atom then refines with EM ---
  converged <- FALSE
  for (atom_idx in seq_len(max_fw_atoms + 1L)) {
    if (atom_idx == 1L) {
      # Initialisation: one atom per face, projected from mode if it exists, otherwise use mean.
      atoms <- lapply(face_descriptors, function(fd) {
        fd$init_point(if (anyNA(q@mode)) q@mean else q@mode)
      })
      atom_face_idx <- seq_along(face_descriptors)
      for (th in atoms) {
        workspace$add_atom_col(th)
      }

      weights <- rep(1 / n_faces, n_faces)
      kl <- workspace$kl_loss_and_grad(weights)$loss
      kl_prev <- kl

      record_trace(0L, "init", n_faces, kl)

      kl_after_fw <- kl
      eps_star <- NA_real_
    } else {
      # --- Frank-Wolfe step ---
      log_Pw_gpu <- workspace$compute_log_Pw_gpu(weights)
      fw <- fw_oracle(
        face_descriptors,
        likelihood,
        workspace$log_q_mass_gpu,
        log_Pw_gpu,
        oracle_grid
      )

      log_tm_new <- likelihood$log_pmf(fw$best_theta)
      eps_star <- line_search(log_Pw_gpu, log_tm_new, workspace$q_mass_gpu)

      weights <- c(weights * (1 - eps_star), eps_star)
      atoms <- c(atoms, list(fw$best_theta))
      atom_face_idx <- c(atom_face_idx, fw$best_fi)
      workspace$add_atom_col(fw$best_theta)

      kl_after_fw <- workspace$kl_loss_and_grad(weights)$loss
      record_trace(atom_idx, "fw", workspace$n_live(), kl_after_fw)
      kl <- kl_after_fw
    }

    if (em_iters > 0L) {
      em_result <- run_em_step(
        workspace,
        face_descriptors,
        likelihood,
        weights,
        atoms,
        atom_face_idx,
        oracle_grid,
        em_iters,
        kl_atol,
        kl_rtol,
        kl
      )
      weights <- em_result$weights
      atoms <- em_result$atoms
      for (kl_em in em_result$kl_trace) {
        record_trace(atom_idx, "em", workspace$n_live(), kl_em)
      }
      kl <- em_result$kl
    }

    log_Pw_gpu <- workspace$compute_log_Pw_gpu(weights)
    fw <- fw_oracle(
      face_descriptors,
      likelihood,
      workspace$log_q_mass_gpu,
      log_Pw_gpu,
      oracle_grid
    )
    E_star <- fw$E_star

    outer_row <- outer_row + 1L
    outer_iter[outer_row] <- atom_idx
    outer_E_ratio[outer_row] <- E_star
    outer_kl_after_em[outer_row] <- kl
    outer_face_idx[outer_row] <- fw$best_fi
    outer_kl_after_fw[outer_row] <- kl_after_fw
    outer_eps_star[outer_row] <- eps_star

    if (verbose) {
      message(sprintf(
        "Atom %d/%d: E_th[E]-1 = %e, KL = %e, delta_KL = %e, eps* = %.4f",
        n_faces + atom_idx - 1L,
        max_atoms,
        E_star - 1,
        kl,
        kl_prev - kl,
        eps_star
      ))
    }

    if (E_star - 1 < gap_tol) {
      converged <- TRUE
      break
    }

    kl_prev <- kl
  }

  if (converged && verbose) {
    message(sprintf(
      "Converged after %d atoms (max_E_ratio - 1 = %e, kl = %e).",
      workspace$n_live(),
      E_star - 1,
      kl
    ))
  }

  kl_trace <- data.frame(
    iter = trace_iter[seq_len(trace_row)],
    step_type = trace_type[seq_len(trace_row)],
    n_atoms = trace_n_atoms[seq_len(trace_row)],
    kl = trace_kl[seq_len(trace_row)],
    stringsAsFactors = FALSE
  )

  outer_history <- data.frame(
    iter = outer_iter[seq_len(outer_row)],
    face_idx = outer_face_idx[seq_len(outer_row)],
    E_ratio = outer_E_ratio[seq_len(outer_row)],
    eps_star = outer_eps_star[seq_len(outer_row)],
    kl_after_fw = outer_kl_after_fw[seq_len(outer_row)],
    kl_after_em = outer_kl_after_em[seq_len(outer_row)],
    stringsAsFactors = FALSE
  )

  n_live <- workspace$n_live()
  list(
    atoms = atoms[seq_len(n_live)],
    atom_face_idx = atom_face_idx[seq_len(n_live)],
    weights = weights,
    kl_trace = kl_trace,
    outer_history = outer_history,
    E_star = E_star,
    kl = kl,
    converged = converged
  )
}
