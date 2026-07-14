box::use(
  stats[rgamma, optim, optimize],
  torch[
    torch_tensor,
    torch_full,
    torch_logsumexp,
    torch_maximum
  ],
  ripr / torch_settings[device, dtype],
  ripr / tensor_ops[add_ninf_any_, logsumexp_inplace_],
  ripr / mixture[log_pmf, n_categories],
  ripr / multinomial[make_multinomial_likelihood],
  ripr /
    simplex_utils[
      alpha_from_v,
      v_from_alpha,
      softmax_jacobian
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

#' Generic per-face optimiser: random Dirichlet seed search + BFGS refinement.
#'
#' `obj_and_grad(v)` is the BFGS objective in the unconstrained softmax
#' parametrisation `v`. `obj_grid_eval(alpha_mat)` evaluates the negative
#' objective at a batch of simplex points (used to pick BFGS seeds).
#' `n_seeds` controls how many random Dirichlet points are sampled for seed selection.
#' `seed_alpha` can be set to skip the global search and just run BFGS from a
#' single initial point.
optimise_on_face <- function(
  fd,
  obj_and_grad,
  obj_grid_eval,
  n_seeds = NULL,
  n_restarts = 25L,
  seed_alpha = NULL
) {
  n_vertices <- fd$n_vertices

  run_bfgs <- function(init_alpha, fallback_value) {
    init_alpha <- pmax(init_alpha, 1e-8)
    init_alpha <- init_alpha / sum(init_alpha)
    v0 <- v_from_alpha(init_alpha)
    tryCatch(
      optim(
        v0,
        fn = function(v) obj_and_grad(v)$value,
        gr = function(v) obj_and_grad(v)$gradient,
        method = "BFGS"
      ),
      error = function(e) list(par = v0, value = fallback_value)
    )
  }

  # Local optimisation from a single seed.
  if (!is.null(seed_alpha)) {
    res <- run_bfgs(seed_alpha, fallback_value = Inf)
    return(list(alpha_star = alpha_from_v(res$par), neg_value = res$value))
  }

  # Global (hopefully) optimisation via random Dirichlet restarts.
  alpha_mat <- matrix(
    rgamma(n_seeds * n_vertices, shape = 1),
    nrow = n_seeds,
    ncol = n_vertices
  )
  alpha_mat <- alpha_mat / rowSums(alpha_mat)

  neg_obj_grid <- obj_grid_eval(alpha_mat)
  top_idx <- order(neg_obj_grid, decreasing = TRUE)[seq_len(min(
    n_restarts,
    length(neg_obj_grid)
  ))]

  best <- list(par = NULL, value = Inf)
  for (idx in top_idx) {
    res <- run_bfgs(alpha_mat[idx, ], fallback_value = -neg_obj_grid[idx])
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
  n_seeds
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
    n_seeds = n_seeds
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
  n_seeds
) {
  face_results <- lapply(face_descriptors, function(fd) {
    fw_oracle_face(fd, likelihood, log_q_mass_gpu, log_Pw_gpu, n_seeds)
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

#' Optimal mixing weight eps in [1e-10, 1-1e-10] for adding a new atom to P_w.
fw_line_search <- function(log_Pw_gpu, log_tm_new, q_mass_gpu, tol = 1e-12) {
  m <- torch_maximum(log_Pw_gpu, log_tm_new)
  exp_a <- (log_Pw_gpu - m)$exp()
  exp_b <- (log_tm_new - m)$exp()
  g <- function(eps) {
    log_mix <- m + ((1 - eps) * exp_a + eps * exp_b)$log()
    -(q_mass_gpu * log_mix)$nan_to_num(nan = 0.0)$sum()$item()
  }

  eps_star <- optimize(g, interval = c(1e-10, 1 - 1e-10), tol = tol)$minimum
  if (g(eps_star) >= g(1e-10)) {
    eps_star <- 1e-10
  }
  eps_star
}

#' Pairwise (vertex exchange) line search: transfer alpha in [0, w_worst] from
#' the worst atom to the new FW atom. Works in linear probability space since
#' the update direction can be negative.
pairwise_line_search <- function(
  log_Pw_gpu,
  log_tm_new,
  log_tm_worst,
  w_worst,
  q_mass_gpu,
  tol = 1e-12
) {
  if (w_worst < 1e-10) {
    return(0)
  }
  Pw <- log_Pw_gpu$exp()
  p_new <- log_tm_new$exp()
  p_worst <- log_tm_worst$exp()
  delta <- p_new - p_worst
  g <- function(alpha) {
    -(q_mass_gpu * (Pw + alpha * delta)$log())$nan_to_num(
      nan = 0.0
    )$sum()$item()
  }
  alpha_star <- optimize(g, interval = c(0, w_worst), tol = tol)$minimum
  if (g(alpha_star) >= g(0)) {
    alpha_star <- 0
  }
  alpha_star
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

  #' E_ratio for every live atom simultaneously; returns numeric vector of length n_live.
  compute_e_ratios <- function(log_Pw_gpu) {
    lc_live <- lc$narrow(2L, 1L, n_live)
    log_terms <- (lc_live +
      log_q_mass_gpu$unsqueeze(2L) -
      log_Pw_gpu$unsqueeze(2L))$nan_to_num(nan = -Inf)
    as.numeric(torch_logsumexp(log_terms, dim = 1L)$exp()$cpu())
  }

  list(
    H_Q = H_Q,
    log_q_mass_gpu = log_q_mass_gpu,
    q_mass_gpu = q_mass_gpu,
    n_live = function() n_live,
    write_atom_col = write_atom_col,
    add_atom_col = add_atom_col,
    compute_log_Pw_gpu = compute_log_Pw_gpu,
    kl_loss_and_grad = kl_loss_and_grad,
    compute_responsibilities = compute_responsibilities,
    compute_e_ratios = compute_e_ratios,
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

#' Generic RIPr optimiser via Frank-Wolfe with intermediate EM steps.
#'
#' Minimises D(Q || P_W) over mixtures W supported on a piecewise-convex null
#' hypothesis described by face descriptors. The caller supplies initial atoms;
#' the optimiser then runs EM to refine them and iterates FW + EM for
#' `fw_iters` outer steps.
#'
#' @param face_descriptors List of face descriptors. See
#'   [boundary_face_descriptors()].
#' @param likelihood Likelihood interface. See [make_multinomial_likelihood()].
#' @param q A `simplex_mixture` representing the numerator distribution Q.
#' @param init_atoms K × l numeric matrix of initial atom locations on the null
#'   boundary. Each column is one atom.
#' @param init_atom_faces Integer vector of length l giving the 1-based face
#'   index for each initial atom.
#' @param fw_iters Number of Frank-Wolfe iterations. Each iteration runs the
#'   oracle and adds or (with pairwise FW) replaces one atom. Set to 0L for
#'   pure EM with no FW steps.
#' @param em_iters Max EM iterations per outer step. Set to 0L to skip EM.
#' @param n_seeds Number of random Dirichlet seeds per face for the oracle.
#'   Default 200.
#' @param kl_atol Absolute tolerance for EM convergence. Default 1e-12.
#' @param kl_rtol Relative tolerance for EM convergence. Default 1e-8.
#' @param gap_tol Convergence tolerance on the FW gap (E_star - 1). Default 1e-8.
#' @param ls_tol Line-search tolerance. Default 1e-12.
#' @param removal_thresh Weight threshold below which the worst atom is fully
#'   replaced in-place by the new FW atom (pairwise mode only). Default 1e-8.
#' @param fw_variant Frank-Wolfe variant: `"pairwise"` (default), `"linesearch"`,
#'   or `"standard"`. `"pairwise"` uses vertex-exchange steps (some iterations
#'   replace rather than add atoms, so final atom count may be less than
#'   `ncol(init_atoms) + fw_iters`). `"linesearch"` uses standard FW with an
#'   exact 1-D line search. `"standard"` uses the fixed schedule
#'   gamma = 2/(k+2).
#' @param checkpoint_iters Integer vector of outer-iteration indices (0 =
#'   post-init, before any FW step) at which to snapshot `(atoms, weights,
#'   atom_face_idx)`. Snapshots are taken after that iteration's EM
#'   refinement (if any). Default `NULL` records nothing.
#' @param verbose Print per-iteration progress. Default TRUE. Each outer
#'   iteration prints: `Gap` (FW duality gap = E_star - 1), `KL` (current KL
#'   divergence, with delta showing change since the previous iteration), `ULB`
#'   (running lower bound on optimal KL derived from the FW bound KL - Gap),
#'   `KL-ULB` (distance from current KL to that lower bound), and `alpha*`
#'   (mixing weight). The ULB is only meaningful when `n_seeds` is large enough
#'   that the oracle reliably finds the true maximum of E_theta[Q/P_w].
#' @return List with:
#'   - `atoms`: list of atom theta vectors (length = n_live).
#'   - `atom_face_idx`: integer vector of face indices (length = n_live).
#'   - `weights`: numeric mixture weights (length = n_live).
#'   - `kl_trace`: data.frame with columns `iter`, `step_type`
#'       (`"init"`, `"em"`, `"fw"`), `n_atoms`, `kl`.
#'   - `history`: data.frame with columns `iter`, `face_idx`, `gap`,
#'       `eps_star`, `prop_star`, `kl_after_fw`, `kl_after_em`, `kl_ulb`.
#'   - `gap`: terminal FW gap value (E_star - 1).
#'   - `oracle_theta`: terminal maximising theta.
#'   - `kl`: terminal KL divergence.
#'   - `kl_ulb`: tightest lower bound on optimal KL seen across iterations.
#'   - `converged`: TRUE if FW gap fell below `gap_tol`.
#'   - `checkpoints`: list of `(iter, atoms, weights, atom_face_idx)` snapshots,
#'       one per requested `checkpoint_iters` value actually reached.
#' @export
run_ripr <- function(
  face_descriptors,
  likelihood,
  q,
  init_atoms,
  init_atom_faces,
  fw_iters,
  em_iters,
  n_seeds = 200L,
  kl_atol = 1e-12,
  kl_rtol = 1e-8,
  gap_tol = 1e-8,
  ls_tol = 1e-12,
  removal_thresh = 1e-8,
  fw_variant = c("pairwise", "linesearch", "standard"),
  checkpoint_iters = NULL,
  verbose = TRUE
) {
  n_init <- ncol(init_atoms)
  if (length(init_atom_faces) != n_init) {
    stop(sprintf(
      "length(init_atom_faces) (%d) must equal ncol(init_atoms) (%d).",
      length(init_atom_faces),
      n_init
    ))
  }

  workspace <- make_ripr_workspace(likelihood, q, n_init + fw_iters)
  t_start <- proc.time()[["elapsed"]]

  # --- History accumulators ---
  trace_rows <- list()
  outer_rows <- list()
  checkpoints <- list()
  kl_ulb <- -Inf

  record_trace <- function(iter, type, n_atoms, kl_val) {
    trace_rows[[length(trace_rows) + 1L]] <<- data.frame(
      iter = iter,
      step_type = type,
      n_atoms = n_atoms,
      kl = kl_val
    )
  }

  record_checkpoint <- function(iter, oracle_theta_cp = NULL) {
    if (is.null(checkpoint_iters) || !(iter %in% checkpoint_iters)) {
      return(invisible(NULL))
    }
    n_live <- workspace$n_live()
    checkpoints[[length(checkpoints) + 1L]] <<- list(
      iter = iter,
      atoms = atoms[seq_len(n_live)],
      weights = weights[seq_len(n_live)],
      atom_face_idx = atom_face_idx[seq_len(n_live)],
      oracle_theta = oracle_theta_cp
    )
  }

  fw_variant <- match.arg(fw_variant)
  fw_mode <- if (fw_iters == 0L) "em-only" else fw_variant

  if (verbose) {
    message(sprintf(
      "run_ripr: K=%d, M=%d outcomes, n_init=%d, fw_iters=%d, mode=%s, em_iters=%d, kl_atol=%g, kl_rtol=%g, gap_tol=%g, n_seeds=%d",
      likelihood$K,
      likelihood$M,
      n_init,
      fw_iters,
      fw_mode,
      em_iters,
      kl_atol,
      kl_rtol,
      gap_tol,
      n_seeds
    ))
  }

  # --- Initialisation ---
  atoms <- lapply(seq_len(n_init), function(j) init_atoms[, j])
  atom_face_idx <- init_atom_faces
  for (th in atoms) {
    workspace$add_atom_col(th)
  }
  weights <- rep(1 / n_init, n_init)
  kl <- workspace$kl_loss_and_grad(weights)$loss
  record_trace(0L, "init", workspace$n_live(), kl)

  # --- Initial EM ---
  if (em_iters > 0L) {
    em_result <- run_em_step(
      workspace,
      face_descriptors,
      likelihood,
      weights,
      atoms,
      atom_face_idx,
      em_iters,
      kl_atol,
      kl_rtol,
      kl
    )
    weights <- em_result$weights
    atoms <- em_result$atoms
    for (kl_em in em_result$kl_trace) {
      record_trace(0L, "em", workspace$n_live(), kl_em)
    }
    kl <- em_result$kl
  }
  # --- Initial oracle ---
  gap <- NA_real_
  oracle_theta <- NULL
  converged <- FALSE
  log_Pw_gpu <- workspace$compute_log_Pw_gpu(weights)
  fw <- fw_oracle(
    face_descriptors,
    likelihood,
    workspace$log_q_mass_gpu,
    log_Pw_gpu,
    n_seeds
  )
  gap <- fw$E_star - 1
  oracle_theta <- fw$best_theta
  kl_ulb <- kl - gap
  outer_rows[[1L]] <- data.frame(
    iter = 0L,
    face_idx = fw$best_fi,
    gap = gap,
    eps_star = NA_real_,
    prop_star = NA_real_,
    kl_after_fw = NA_real_,
    kl_after_em = kl,
    kl_ulb = kl_ulb,
    elapsed_s = proc.time()[["elapsed"]] - t_start
  )
  if (verbose) {
    message(sprintf(
      "Init [%d atoms]: Gap %e, KL %e, ULB %e, GR %e",
      workspace$n_live(),
      gap,
      kl,
      kl_ulb,
      kl - log1p(gap)
    ))
  }
  record_checkpoint(0L, oracle_theta_cp = fw$best_theta)
  if (gap < gap_tol) {
    converged <- TRUE
  }

  # --- FW loop ---
  kl_prev <- kl
  for (fw_idx in seq_len(fw_iters)) {
    if (converged) {
      break
    }

    # FW step — uses fw$best_theta from the oracle at end of previous phase
    log_tm_new <- likelihood$log_pmf(fw$best_theta)

    if (fw_variant == "pairwise") {
      e_ratios <- workspace$compute_e_ratios(log_Pw_gpu)
      active_idx <- which(weights > removal_thresh)
      k_worst <- active_idx[which.min(e_ratios[active_idx])]
      w_worst <- weights[k_worst]
      alpha_star <- pairwise_line_search(
        log_Pw_gpu,
        log_tm_new,
        workspace$lc_col(k_worst),
        w_worst,
        workspace$q_mass_gpu,
        tol = ls_tol
      )
      if (alpha_star == 0) {
        warning(sprintf(
          "pairwise_line_search returned alpha=0 at fw_iter %d; oracle atom and worst atom may coincide.",
          fw_idx
        ))
      }
      eps_star <- alpha_star
      if (w_worst - alpha_star < removal_thresh) {
        workspace$write_atom_col(k_worst, fw$best_theta)
        atoms[[k_worst]] <- fw$best_theta
        atom_face_idx[k_worst] <- fw$best_fi
        weights[k_worst] <- w_worst
        prop_star <- 1
      } else {
        weights[k_worst] <- w_worst - alpha_star
        weights <- c(weights, alpha_star)
        atoms <- c(atoms, list(fw$best_theta))
        atom_face_idx <- c(atom_face_idx, fw$best_fi)
        workspace$add_atom_col(fw$best_theta)
        prop_star <- alpha_star / w_worst
      }
    } else if (fw_variant == "linesearch") {
      eps_star <- fw_line_search(
        log_Pw_gpu,
        log_tm_new,
        workspace$q_mass_gpu,
        tol = ls_tol
      )
      weights <- c(weights * (1 - eps_star), eps_star)
      atoms <- c(atoms, list(fw$best_theta))
      atom_face_idx <- c(atom_face_idx, fw$best_fi)
      workspace$add_atom_col(fw$best_theta)
      prop_star <- NA_real_
    } else {
      eps_star <- 2 / (fw_idx + 2)
      weights <- c(weights * (1 - eps_star), eps_star)
      atoms <- c(atoms, list(fw$best_theta))
      atom_face_idx <- c(atom_face_idx, fw$best_fi)
      workspace$add_atom_col(fw$best_theta)
      prop_star <- NA_real_
    }

    kl_after_fw <- workspace$kl_loss_and_grad(weights)$loss
    record_trace(fw_idx, "fw", workspace$n_live(), kl_after_fw)

    # EM refinement
    if (em_iters > 0L) {
      em_result <- run_em_step(
        workspace,
        face_descriptors,
        likelihood,
        weights,
        atoms,
        atom_face_idx,
        em_iters,
        kl_atol,
        kl_rtol,
        kl_after_fw
      )
      weights <- em_result$weights
      atoms <- em_result$atoms
      for (kl_em in em_result$kl_trace) {
        record_trace(fw_idx, "em", workspace$n_live(), kl_em)
      }
      kl <- em_result$kl
    } else {
      kl <- kl_after_fw
    }
    # Oracle — single call per iteration; result used for next iteration's FW step
    log_Pw_gpu <- workspace$compute_log_Pw_gpu(weights)
    fw <- fw_oracle(
      face_descriptors,
      likelihood,
      workspace$log_q_mass_gpu,
      log_Pw_gpu,
      n_seeds
    )
    gap <- fw$E_star - 1
    oracle_theta <- fw$best_theta
    kl_ulb <- max(kl_ulb, kl - gap)
    outer_rows[[length(outer_rows) + 1L]] <- data.frame(
      iter = fw_idx,
      face_idx = fw$best_fi,
      gap = gap,
      eps_star = eps_star,
      prop_star = if (is.na(prop_star)) NA_real_ else prop_star,
      kl_after_fw = kl_after_fw,
      kl_after_em = kl,
      kl_ulb = kl_ulb,
      elapsed_s = proc.time()[["elapsed"]] - t_start
    )
    if (verbose) {
      message(sprintf(
        "Iter %d [%d atoms]: Gap %e, KL %e (delta %.1e), ULB %e, KL-ULB %e, GR %e, alpha* %.2e, prop* %.2f",
        fw_idx,
        workspace$n_live(),
        gap,
        kl,
        kl_prev - kl,
        kl_ulb,
        kl - kl_ulb,
        kl - log1p(gap),
        eps_star,
        if (is.na(prop_star)) NaN else prop_star
      ))
    }
    record_checkpoint(fw_idx, oracle_theta_cp = fw$best_theta)
    if (gap < gap_tol) {
      converged <- TRUE
      if (verbose) {
        message(sprintf(
          "Converged after %d FW iterations (gap = %e, kl = %e).",
          fw_idx,
          gap,
          kl
        ))
      }
    }
    kl_prev <- kl
  }

  kl_trace <- do.call(rbind, trace_rows)
  history <- do.call(rbind, outer_rows)

  n_live <- workspace$n_live()
  list(
    atoms = atoms[seq_len(n_live)],
    atom_face_idx = atom_face_idx[seq_len(n_live)],
    weights = weights[seq_len(n_live)],
    kl_trace = kl_trace,
    history = history,
    gap = gap,
    oracle_theta = oracle_theta,
    kl = kl,
    kl_ulb = kl_ulb,
    converged = converged,
    checkpoints = checkpoints
  )
}


#' Frank-Wolfe gap for two simplex mixtures over a null hypothesis boundary.
#'
#' Computes max_theta E_theta[Q / P] over the null hypothesis faces, returning
#' both the gap value and the maximising theta. Values above 1 indicate the
#' mixture P has not yet minimised KL(Q || P) over the null.
#'
#' @param Q A `simplex_mixture` for the alternative distribution.
#' @param P A `simplex_mixture` for the null mixture.
#' @param face_descriptors List of face descriptors. See [boundary_face_descriptors()].
#' @param n Integer. Multinomial sample size.
#' @param n_seeds Integer. Random Dirichlet seeds per face. Default 200L.
#' @return List with `E_star` (numeric) and `theta_star` (numeric vector).
#' @export
fw_gap <- function(Q, P, face_descriptors, n, n_seeds = 200L) {
  K <- n_categories(Q)
  likelihood <- make_multinomial_likelihood(n, K)
  log_q_mass_gpu <- log_pmf(Q, likelihood$support_tensor)
  log_Pw_gpu <- log_pmf(P, likelihood$support_tensor)
  fw <- fw_oracle(
    face_descriptors,
    likelihood,
    log_q_mass_gpu,
    log_Pw_gpu,
    n_seeds
  )
  list(
    gap = fw$E_star - 1,
    oracle_theta = fw$best_theta,
    face_results = fw$face_results
  )
}
