box::use(
  torch[
    torch_tensor,
    torch_full,
    torch_logsumexp
  ],
  stats[rgamma],
  ripr / torch_settings[device, dtype],
  ripr / tensor_ops[add_ninf_any_, logsumexp_inplace_],
  ripr / mixture[log_pmf]
)

#' Pure EM optimiser for RIPr.
#'
#' Minimises D(Q || P_W) over mixtures W supported on a piecewise-convex null
#' hypothesis described by face descriptors. Uses pure EM with random atom
#' initialisation. The number of atoms per face grows across outer iterations;
#' the algorithm stops when growing the support no longer improves KL by `tol`.
#'
#' Each outer iteration:
#'   1. Randomly initialise `atoms_per_face` atoms on each face (Dirichlet(1)).
#'   2. Run EM until KL stabilises (decrease per iteration below `em_tol`).
#'   3. Compare best EM KL to previous outer iteration's KL.
#'   4. If improvement exceeds `tol`, increment `atoms_per_face` and continue.
#'      Otherwise stop and return the best result seen.
#'
#' Several random restarts can be done per outer iteration via `n_restarts`.
#'
#' @param face_descriptors List of face descriptors.
#' @param likelihood Likelihood interface.
#' @param q A `mixture_mnom`.
#' @param init_atoms_per_face Initial atoms per face. Default 2.
#' @param max_atoms_per_face Hard cap on atoms per face. Default 64.
#' @param n_restarts Random restarts per outer iteration. Default 3.
#' @param em_maxit Max EM iterations per restart. Default 1000.
#' @param em_tol Stop EM when KL decrease per iteration falls below this. Default 1e-10.
#' @param tol Stop outer loop when KL improvement falls below this. Default 1e-6.
#' @param verbose Print progress. Default TRUE.
#' @return List with `atoms`, `atom_face_idx`, `weights`, `kl`, `history`.
#' @export
run_em <- function(
  face_descriptors,
  likelihood,
  q,
  init_atoms_per_face = 2L,
  max_atoms_per_face = 64L,
  n_restarts = 3L,
  em_maxit = 1000L,
  em_tol = 1e-10,
  tol = 1e-6,
  verbose = TRUE
) {
  n_faces <- length(face_descriptors)
  M <- likelihood$M
  X_mat_gpu <- likelihood$support_tensor

  # --- Precompute Q-related quantities ---
  log_q_mass_gpu <- log_pmf(q, X_mat_gpu)
  q_mass_gpu <- log_q_mass_gpu$exp()
  finite_q_gpu <- q_mass_gpu > 0
  q_mass_f_gpu <- q_mass_gpu[finite_q_gpu]
  M_f <- q_mass_f_gpu$numel()

  # --- Pre-allocate buffers sized for max possible atoms ---
  max_total_atoms <- max_atoms_per_face * n_faces
  lc <- torch_full(
    c(M, max_total_atoms),
    -Inf,
    device = device,
    dtype = dtype
  )
  lc_f <- torch_full(
    c(M_f, max_total_atoms),
    -Inf,
    device = device,
    dtype = dtype
  )
  buf_MA <- torch_full(
    c(M, max_total_atoms),
    -Inf,
    device = device,
    dtype = dtype
  )
  buf_Mf <- torch_full(
    c(M_f, max_total_atoms),
    -Inf,
    device = device,
    dtype = dtype
  )
  lc_ninf <- lc$isneginf()
  lc_f_ninf <- lc_f$isneginf()
  n_live <- 0L

  # --- Atom buffer management ---
  write_atom_col <- function(k, theta) {
    lm <- likelihood$log_pmf(theta)
    lc[, k] <<- lm
    lc_ninf[, k] <<- lm$isneginf()
    lm_f <- lm[finite_q_gpu]
    lc_f[, k] <<- lm_f
    lc_f_ninf[, k] <<- lm_f$isneginf()
  }

  reset_support <- function() {
    n_live <<- 0L
  }

  add_atom <- function(theta) {
    n_live <<- n_live + 1L
    write_atom_col(n_live, theta)
  }

  # --- Mixture log-PMF ---
  compute_log_Pw <- function(w) {
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

  # --- KL loss ---
  kl_loss <- function(w) {
    log_Pw <- compute_log_Pw(w)
    log_Pw_f <- log_Pw[finite_q_gpu]
    -(q_mass_f_gpu * log_Pw_f)$nan_to_num_(nan = 0.0)$sum()$item()
  }

  # --- E-step: log responsibilities ---
  compute_responsibilities <- function(log_Pw, weights) {
    w_log <- torch_tensor(
      log(pmax(weights, 1e-300)),
      device = device,
      dtype = dtype
    )
    lc_live <- lc$narrow(2L, 1L, n_live)
    log_r <- lc_live + w_log$unsqueeze(1L) - log_Pw$unsqueeze(2L)
    log_r$nan_to_num_(nan = -Inf)
    log_r - torch_logsumexp(log_r, dim = 2L, keepdim = TRUE)
  }

  # --- M-step location update for atom k on face fd, weighted by r_k ---
  m_step_location <- function(fd, log_r_k) {
    log_weights <- log_q_mass_gpu + log_r_k
    weights_x <- log_weights$exp()$nan_to_num(nan = 0.0)
    J <- fd$jacobian()

    obj_and_grad <- function(alpha) {
      theta <- fd$parametrise(alpha)
      if (any(theta < 0)) {
        return(list(value = Inf, gradient = rep(0, length(alpha))))
      }
      log_tm <- likelihood$log_pmf(theta)
      obj_val <- (weights_x * log_tm)$nan_to_num(nan = 0.0)$sum()$item()
      # Score without -n term: d log p_theta(x) / d theta_l = x_l / theta_l
      score_gpu <- likelihood$score(theta) + likelihood$n
      grad_theta <- as.numeric(
        (weights_x$unsqueeze(2L) * score_gpu)$nan_to_num(nan = 0.0)$sum(
          dim = 1L
        )$cpu()
      )
      grad_alpha <- as.vector(grad_theta %*% J)
      list(value = -obj_val, gradient = -grad_alpha)
    }

    # L-BFGS-B in reduced (n_vertices - 1) coordinates with simplex constraints
    n_vertices <- fd$n_vertices
    alpha_init <- rep(1 / n_vertices, n_vertices)

    obj_and_grad_reduced <- function(alpha_reduced) {
      alpha_last <- 1 - sum(alpha_reduced)
      alpha <- c(alpha_reduced, alpha_last)
      og <- obj_and_grad(alpha)
      grad_reduced <- og$gradient[-n_vertices] - og$gradient[n_vertices]
      list(value = og$value, gradient = grad_reduced)
    }

    res <- tryCatch(
      stats::optim(
        alpha_init[-n_vertices],
        fn = function(a) obj_and_grad_reduced(a)$value,
        gr = function(a) obj_and_grad_reduced(a)$gradient,
        method = "L-BFGS-B",
        lower = rep(0, n_vertices - 1L),
        upper = rep(1, n_vertices - 1L)
      ),
      error = function(e) list(par = alpha_init[-n_vertices])
    )

    alpha_star <- c(res$par, 1 - sum(res$par))
    alpha_star <- pmax(alpha_star, 0)
    alpha_star <- alpha_star / sum(alpha_star)
    fd$parametrise(alpha_star)
  }

  # --- Random initialisation: sample atoms uniformly from each face ---
  random_init <- function(atoms_per_face) {
    reset_support()
    atoms <- list()
    atom_face_idx <- list()
    for (fd_idx in seq_along(face_descriptors)) {
      fd <- face_descriptors[[fd_idx]]
      for (rep in seq_len(atoms_per_face)) {
        alpha <- rgamma(fd$n_vertices, shape = 1)
        alpha <- alpha / sum(alpha)
        theta <- fd$parametrise(alpha)
        atoms[[length(atoms) + 1L]] <- theta
        atom_face_idx[[length(atom_face_idx) + 1L]] <- fd_idx
        add_atom(theta)
      }
    }
    list(atoms = atoms, atom_face_idx = atom_face_idx)
  }

  # --- One EM run to convergence on the current support ---
  run_em_to_convergence <- function(atoms, atom_face_idx) {
    weights <- rep(1 / n_live, n_live)
    kl <- kl_loss(weights)
    for (em_idx in seq_len(em_maxit)) {
      log_Pw <- compute_log_Pw(weights)
      log_r <- compute_responsibilities(log_Pw, weights)

      # M-step weights (closed form)
      new_weights <- (log_r$exp() * q_mass_gpu$unsqueeze(2L))$sum(dim = 1L)
      new_weights <- pmax(as.numeric(new_weights$cpu()), 1e-300)
      new_weights <- new_weights / sum(new_weights)

      # M-step locations
      for (k in seq_len(n_live)) {
        fd_k <- face_descriptors[[atom_face_idx[[k]]]]
        new_theta <- m_step_location(fd_k, log_r[, k])
        write_atom_col(k, new_theta)
        atoms[[k]] <- new_theta
      }

      weights <- new_weights
      kl_new <- kl_loss(weights)
      if (kl - kl_new < em_tol) {
        break
      }
      kl <- kl_new
    }
    list(
      atoms = atoms,
      atom_face_idx = atom_face_idx,
      weights = weights,
      kl = kl
    )
  }

  # --- One outer iteration: n_restarts random inits, keep best ---
  best_for_support_size <- function(atoms_per_face) {
    best <- NULL
    best_kl <- Inf
    for (restart in seq_len(n_restarts)) {
      init <- random_init(atoms_per_face)
      result <- run_em_to_convergence(init$atoms, init$atom_face_idx)
      if (result$kl < best_kl) {
        best_kl <- result$kl
        best <- result
      }
    }
    best
  }

  # --- Outer loop: grow support until KL stops improving ---
  if (verbose) {
    message(sprintf(
      "run_em: K=%d, M=%d outcomes, %d faces, init_atoms_per_face=%d",
      likelihood$K,
      M,
      n_faces,
      init_atoms_per_face
    ))
  }

  atoms_per_face <- init_atoms_per_face
  best_result <- NULL
  best_kl <- Inf
  history <- list()
  iter <- 1L

  repeat {
    result <- best_for_support_size(atoms_per_face)
    improvement <- best_kl - result$kl

    history[[iter]] <- list(
      atoms_per_face = atoms_per_face,
      kl = result$kl,
      improvement = improvement
    )

    if (verbose) {
      message(sprintf(
        "Iter %d: atoms_per_face=%d, KL=%e, improvement=%e",
        iter,
        atoms_per_face,
        result$kl,
        improvement
      ))
    }

    if (result$kl < best_kl) {
      best_kl <- result$kl
      best_result <- result
    }

    if (improvement < tol && iter > 1L) {
      if (verbose) {
        message("Stopping: improvement below tol.")
      }
      break
    }
    if (atoms_per_face >= max_atoms_per_face) {
      if (verbose) {
        message("Stopping: max_atoms_per_face reached.")
      }
      break
    }

    atoms_per_face <- atoms_per_face + 1L
    iter <- iter + 1L
  }

  list(
    atoms = best_result$atoms,
    atom_face_idx = best_result$atom_face_idx,
    weights = best_result$weights,
    kl = best_result$kl,
    history = history
  )
}
