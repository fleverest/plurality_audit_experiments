box::use(
  torch[
    torch_tensor,
    torch_full,
    torch_logsumexp
  ],
  ripr / torch_settings[device, dtype],
  ripr / tensor_ops[add_ninf_any_, logsumexp_inplace_],
  ripr / mixture[log_pmf],
  ripr /
    simplex_utils[
      alpha_from_v,
      v_from_alpha,
      softmax_jacobian,
      mirror_descent,
      simplex_lattice
    ]
)

#' Generic RIPr optimiser via Frank-Wolfe + EM hybrid.
#'
#' Minimises D(Q || P_W) over mixtures W supported on a piecewise-convex null
#' hypothesis described by face descriptors. The optimisation alternates
#' Frank-Wolfe column generation (adding the most adversarial atom per face)
#' with EM refinement (jointly improving weights and atom locations).
#'
#' @param face_descriptors List of face descriptors. Each descriptor is a list
#'   with components `parametrise`, `parametrise_batch`, `jacobian`,
#'   `n_vertices`, `init_point`, `face_index`. See [plurality_face_descriptors()]
#'   for the plurality null example.
#' @param likelihood Likelihood interface, with components `support_tensor`,
#'   `M`, `log_pmf`, `log_pmf_batch`, `score`, `n`, `K`. See
#'   [make_multinomial_likelihood()] for the multinomial example.
#' @param q A `mixture_mnom` -- the numerator distribution Q.
#' @param atoms_per_face Maximum atoms per face. Default 50.
#' @param oracle_grid Grid density for per-face oracle. Default 200.
#' @param reweight_maxit Max mirror descent iterations. Default 1000.
#' @param n_em_iter Max EM iterations per Frank-Wolfe step. Default 3.
#' @param tol Convergence tolerance on KL divergence. Default 1e-10.
#' @param verbose Print progress. Default TRUE.
#' @return List with `atoms`, `atom_faces`, `weights`, `history`, `converged`.
#' @export
run_ripr <- function(
  face_descriptors,
  likelihood,
  q,
  atoms_per_face = 50L,
  oracle_grid = 200L,
  reweight_maxit = 1000L,
  n_em_iter = 3L,
  tol = 1e-10,
  verbose = TRUE
) {
  n_faces <- length(face_descriptors)
  max_atoms <- atoms_per_face * n_faces
  M <- likelihood$M
  X_mat_gpu <- likelihood$support_tensor

  # --- Precompute Q-related quantities ---
  log_q_mass_gpu <- log_pmf(q, X_mat_gpu)
  q_mass_gpu <- log_q_mass_gpu$exp()
  finite_q_gpu <- q_mass_gpu > 0
  q_mass_f_gpu <- q_mass_gpu[finite_q_gpu]
  M_f <- q_mass_f_gpu$numel()

  # --- Pre-allocate GPU buffers for atom log-PMFs ---
  lc <- torch_full(c(M, max_atoms), -Inf, device = device, dtype = dtype)
  lc_f <- torch_full(c(M_f, max_atoms), -Inf, device = device, dtype = dtype)
  buf_MA <- torch_full(c(M, max_atoms), -Inf, device = device, dtype = dtype)
  buf_Mf <- torch_full(c(M_f, max_atoms), -Inf, device = device, dtype = dtype)
  lc_ninf <- lc$isneginf()
  lc_f_ninf <- lc_f$isneginf()
  n_live <- 0L

  # --- Atom buffer updates ---
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

  # --- Mixture log-PMF ---
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

  # --- Oracle-side quantities ---
  compute_E_ratio <- function(log_tm_gpu, log_Pw_gpu) {
    log_terms <- (log_tm_gpu + log_q_mass_gpu - log_Pw_gpu)$nan_to_num(
      nan = -Inf
    )
    torch_logsumexp(log_terms, dim = 1L)$exp()$item()
  }

  compute_E_ratio_grad_theta <- function(log_tm_gpu, log_Pw_gpu, theta) {
    log_terms <- (log_tm_gpu + log_q_mass_gpu - log_Pw_gpu)$nan_to_num(
      nan = -Inf
    )
    weights_x <- log_terms$exp()
    score_gpu <- likelihood$score(theta)
    as.numeric(
      (weights_x$unsqueeze(2L) * score_gpu)$nan_to_num(nan = 0.0)$sum(
        dim = 1L
      )$cpu()
    )
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
      loss = -(q_mass_f_gpu * log_Pw_f)$nan_to_num_(nan = 0.0)$sum()$item(),
      grad = as.numeric(-buf_f_live$sum(dim = 1L)$cpu())
    )
  }

  # --- Generic per-face optimiser: grid search + BFGS refinement ---
  optimise_on_face <- function(fd, obj_and_grad, obj_grid_eval) {
    n_vertices <- fd$n_vertices
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

  # --- Frank-Wolfe oracle on face fd: argmax_theta E_theta[Q / P_w] ---
  find_best_on_face <- function(fd, log_Pw_gpu) {
    J <- fd$jacobian()

    obj_and_grad <- function(v) {
      alpha <- alpha_from_v(v)
      theta <- fd$parametrise(alpha)
      log_tm <- likelihood$log_pmf(theta)
      E <- compute_E_ratio(log_tm, log_Pw_gpu)
      grad_theta <- compute_E_ratio_grad_theta(log_tm, log_Pw_gpu, theta)
      grad_v <- as.vector(grad_theta %*% J %*% softmax_jacobian(alpha))
      list(value = -E, gradient = -grad_v)
    }

    obj_grid_eval <- function(alpha_mat) {
      log_tm_batch <- likelihood$log_pmf_batch(fd$parametrise_batch(alpha_mat))
      log_terms <- (log_tm_batch +
        log_q_mass_gpu$unsqueeze(2L) -
        log_Pw_gpu$unsqueeze(2L))$nan_to_num(nan = -Inf)
      as.numeric(torch_logsumexp(log_terms, dim = 1L)$exp()$cpu())
    }

    res <- optimise_on_face(fd, obj_and_grad, obj_grid_eval)
    list(theta = fd$parametrise(res$alpha_star), E_ratio = -res$neg_value)
  }

  # --- EM M-step location: argmax_theta sum_x q(x) r_k(x) log p_theta(x) ---
  find_best_on_face_weighted <- function(fd, log_r_k) {
    J <- fd$jacobian()
    log_weights <- log_q_mass_gpu + log_r_k
    weights_x <- log_weights$exp()$nan_to_num(nan = 0.0)

    obj_and_grad <- function(v) {
      alpha <- alpha_from_v(v)
      theta <- fd$parametrise(alpha)
      log_tm <- likelihood$log_pmf(theta)
      obj_val <- (weights_x * log_tm)$nan_to_num(nan = 0.0)$sum()$item()
      score_gpu <- likelihood$score(theta) + likelihood$n # drop the -n term
      grad_theta <- as.numeric(
        (weights_x$unsqueeze(2L) * score_gpu)$nan_to_num(nan = 0.0)$sum(
          dim = 1L
        )$cpu()
      )
      grad_v <- as.vector(grad_theta %*% J %*% softmax_jacobian(alpha))
      list(value = -obj_val, gradient = -grad_v)
    }

    obj_grid_eval <- function(alpha_mat) {
      log_tm_batch <- likelihood$log_pmf_batch(fd$parametrise_batch(alpha_mat))
      as.numeric(
        (weights_x$unsqueeze(2L) * log_tm_batch)$nan_to_num(nan = 0.0)$sum(
          dim = 1L
        )$cpu()
      )
    }

    res <- optimise_on_face(fd, obj_and_grad, obj_grid_eval)
    list(theta = fd$parametrise(res$alpha_star))
  }

  # --- E-step responsibilities ---
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

  # --- Frank-Wolfe oracle over all faces ---
  frank_wolfe_oracle <- function(weights) {
    log_Pw_gpu <- compute_log_Pw_gpu(weights)
    face_results <- lapply(face_descriptors, function(fd) {
      find_best_on_face(fd, log_Pw_gpu)
    })
    E_ratios <- vapply(face_results, `[[`, "E_ratio", FUN.VALUE = numeric(1L))
    list(face_results = face_results, E_star = max(E_ratios))
  }

  # --- EM refinement ---
  run_em_step <- function(weights, n_em_iter, kl_init) {
    kl <- kl_init
    for (em_idx in seq_len(n_em_iter)) {
      log_Pw_gpu <- compute_log_Pw_gpu(weights)
      log_r <- compute_responsibilities(log_Pw_gpu, weights)

      new_weights <- (log_r$exp() * q_mass_gpu$unsqueeze(2L))$sum(dim = 1L)
      new_weights <- pmax(as.numeric(new_weights$cpu()), 1e-300)
      new_weights <- new_weights / sum(new_weights)

      for (k in seq_len(n_live)) {
        fd_k <- face_descriptors[[atom_face_idx[[k]]]]
        result <- find_best_on_face_weighted(fd_k, log_r[, k])
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

  # --- Initialisation ---
  if (verbose) {
    message(sprintf(
      "run_ripr: K=%d, M=%d outcomes, %d atoms/face, %d faces, tol=%g",
      likelihood$K,
      M,
      atoms_per_face,
      n_faces,
      tol
    ))
  }

  q_max <- q@atoms[, which.max(q@weights)]
  atoms <- lapply(face_descriptors, function(fd) fd$init_point(q_max))
  # atom_face_idx[[k]] is the index into face_descriptors for atom k
  atom_face_idx <- as.list(seq_along(face_descriptors))
  for (th in atoms) {
    add_atom_col(th)
  }

  weights <- rep(1 / n_faces, n_faces)
  converged <- FALSE
  history <- vector("list", atoms_per_face)
  kl_prev <- NULL

  # --- Main loop ---
  for (atom_idx in seq_len(atoms_per_face)) {
    weights <- mirror_descent(
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

    if (kl_prev - kl < tol) {
      converged <- TRUE
      break
    }

    kl_prev <- kl

    if (atom_idx < atoms_per_face) {
      new_atoms <- lapply(face_results, `[[`, "theta")
      new_face_idx <- as.list(seq_along(face_descriptors))
      atoms <- c(atoms, new_atoms)
      atom_face_idx <- c(atom_face_idx, new_face_idx)
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
      "Converged after %d atoms (max_E_ratio - 1 = %e, kl = %e).",
      n_live,
      E_star - 1,
      kl
    ))
  }

  list(
    atoms = atoms,
    atom_face_idx = atom_face_idx,
    weights = weights,
    history = history[!sapply(history, is.null)],
    converged = converged
  )
}
