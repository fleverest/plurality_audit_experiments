box::use(
  torch[
    optimizer,
    with_no_grad,
    is_undefined_tensor,
    nnf_softmax,
    torch_randn,
    torch_full,
    torch_tensor,
    torch_zeros_like,
    torch_where,
    torch_logsumexp,
    torch_lr_step = lr_step
  ],
  ripr / torch_settings[device, dtype],
  ripr / tensor_ops[matmul_0_ninf, add_ninf_any],
  ripr / multinomial[build_counts_tensor, mnom_logpmf]
)

# Sample n rows from a Dirichlet(alpha) distribution.
rdirichlet <- function(n, alpha) {
  k <- length(alpha)
  samples <- matrix(0, nrow = n, ncol = k)
  for (i in seq_len(k)) {
    samples[, i] <- rgamma(n, shape = alpha[i], rate = 1)
  }
  samples / rowSums(samples)
}

#' Frank-Wolfe optimizer for simplex-constrained mixture weights
#'
#' Each step finds the simplex vertex minimising the inner product with the
#' gradient, then moves toward it with an adaptive step size
#' `(c / (t + c))^alpha`. Supports batched simplexes of shape (R, C).
#'
#' @param params List of tensors to optimise.
#' @param step Step-size constant c. Default: 2.0.
#' @param alpha Step-size decay exponent. Default: 1.0.
#' @param ... Ignored (for compatible signatures).
#' @export
optim_frank_wolfe <- optimizer(
  initialize = function(params, step = 2.0, alpha = 1.0, ...) {
    defaults <- list(step = step, alpha = alpha)
    super$initialize(params, defaults)
    private$iteration <- 0
  },

  step = function(closure = NULL) {
    private$iteration <- private$iteration + 1
    loss <- NULL
    if (!is.null(closure)) {
      loss <- closure()
    }

    with_no_grad({
      for (group in self$param_groups) {
        for (param in group$params) {
          if (is.null(param$grad) || is_undefined_tensor(param$grad)) {
            next
          }
          grad <- param$grad

          if (length(param$shape) == 1) {
            best_idx <- grad$argmin()
            s <- torch_zeros_like(param)
            s[best_idx] <- 1.0
          } else {
            best_idx <- grad$argmin(dim = 2, keepdim = TRUE)
            s <- torch_zeros_like(param)
            s$scatter_(2, best_idx, 1.0)
          }

          gamma <- (group$step / (private$iteration + group$step))^group$alpha
          param$mul_(1 - gamma)$add_(s, alpha = gamma)
        }
      }
    })
    loss
  },

  private = list(iteration = NULL)
)

#' Projected gradient descent optimizer for simplex-constrained mixture weights
#'
#' Projects the gradient onto the tangent space of the simplex (removes the
#' component perpendicular to the simplex and zeroes gradients pointing away
#' from active boundary constraints), takes a gradient step, then re-projects
#' back onto the simplex by clamping and renormalising. Supports momentum and
#' batched simplexes of shape (R, C).
#'
#' @param params List containing a single tensor of shape (R, C).
#' @param lr Learning rate. Default: 0.01.
#' @param momentum Momentum coefficient. 0 disables momentum. Default: 0.0.
#' @param eps Minimum weight after clamping, to avoid exact zeros. Default: 1e-8.
#' @param ... Ignored.
#' @export
optim_projected_gd <- optimizer(
  initialize = function(params, lr = 0.01, momentum = 0.0, eps = 1e-8, ...) {
    defaults <- list(lr = lr, momentum = momentum, eps = eps)
    super$initialize(params, defaults)
    if (momentum > 0) private$momentum_buffer <- NULL
  },

  step = function(closure = NULL) {
    param <- self$param_groups[[1]]$params[[1]]
    grad <- param$grad
    lr <- self$param_groups[[1]]$lr
    eps <- self$param_groups[[1]]$eps
    momentum <- self$param_groups[[1]]$momentum

    grad_mean <- grad$mean(dim = 2, keepdim = TRUE)
    grad_projected <- grad - grad_mean

    at_boundary <- param <= 0
    grad_projected <- torch_where(
      at_boundary & (grad_projected > 0),
      torch_zeros_like(grad_projected),
      grad_projected
    )

    if (momentum > 0) {
      if (is.null(private$momentum_buffer)) {
        private$momentum_buffer <- torch_zeros_like(grad_projected)
      }
      buf <- private$momentum_buffer
      buf$mul_(momentum)$add_(grad_projected)
      grad_projected <- buf
    }

    param$add_(grad_projected, alpha = -lr)$clamp_(min = eps)
    param$div_(param$sum(dim = 2, keepdim = TRUE))
  },

  private = list(momentum_buffer = NULL)
)

# Thin wrapper so callers pass `gamma` rather than knowing torch's lr_step API.
lr_step <- function(op, step_size, gamma, ...) {
  torch_lr_step(op, step_size = step_size, gamma = gamma)
}

#' Minimise max_theta E_theta[Q(X)/P_w(X)] over mixture weights via parallel restarts
#'
#' Runs `n_restarts` simultaneous gradient-descent optimisations from random
#' Dirichlet initialisations. Pre-computes the count tensor, log-PMFs, and
#' per-component log-denominators once, then updates all restarts in parallel
#' each iteration. Tracks the best weights found across all iterations.
#'
#' @param n Total number of trials per observation.
#' @param log_theta Tensor of shape (m, T) — log DGP probabilities.
#' @param log_q Tensor of shape (m,) — log numerator probabilities.
#' @param log_ws Tensor of shape (m, C) — log mixture component probabilities.
#' @param optim_fn Optimizer constructor, e.g. `optim_adam` or
#'   [optim_frank_wolfe()]. Called as `optim_fn(params, ...)`.
#' @param use_softmax If `TRUE`, optimise unconstrained logits and apply softmax
#'   before each forward pass. Default: `FALSE`.
#' @param n_restarts Number of parallel random restarts. Default: 10.
#' @param tol Convergence threshold on max absolute loss change across restarts.
#'   Default: 1e-6.
#' @param batch_size Iterations per batch (also the LR scheduler step interval).
#'   Default: 1000.
#' @param n_batches Maximum number of batches. Default: 100.
#' @param eps Minimum weight when not using softmax. Default: 0.
#' @param ... Forwarded to `optim_fn` and `lr_step` (e.g. `lr`, `gamma`).
#' @return List with components:
#'   - `weights`: tensor of shape (R, C), best weights per restart.
#'   - `final_loss`: tensor of shape (R,), best loss achieved per restart.
#'   - `loss_history`: tensor of shape (iterations/100, R).
#'   - `expectation_profile`: tensor of shape (R, T), E_theta[Q(X)/P_w(X)] per
#'     restart at final weights.
#' @export
optimize_mixture_weights <- function(
  n,
  log_theta,
  log_q,
  log_ws,
  optim_fn,
  use_softmax = FALSE,
  n_restarts = 10,
  tol = 1e-6,
  batch_size = 1000,
  n_batches = 100,
  eps = 0.0,
  emit_fn = message,
  ...
) {
  X <- build_counts_tensor(n)
  M_ <- X$size(1)
  C_ <- log_ws$size(2)
  T_ <- log_theta$size(2)

  log_pmf <- mnom_logpmf(X, log_theta, n)
  llr_num <- matmul_0_ninf(X, log_q$unsqueeze(2))$squeeze(2)
  log_comp_denoms <- matmul_0_ninf(X, log_ws)

  best_losses <- torch_full(n_restarts, Inf, device = device, dtype = dtype)
  best_weights <- torch_full(
    c(n_restarts, C_),
    0,
    device = device,
    dtype = dtype
  )

  max_iter <- batch_size * n_batches
  track_freq <- 100
  losses_history <- torch_full(
    c(ceiling(max_iter / track_freq), n_restarts),
    Inf,
    device = device,
    dtype = dtype
  )
  current_losses <- losses_history[1, ]

  if (use_softmax) {
    unconstrained_weights <- torch_randn(
      c(n_restarts, C_),
      device = device,
      dtype = dtype,
      requires_grad = FALSE
    )
    optimizer <- optim_fn(list(unconstrained_weights), ...)
  } else {
    weights <- rdirichlet(n_restarts, rep(1, C_)) |>
      torch_tensor(device = device, dtype = dtype, requires_grad = FALSE)
    optimizer <- optim_fn(list(weights), ...)
  }

  scheduler <- lr_step(optimizer, step_size = 1, ...)

  for (batch_idx in seq_len(n_batches)) {
    for (i in seq_len(batch_size)) {
      iter <- (batch_idx - 1) * batch_size + i
      optimizer$zero_grad()

      with_no_grad({
        if (use_softmax) {
          weights <- nnf_softmax(unconstrained_weights, dim = 2)
          log_wts <- weights$log()
        } else {
          log_wts <- weights$log()
        }

        log_denoms_expanded <- add_ninf_any(
          log_comp_denoms$unsqueeze(2)$expand(c(M_, n_restarts, C_)),
          log_wts$unsqueeze(1)$expand(c(M_, n_restarts, C_))
        )
        llr_denom <- torch_logsumexp(log_denoms_expanded, dim = 3)
        llr <- llr_num$unsqueeze(2)$expand(c(M_, n_restarts)) - llr_denom

        log_pmf_llr <- add_ninf_any(
          log_pmf$unsqueeze(3)$expand(c(M_, T_, n_restarts)),
          llr$unsqueeze(2)$expand(c(M_, T_, n_restarts))
        )
        exp_llr <- torch_logsumexp(log_pmf_llr, dim = 1)$exp()

        max_result <- exp_llr$max(dim = 1)
        argmax_exp_llr <- max_result[[2]]
        max_exp_llr <- max_result[[1]]

        prev_losses <- current_losses
        current_losses <- max_exp_llr
        improved <- current_losses < best_losses
        best_losses <- torch_where(improved, current_losses, best_losses)
        best_weights[improved, ] <- weights[improved, ]
        if (iter %% track_freq == 0) {
          losses_history[iter / track_freq, ] <- current_losses
        }

        worst_log_pmf <- log_pmf$index_select(2, argmax_exp_llr)
        log_grad_terms <- (worst_log_pmf$unsqueeze(3) +
          llr$unsqueeze(3) +
          log_comp_denoms$unsqueeze(2) -
          llr_denom$unsqueeze(3))
        grad_w <- -log_grad_terms$logsumexp(dim = 1)$exp()
        if (use_softmax) {
          grad_u <- weights *
            (grad_w - (grad_w * weights)$sum(dim = 2, keepdim = TRUE))
        }
      })

      if (use_softmax) {
        unconstrained_weights$grad <- grad_u
      } else {
        weights$grad <- grad_w
      }
      optimizer$step()
    }

    scheduler$step()

    emit_fn(sprintf(
      "Batch %03d/%03d: best=%.8f worst=%.8f mean=%.8f lr=%.2e",
      batch_idx,
      n_batches,
      best_losses$min()$item(),
      best_losses$max()$item(),
      best_losses$mean()$item(),
      optimizer$param_groups[[1]]$lr
    ))

    if (batch_idx > 1) {
      max_change <- (prev_losses - current_losses)$abs()$max()$item()
      if (max_change < tol) {
        emit_fn(sprintf("Converged at batch %03d", batch_idx))
        break
      }
    }
  }

  with_no_grad({
    log_wts_final <- best_weights$log()
    log_denoms_final <- add_ninf_any(
      log_comp_denoms$unsqueeze(2)$expand(c(M_, n_restarts, C_)),
      log_wts_final$unsqueeze(1)$expand(c(M_, n_restarts, C_))
    )
    llr_denom_final <- torch_logsumexp(log_denoms_final, dim = 3)
    llr_final <- llr_num$unsqueeze(2)$expand(c(M_, n_restarts)) - llr_denom_final
    log_pmf_llr_final <- add_ninf_any(
      log_pmf$unsqueeze(3)$expand(c(M_, T_, n_restarts)),
      llr_final$unsqueeze(2)$expand(c(M_, T_, n_restarts))
    )
    expectation_profile <- torch_logsumexp(log_pmf_llr_final, dim = 1)$exp()$t()
  })

  list(
    weights = best_weights,
    final_loss = best_losses,
    loss_history = losses_history,
    expectation_profile = expectation_profile
  )
}

#' Run RIPr optimisation given pre-built grids
#'
#' High-level wrapper around [optimize_mixture_weights()] that accepts grids as
#' plain R lists of probability vectors and handles the conversion to
#' log-tensors internally.
#'
#' @param n Total number of trials per observation.
#' @param thetas R list of numeric vectors — DGP grid, e.g. from
#'   [make_simplex_grid()].
#' @param q Numeric vector of length m — the numerator distribution Q.
#' @param ws R list of numeric vectors — mixture component grid, e.g. from
#'   [make_simplex_grid()].
#' @param n_restarts Number of parallel random restarts.
#' @param optim_fn Optimizer constructor. Called as `optim_fn(params, ...)`.
#' @param batch_size Iterations per batch. Default: 1000.
#' @param n_batches Maximum number of batches. Default: 250.
#' @param tol Convergence tolerance. Default: 1e-8.
#' @param use_softmax Optimise in unconstrained space via softmax. Default: `TRUE`.
#' @param ... Forwarded to `optim_fn` and the LR scheduler (e.g. `lr`, `gamma`).
#' @return List with `weights`, `final_loss`, `loss_history`, and `expectation_profile`.
#' @export
run_ripr <- function(
  n,
  thetas,
  q,
  ws,
  n_restarts,
  optim_fn,
  batch_size = 1000,
  n_batches = 250,
  tol = 1e-8,
  use_softmax = TRUE,
  emit_fn = message,
  ...
) {
  to_log_tensor <- function(prob_list) {
    do.call(what = rbind, args = prob_list) |>
      torch_tensor(device = device, dtype = dtype) |>
      (\(x) x$transpose(1, 2))() |>
      (\(x) x$log())()
  }

  optimize_mixture_weights(
    n,
    log_theta = to_log_tensor(thetas),
    log_q = torch_tensor(q, device = device, dtype = dtype)$log(),
    log_ws = to_log_tensor(ws),
    optim_fn = optim_fn,
    use_softmax = use_softmax,
    n_restarts = n_restarts,
    batch_size = batch_size,
    n_batches = n_batches,
    tol = tol,
    emit_fn = emit_fn,
    ...
  )
}
