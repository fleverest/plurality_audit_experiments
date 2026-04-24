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
    torch_full_like,
    torch_where,
    torch_logsumexp,
    torch_cat
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
    samples[, i] <- stats::rgamma(n, shape = alpha[i], rate = 1)
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
  initialize = function(params, step = 2.0, alpha = 1.0, pairwise = FALSE, ...) {
    defaults <- list(step = step, alpha = alpha, pairwise = pairwise)
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
          gamma <- (group$step / (private$iteration + group$step))^group$alpha

          if (length(param$shape) == 1) {
            best_idx <- grad$argmin()
            if (group$pairwise) {
              grad_support <- torch_where(param > 0, grad, torch_full_like(grad, -Inf))
              away_idx <- grad_support$argmax()
              gamma <- min(gamma, param[away_idx]$item())
              d <- torch_zeros_like(param)
              d[best_idx] <- 1.0
              d[away_idx] <- d[away_idx] - 1.0
              param$add_(d, alpha = gamma)
            } else {
              s <- torch_zeros_like(param)
              s[best_idx] <- 1.0
              param$mul_(1 - gamma)$add_(s, alpha = gamma)
            }
          } else {
            best_idx <- grad$argmin(dim = 2, keepdim = TRUE)
            if (group$pairwise) {
              grad_support <- torch_where(param > 0, grad, torch_full_like(grad, -Inf))
              away_idx <- grad_support$argmax(dim = 2, keepdim = TRUE)
              gamma_eff <- param$gather(2, away_idx)$clamp(max = gamma)
              d <- torch_zeros_like(param)
              d$scatter_(2, best_idx, 1.0)
              d$scatter_add_(2, away_idx, torch_full_like(param$gather(2, away_idx), -1.0))
                param$add_(d * gamma_eff)
            } else {
              s <- torch_zeros_like(param)
              s$scatter_(2, best_idx, 1.0)
              param$mul_(1 - gamma)$add_(s, alpha = gamma)
            }
          }
        }
      }
    })
    loss
  },

  private = list(iteration = NULL)
)

#' Mirror descent (multiplicative weights) optimizer for simplex-constrained mixture weights
#'
#' Each step applies the multiplicative-weights update:
#'   w ← w * exp(-lr * ∇f(w)) / Z
#' where Z is the normalisation constant. This is mirror descent with KL
#' divergence as the Bregman divergence, and naturally keeps iterates in the
#' relative interior of the simplex (no weight ever reaches exactly zero).
#' Unlike Frank-Wolfe, the update is dense at every step, making it well-suited
#' when the optimum is an interior point.
#'
#' @param params List containing a single tensor of shape (R, C) or (C,).
#' @param lr Learning rate. Default: 0.01.
#' @param ... Ignored.
#' @export
optim_mirror_descent <- optimizer(
  initialize = function(params, lr = 0.01, ...) {
    defaults <- list(lr = lr)
    super$initialize(params, defaults)
  },

  step = function(closure = NULL) {
    loss <- NULL
    if (!is.null(closure)) loss <- closure()

    with_no_grad({
      for (group in self$param_groups) {
        for (param in group$params) {
          if (is.null(param$grad) || is_undefined_tensor(param$grad)) next
          grad <- param$grad
          lr <- group$lr

          if (length(param$shape) == 1) {
            param$mul_((-lr * grad)$exp())
            param$div_(param$sum())
          } else {
            param$mul_((-lr * grad)$exp())
            param$div_(param$sum(dim = 2, keepdim = TRUE))
          }
        }
      }
    })
    loss
  }
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

#' Minimise sum_theta max(0, E_theta[Q(X)/P_w(X)] - 1) over mixture weights via parallel restarts
#'
#' Runs `n_restarts` simultaneous optimisations from random Dirichlet
#' initialisations. Pre-computes the count tensor, log-PMFs, and per-component
#' log-denominators once, then updates all restarts in parallel each iteration.
#' Tracks the best weights found across all iterations.
#'
#' @param n Total number of trials per observation.
#' @param log_theta Tensor of shape (m, T) — log DGP probabilities.
#' @param log_q Tensor of shape (m,) — log numerator probabilities.
#' @param log_ws Tensor of shape (m, C) — log mixture component probabilities.
#' @param optim Function `optim(params, eval_fn = NULL)` that returns a step
#'   closure `function(loss)`. The closure is called every iteration with the
#'   current scalar loss; any value it returns invisibly is recorded in
#'   `tracked_history`. `eval_fn` is a function `eval_fn(log_wts)` that
#'   evaluates the objective for a batch of log-weight tensors — pass it to
#'   optimizers that need it (e.g. Frank-Wolfe with exact line search).
#'   Example (Adam + reduce-on-plateau, tracking lr):
#'   ```r
#'   optim = function(params, eval_fn = NULL) {
#'     op    <- optim_adam(params, lr = 0.01)
#'     sched <- lr_reduce_on_plateau(op, patience = 25, factor = 0.99)
#'     function(loss) {
#'       op$step()
#'       sched$step(loss)
#'       invisible(op$param_groups[[1]]$lr)
#'     }
#'   }
#'   ```
#'   Example (Frank-Wolfe with exact line search):
#'   ```r
#'   optim = function(params, eval_fn = NULL) {
#'     op <- optim_frank_wolfe(params, line_search = TRUE)
#'     function(loss) op$step(eval_fn = eval_fn)
#'   }
#'   ```
#' @param use_softmax If `TRUE`, optimise unconstrained logits and apply softmax
#'   before each forward pass. Default: `FALSE`.
#' @param n_restarts Number of parallel random restarts. Default: 10.
#' @param iters Total number of update steps. Default: 100000.
#' @param track_interval Record metrics every this many iterations. Default: 1000.
#' @param report_interval Emit a progress line every this many iterations. Default: 10000.
#' @param smooth_lambda Laplacian smoothness penalty weight. Penalises
#'   `sum_i (w_{i+1} - w_i)^2` along the grid ordering, encouraging smooth
#'   weight profiles. 0 disables (default).
#' @return List with components:
#'   - `weights`: tensor of shape (R, C), best weights per restart by surrogate
#'     loss (sum of thresholded expectations). Useful for convergence diagnostics.
#'   - `final_loss`: tensor of shape (R,), best surrogate loss per restart.
#'   - `weights_max_exp`: tensor of shape (R, C), best weights per restart by
#'     max_theta E_theta[Q(X)/P_w(X)] — the quantity of primary interest.
#'   - `final_max_exp`: tensor of shape (R,), best max expectation per restart.
#'   - `loss_history`: tensor of shape (floor(iters/track_interval), R).
#'   - `tracked_history`: numeric vector of length floor(iters/track_interval),
#'     invisible return value of the `optim` closure at each tracked iteration
#'     (e.g. current learning rate). `NA` when the closure returns nothing.
#'   - `expectation_profile`: tensor of shape (R, T), E_theta[Q(X)/P_w(X)] per
#'     restart at `weights`.
#'   - `expectation_profile_max_exp`: tensor of shape (R, T),
#'     E_theta[Q(X)/P_w(X)] per restart at `weights_max_exp`.
#' @export
optimize_mixture_weights <- function(
  n,
  log_theta,
  log_q,
  log_ws,
  optim,
  use_softmax = TRUE,
  n_restarts = 10,
  iters = 100000L,
  track_interval = 1000L,
  report_interval = 10000L,
  smooth_lambda = 0,
  emit_fn = message
) {
  X <- build_counts_tensor(n)
  M_ <- X$size(1)
  C_ <- log_ws$size(2)
  T_ <- log_theta$size(2)

  log_pmf <- mnom_logpmf(X, log_theta, n)
  llr_num <- matmul_0_ninf(X, log_q$unsqueeze(2))$squeeze(2)
  log_comp_denoms <- matmul_0_ninf(X, log_ws)

  # Forward pass: compute losses and intermediates for a batch of weight vectors.
  # log_wts: (n_restarts, C) log mixture weights.
  # Returns list(losses, exp_llr, max_exp_llr, above_mask, llr, llr_denom).
  eval_weights <- function(log_wts) {
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
    above_mask <- exp_llr > 1
    list(
      losses      = (exp_llr - 1)$clamp(min = 0)$sum(dim = 1L),
      exp_llr     = exp_llr,
      max_exp_llr = exp_llr$max(dim = 1L)[[1]],
      above_mask  = above_mask,
      llr         = llr,
      llr_denom   = llr_denom
    )
  }

  # Gradient of sum_theta max(0, E_theta[...] - 1) w.r.t. weights (and logits).
  # weights: (n_restarts, C) simplex weights.
  # above_mask, llr, llr_denom: intermediates from eval_weights.
  # Returns list(grad_w, grad_u) where grad_u is only present when use_softmax.
  compute_grad <- function(weights, above_mask, llr, llr_denom) {
    active_pmf_sum <- log_pmf$exp()$matmul(above_mask$to(dtype = dtype))
    log_grad_terms <- (active_pmf_sum$log()$unsqueeze(3) +
      llr$unsqueeze(3) +
      log_comp_denoms$unsqueeze(2) -
      llr_denom$unsqueeze(3))
    grad_w <- -log_grad_terms$logsumexp(dim = 1)$exp()

    if (smooth_lambda > 0) {
      # Path-graph Laplacian: penalises sum of squared first differences along
      # the grid ordering. Gradient is 2*lambda * L*w where L is tridiagonal.
      fd <- weights[, 2:C_] - weights[, 1:(C_ - 1L)]  # (R, C-1) first diffs
      lap_w <- torch_cat(list(
        -fd[, 1:1],
        fd[, 1:(C_ - 2L)] - fd[, 2:(C_ - 1L)],
        fd[, (C_ - 1L):(C_ - 1L)]
      ), dim = 2)
      grad_w <- grad_w + 2 * smooth_lambda * lap_w
    }

    if (use_softmax) {
      list(
        grad_w = grad_w,
        grad_u = weights * (grad_w - (grad_w * weights)$sum(dim = 2, keepdim = TRUE))
      )
    } else {
      list(grad_w = grad_w)
    }
  }

  best_losses   <- torch_full(n_restarts, Inf, device = device, dtype = dtype)
  best_weights  <- torch_full(c(n_restarts, C_), 0, device = device, dtype = dtype)
  best_max_exp  <- torch_full(n_restarts, Inf, device = device, dtype = dtype)
  best_weights_max_exp <- torch_full(c(n_restarts, C_), 0, device = device, dtype = dtype)

  n_snapshots <- iters %/% track_interval
  losses_history <- torch_full(
    c(n_snapshots, n_restarts), Inf, device = device, dtype = dtype
  )
  tracked_history <- rep(NA_real_, n_snapshots)

  if (use_softmax) {
    unconstrained_weights <- torch_randn(
      c(n_restarts, C_), device = device, dtype = dtype, requires_grad = FALSE
    )
    step_fn <- optim(list(unconstrained_weights), eval_fn = eval_weights)
  } else {
    weights <- rdirichlet(n_restarts, rep(1, C_)) |>
      torch_tensor(device = device, dtype = dtype, requires_grad = FALSE)
    step_fn <- optim(list(weights), eval_fn = eval_weights)
  }

  tryCatch(
    for (iter in seq_len(iters)) {
      if (use_softmax) weights <- nnf_softmax(unconstrained_weights, dim = 2)
      fwd <- eval_weights(weights$log())

      improved <- fwd$losses < best_losses
      best_losses <- torch_where(improved, fwd$losses, best_losses)
      best_weights[improved, ] <- weights[improved, ]

      improved_max <- fwd$max_exp_llr < best_max_exp
      best_max_exp <- torch_where(improved_max, fwd$max_exp_llr, best_max_exp)
      best_weights_max_exp[improved_max, ] <- weights[improved_max, ]

      grads <- compute_grad(weights, fwd$above_mask, fwd$llr, fwd$llr_denom)
      if (use_softmax) {
        unconstrained_weights$grad <- grads$grad_u
      } else {
        weights$grad <- grads$grad_w
      }

      tracked <- step_fn(fwd$losses$min())

      if (iter %% track_interval == 0) {
        idx <- iter %/% track_interval
        losses_history[idx, ] <- fwd$losses
        tracked_history[idx] <- if (is.null(tracked)) NA_real_ else as.numeric(tracked)
      }

      if (iter %% report_interval == 0) {
        reg_now <- if (smooth_lambda > 0) {
          fd <- weights[, 2:C_] - weights[, 1:(C_ - 1L)]
          (smooth_lambda * fd$pow(2)$sum(dim = 2L))$min()$item()
        } else 0
        emit_fn(sprintf(
          "Iter %06d/%06d: best_loss=%.6f best_max_exp=%.6f cur_loss=%.6f cur_reg=%.6f cur_max_exp=%.6f tracked=%.4g norm=%.3f",
          iter,
          iters,
          best_losses$min()$item(),
          best_max_exp$min()$item(),
          fwd$losses$min()$item(),
          reg_now,
          fwd$max_exp_llr$min()$item(),
          if (!is.null(tracked)) as.numeric(tracked) else NA_real_,
          if (use_softmax) unconstrained_weights$norm(dim = 2L)$mean()$item() else NA_real_
        ))
      }
    },
    interrupt = function(e) invisible(NULL)
  )

  list(
    weights                  = best_weights,
    final_loss               = best_losses,
    weights_max_exp          = best_weights_max_exp,
    final_max_exp            = best_max_exp,
    loss_history             = losses_history,
    tracked_history          = tracked_history,
    expectation_profile      = eval_weights(best_weights$log())$exp_llr$t(),
    expectation_profile_max_exp = eval_weights(best_weights_max_exp$log())$exp_llr$t()
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
#' @param optim Function `optim(params)` returning a step closure. See
#'   [optimize_mixture_weights()] for the full interface and tracking example.
#' @param use_softmax Optimise in unconstrained space via softmax. Default: `TRUE`.
#' @param iters Total number of update steps. Default: 100000.
#' @param track_interval Record metrics every this many iterations. Default: 1000.
#' @param report_interval Emit a progress line every this many iterations. Default: 10000.
#' @param smooth_lambda Laplacian smoothness penalty weight. See
#'   [optimize_mixture_weights()] for details. Default: 0.
#' @return List with `weights`, `final_loss`, `weights_max_exp`, `final_max_exp`,
#'   `loss_history`, `tracked_history`, `expectation_profile`, and
#'   `expectation_profile_max_exp`. See [optimize_mixture_weights()] for details.
#' @export
run_ripr <- function(
  n,
  thetas,
  q,
  ws,
  n_restarts,
  optim,
  use_softmax = TRUE,
  iters = 100000L,
  track_interval = 1000L,
  report_interval = 10000L,
  smooth_lambda = 0,
  emit_fn = message
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
    optim = optim,
    use_softmax = use_softmax,
    n_restarts = n_restarts,
    iters = iters,
    track_interval = track_interval,
    report_interval = report_interval,
    smooth_lambda = smooth_lambda,
    emit_fn = emit_fn
  )
}
