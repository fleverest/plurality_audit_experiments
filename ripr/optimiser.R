box::use(
  torch[
    optimizer,
    optim_adam,
    optim_lbfgs,
    with_no_grad,
    is_undefined_tensor,
    nnf_softmax,
    torch_randn,
    torch_full,
    torch_empty,
    torch_tensor,
    torch_zeros_like,
    torch_full_like,
    torch_where,
    torch_logsumexp,
    torch_amax,
    torch_cat,
    torch_isneginf
  ],
  ripr / torch_settings[device, dtype],
  ripr / tensor_ops[matmul_0_ninf, add_ninf_any, add_ninf_any_, logsumexp_inplace_],
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

# Golden section search for the minimum of a convex function f on [lo, hi].
gss <- function(f, lo, hi, tol = 1e-6) {
  phi <- (sqrt(5) - 1) / 2
  x1 <- hi - phi * (hi - lo)
  x2 <- lo + phi * (hi - lo)
  f1 <- f(x1)
  f2 <- f(x2)
  while ((hi - lo) > tol) {
    if (f1 < f2) {
      hi <- x2; x2 <- x1; f2 <- f1
      x1 <- hi - phi * (hi - lo); f1 <- f(x1)
    } else {
      lo <- x1; x1 <- x2; f1 <- f2
      x2 <- lo + phi * (hi - lo); f2 <- f(x2)
    }
  }
  (lo + hi) / 2
}

#' Frank-Wolfe optimizer for simplex-constrained mixture weights
#'
#' Each step finds the simplex vertex minimising the inner product with the
#' gradient (the FW direction), then moves toward it. Supports batched
#' simplexes of shape (R, C).
#'
#' Step size: by default uses the schedule `(step / (t + step))^alpha`.
#' When `line_search = TRUE` and `eval_fn` is passed to `$step()`, uses golden
#' section search for the exact minimiser along the FW direction instead.
#' For batched (R, C) params the line search minimises the SUM of losses across
#' restarts, so all restarts share the same γ at each iteration.
#'
#' @param params List of tensors to optimise.
#' @param step Step-size constant c (schedule mode only). Default: 2.0.
#' @param alpha Step-size decay exponent (schedule mode only). Default: 1.0.
#' @param pairwise If `TRUE`, use the pairwise FW variant. Default: `FALSE`.
#' @param line_search If `TRUE`, use golden section search instead of the
#'   schedule. Requires `eval_fn` to be passed to `$step()`. Default: `FALSE`.
#' @param ls_tol Convergence tolerance for the golden section search. Default: 1e-6.
#' @param ... Ignored.
#' @export
optim_frank_wolfe <- optimizer(
  initialize = function(params, step = 2.0, alpha = 1.0, pairwise = FALSE,
                        line_search = FALSE, ls_tol = 1e-6, ...) {
    defaults <- list(
      step = step, alpha = alpha, pairwise = pairwise,
      line_search = line_search, ls_tol = ls_tol
    )
    super$initialize(params, defaults)
    private$iteration <- 0
  },

  step = function(closure = NULL, eval_fn = NULL) {
    private$iteration <- private$iteration + 1
    loss <- NULL
    if (!is.null(closure)) loss <- closure()

    with_no_grad({
      for (group in self$param_groups) {
        for (param in group$params) {
          if (is.null(param$grad) || is_undefined_tensor(param$grad)) next
          grad <- param$grad
          use_ls <- group$line_search && !is.null(eval_fn)

          if (length(param$shape) == 1) {
            # 1-D case: single simplex of shape (C,)
            best_idx <- grad$argmin()
            s <- torch_zeros_like(param)
            s[best_idx] <- 1.0

            if (group$pairwise) {
              grad_support <- torch_where(param > 0, grad, torch_full_like(grad, -Inf))
              away_idx <- grad_support$argmax()
              gamma_max <- param[away_idx]$item()
              d <- torch_zeros_like(param)
              d[best_idx] <- 1.0
              d[away_idx] <- d[away_idx] - 1.0
              gamma <- if (use_ls) {
                gss(
                  function(g) eval_fn((param + g * d)$log())$losses$sum()$item(),
                  0, gamma_max, tol = group$ls_tol
                )
              } else {
                min((group$step / (private$iteration + group$step))^group$alpha, gamma_max)
              }
              param$add_(d, alpha = gamma)
            } else {
              gamma <- if (use_ls) {
                gss(
                  function(g) eval_fn(((1 - g) * param + g * s)$log())$losses$sum()$item(),
                  0, 1, tol = group$ls_tol
                )
              } else {
                (group$step / (private$iteration + group$step))^group$alpha
              }
              param$mul_(1 - gamma)$add_(s, alpha = gamma)
            }
          } else {
            # 2-D batched case: shape (R, C)
            best_idx <- grad$argmin(dim = 2, keepdim = TRUE)
            s <- torch_zeros_like(param)
            s$scatter_(2, best_idx, 1.0)

            if (group$pairwise) {
              grad_support <- torch_where(param > 0, grad, torch_full_like(grad, -Inf))
              away_idx <- grad_support$argmax(dim = 2, keepdim = TRUE)
              d <- torch_zeros_like(param)
              d$scatter_(2, best_idx, 1.0)
              d$scatter_add_(2, away_idx, torch_full_like(param$gather(2, away_idx), -1.0))

              if (use_ls) {
                # Clamp each restart's step to its own away weight, so all
                # remain on the simplex. Search over [0,1]; each f_r is still
                # convex (decreasing-then-flat), so the sum is convex in g.
                gamma_max_r <- param$gather(2, away_idx)  # (R, 1)
                gamma <- gss(
                  function(g) {
                    gamma_eff <- gamma_max_r$clamp(max = g)
                    eval_fn((param + gamma_eff * d)$log())$losses$sum()$item()
                  },
                  0, 1, tol = group$ls_tol
                )
                param$add_(d * gamma_max_r$clamp(max = gamma))
              } else {
                gamma <- (group$step / (private$iteration + group$step))^group$alpha
                gamma_eff <- param$gather(2, away_idx)$clamp(max = gamma)
                param$add_(d * gamma_eff)
              }
            } else {
              gamma <- if (use_ls) {
                gss(
                  function(g) eval_fn(((1 - g) * param + g * s)$log())$losses$sum()$item(),
                  0, 1, tol = group$ls_tol
                )
              } else {
                (group$step / (private$iteration + group$step))^group$alpha
              }
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

# ---------------------------------------------------------------------------
# Closure factory factories for the `optim` argument of run_ripr
# ---------------------------------------------------------------------------

#' Adam closure factory
#'
#' Returns a closure factory for [run_ripr()] that optimises with Adam.
#' Always tracks the current learning rate via `tracked_history`.
#'
#' @param lr Initial learning rate. Default: 0.01.
#' @param scheduler Optional `function(optimizer)` that wraps the Adam
#'   optimizer in a torch `lr_scheduler`. The scheduler's `$step()` is called
#'   after each parameter update. For schedulers that require a loss value
#'   (e.g. `lr_reduce_on_plateau`), pass `loss = TRUE`; the current loss tensor
#'   is forwarded automatically.
#' @param loss If `TRUE`, passes the current loss to `scheduler$step()`.
#'   Default: `FALSE`.
#' @param restore_best If `TRUE` and a scheduler is provided, restores the
#'   parameters to the best-seen weights whenever the scheduler reduces the
#'   learning rate. Useful with `lr_reduce_on_plateau` to resume from the best
#'   checkpoint after each reduction. Default: `FALSE`.
#' @return A closure factory.
#' @examples
#' \dontrun{
#' run_ripr(..., optim = adam(lr = 0.01))
#' run_ripr(..., optim = adam(
#'   lr = 0.01,
#'   scheduler = function(op) lr_reduce_on_plateau(op, patience = 25, factor = 0.99),
#'   loss = TRUE,
#'   restore_best = TRUE
#' ))
#' }
#' @export
adam <- function(lr = 0.01, scheduler = NULL, loss = FALSE, restore_best = FALSE) {
  function(params, fwd_fn = NULL, bwd_fn = NULL) {
    op    <- optim_adam(params, lr = lr)
    sched <- if (!is.null(scheduler)) scheduler(op) else NULL
    best_loss    <- Inf
    best_weights <- NULL
    function(current_loss) {
      if (restore_best && !is.null(sched)) {
        cl <- as.numeric(current_loss)
        if (cl < best_loss) {
          best_loss    <<- cl
          best_weights <<- params[[1]]$detach()$clone()
        }
      }
      prev_lr <- op$param_groups[[1]]$lr
      op$step()
      if (!is.null(sched)) {
        if (loss) sched$step(current_loss) else sched$step()
        if (restore_best && op$param_groups[[1]]$lr < prev_lr) {
          with_no_grad(params[[1]]$copy_(best_weights))
        }
      }
      invisible(op$param_groups[[1]]$lr)
    }
  }
}

#' Frank-Wolfe closure factory (scheduled step size)
#'
#' Returns a closure factory for [run_ripr()] using the Frank-Wolfe algorithm
#' with the step size schedule `gamma_t = (step / (t + step))^alpha`.
#'
#' @param step Step-size constant. Default: 2.
#' @param alpha Decay exponent. Default: 1.
#' @param pairwise Use pairwise FW variant. Default: `FALSE`.
#' @return A closure factory.
#' @examples
#' \dontrun{
#' run_ripr(..., optim = fw(), use_softmax = FALSE)
#' run_ripr(..., optim = fw(step = 50, pairwise = TRUE), use_softmax = FALSE)
#' }
#' @export
fw <- function(step = 2, alpha = 1, pairwise = FALSE) {
  function(params, fwd_fn = NULL, bwd_fn = NULL) {
    op <- optim_frank_wolfe(params, step = step, alpha = alpha, pairwise = pairwise)
    function(loss) op$step()
  }
}

#' Frank-Wolfe closure factory (exact line search)
#'
#' Returns a closure factory for [run_ripr()] using Frank-Wolfe with golden
#' section search for the exact step size. Requires `fwd_fn` provided by the
#' optimisation loop.
#'
#' @param pairwise Use pairwise FW variant. Default: `FALSE`.
#' @param ls_tol GSS convergence tolerance. Default: 1e-6.
#' @return A closure factory.
#' @examples
#' \dontrun{
#' run_ripr(..., optim = fw_ls(), use_softmax = FALSE)
#' run_ripr(..., optim = fw_ls(pairwise = TRUE), use_softmax = FALSE)
#' }
#' @export
fw_ls <- function(pairwise = FALSE, ls_tol = 1e-6) {
  function(params, fwd_fn = NULL, bwd_fn = NULL) {
    op <- optim_frank_wolfe(params, pairwise = pairwise, line_search = TRUE, ls_tol = ls_tol)
    function(loss) op$step(eval_fn = fwd_fn)
  }
}

#' Mirror descent closure factory
#'
#' Returns a closure factory for [run_ripr()] using the multiplicative-weights
#' (mirror descent) update. Always tracks the current learning rate via
#' `tracked_history`.
#'
#' @param lr Initial learning rate. Default: 0.01.
#' @param scheduler Optional `function(optimizer)` wrapping the optimizer in a
#'   torch `lr_scheduler`. See [adam()] for details.
#' @param loss If `TRUE`, passes the current loss to `scheduler$step()`.
#'   Default: `FALSE`.
#' @return A closure factory.
#' @examples
#' \dontrun{
#' run_ripr(..., optim = mirror(lr = 0.001), use_softmax = FALSE)
#' run_ripr(..., optim = mirror(
#'   lr = 0.01,
#'   scheduler = function(op) lr_cosine_annealing(op, T_max = 100000L)
#' ), use_softmax = FALSE)
#' }
#' @export
mirror <- function(lr = 0.01, scheduler = NULL, loss = FALSE) {
  function(params, fwd_fn = NULL, bwd_fn = NULL) {
    op    <- optim_mirror_descent(params, lr = lr)
    sched <- if (!is.null(scheduler)) scheduler(op) else NULL
    function(current_loss) {
      op$step()
      if (!is.null(sched)) {
        if (loss) sched$step(current_loss) else sched$step()
      }
      invisible(op$param_groups[[1]]$lr)
    }
  }
}

#' L-BFGS closure factory
#'
#' Returns a closure factory for [run_ripr()] using L-BFGS. The step closure
#' passes a full forward+backward closure to L-BFGS, which re-evaluates it
#' during its internal line search. Requires `use_softmax = TRUE` — L-BFGS
#' operates in unconstrained logit space and applies softmax internally.
#' All restarts are optimised jointly (L-BFGS minimises the sum of losses
#' across restarts and approximates a single shared Hessian).
#'
#' @param lr Step size. Default: 1.
#' @param max_iter Maximum L-BFGS iterations per step. Default: 20.
#' @param history_size Number of curvature pairs to retain. Default: 100.
#' @return A closure factory.
#' @examples
#' \dontrun{
#' run_ripr(..., optim = lbfgs(), use_softmax = TRUE)
#' }
#' @export
lbfgs <- function(lr = 1, max_iter = 20, history_size = 100) {
  function(params, fwd_fn = NULL, bwd_fn = NULL) {
    op <- optim_lbfgs(params, lr = lr, max_iter = max_iter, history_size = history_size)
    function(loss) {
      op$step(closure = function() {
        op$zero_grad()
        w   <- nnf_softmax(params[[1]], dim = 2)
        fwd <- fwd_fn(w$log())
        g   <- bwd_fn(w, fwd$above_mask, fwd$llr, fwd$llr_denom)
        params[[1]]$grad <- g$grad_u
        fwd$losses$sum()
      })
    }
  }
}

# ---------------------------------------------------------------------------

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
#' @param optim Closure factory: `function(params, fwd_fn = NULL, bwd_fn = NULL)`
#'   returning a step closure `function(loss)`. The step closure is called every
#'   iteration with the current min loss tensor; any value it returns invisibly
#'   is recorded in `tracked_history`. Use the prebuilt factories [adam()],
#'   [fw()], [fw_ls()], [mirror()], [lbfgs()], or define your own — see section
#'   *Custom optimisers* below.
#' @param use_softmax If `TRUE`, optimise unconstrained logits and apply softmax
#'   before each forward pass. Default: `FALSE`.
#' @param n_restarts Number of parallel random restarts. Default: 10.
#' @param iters Total number of update steps. Default: 100000.
#' @param track_interval Record metrics every this many iterations. Default: 1000.
#' @param report_interval Emit a progress line every this many iterations. Default: 10000.
#' @param init_weights Optional numeric matrix or tensor of shape `(n_restarts, C)`
#'   to use as the starting point instead of random initialisation. Pass the
#'   `weights` or `weights_max_exp` tensor from a previous result directly.
#' @param lin_smooth Path-Laplacian smoothness penalty on raw weights. Default: 0.
#' @param log_smooth Path-Laplacian smoothness penalty on log-weights. Default: 0.
#' @param early_stopping_patience Stop early if no meaningful improvement is seen
#'   for this many consecutive iterations. Checked at each `track_interval`
#'   checkpoint, so the effective resolution is `track_interval` iterations.
#'   `NULL` disables early stopping. Default: `NULL`.
#' @param early_stopping_tol Relative improvement threshold for early stopping.
#'   An interval counts as "no improvement" unless the global best loss drops by
#'   more than `early_stopping_tol * prev_best`. Ignored when
#'   `early_stopping_patience` is `NULL`. Default: `1e-4`.
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
#' @section Custom optimisers:
#'   A custom factory has the signature
#'   `function(params, fwd_fn = NULL, bwd_fn = NULL)` and returns a step
#'   closure `function(loss)`.
#'
#'   `fwd_fn(log_wts)` is the forward pass: given log mixture weights of shape
#'   (R, C) it returns the full evaluation list (`losses`, `exp_llr`,
#'   `above_mask`, `llr`, `llr_denom`). Most first-order methods can ignore it.
#'
#'   `bwd_fn(weights, above_mask, llr, llr_denom)` computes gradients and
#'   returns `list(grad_w, grad_u)`. It is needed only when the optimiser must
#'   re-evaluate loss and gradient on demand — specifically when passing a
#'   closure to L-BFGS, which calls back into the forward+backward pass during
#'   its internal line search.
#'
#'   Example — Adam with reduce-on-plateau, tracking the learning rate:
#'   ```r
#'   function(params, fwd_fn = NULL, bwd_fn = NULL) {
#'     op    <- optim_adam(params, lr = 0.01)
#'     sched <- lr_reduce_on_plateau(op, patience = 25, factor = 0.99)
#'     function(loss) {
#'       op$step()
#'       sched$step(loss)
#'       invisible(op$param_groups[[1]]$lr)
#'     }
#'   }
#'   ```
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
  init_weights = NULL,
  lin_smooth = 0,
  log_smooth = 0,
  early_stopping_patience = NULL,
  early_stopping_tol = 1e-4,
  emit_fn = message,
  monitor_fn = NULL
) {
  X <- build_counts_tensor(n)
  M_ <- X$size(1)
  C_ <- log_ws$size(2)
  T_ <- log_theta$size(2)

  log_pmf <- mnom_logpmf(X, log_theta, n)
  llr_num <- matmul_0_ninf(X, log_q$unsqueeze(2))$squeeze(2)
  log_comp_denoms <- matmul_0_ninf(X, log_ws)

  buf_MRC <- torch_empty(c(M_, n_restarts, C_), device = device, dtype = dtype)
  buf_MTR <- torch_empty(c(M_, T_, n_restarts), device = device, dtype = dtype)
  buf_MT  <- torch_empty(c(M_, T_), device = device, dtype = dtype)

  logpmf_ninf <- torch_isneginf(log_pmf)
  logcd_ninf  <- torch_isneginf(log_comp_denoms)

  # Forward pass: compute losses and intermediates for a batch of weight vectors.
  # log_wts: (n_restarts, C) log mixture weights.
  # Returns list(losses, exp_llr, max_exp_llr, above_mask, llr, llr_denom).
  eval_weights <- function(log_wts) {
    add_ninf_any_(buf_MRC, log_comp_denoms$unsqueeze(2L), log_wts$unsqueeze(1L), logcd_ninf$unsqueeze(2L))
    llr_denom <- logsumexp_inplace_(buf_MRC, dim = 3L)
    llr <- llr_num$unsqueeze(2L) - llr_denom
    add_ninf_any_(buf_MTR, log_pmf$unsqueeze(3L), llr$unsqueeze(2L), logpmf_ninf$unsqueeze(3L))
    exp_llr <- logsumexp_inplace_(buf_MTR, dim = 1L)$exp()
    above_mask <- exp_llr > 1
    base_losses <- (exp_llr - 1)$clamp(min = 0)$sum(dim = 1L)
    penalty <- torch_zeros_like(base_losses)
    if (lin_smooth > 0 || log_smooth > 0) {
      w <- log_wts$exp()
      if (lin_smooth > 0) {
        fd      <- w[, 2:C_] - w[, 1:(C_ - 1L)]
        penalty <- penalty + lin_smooth * fd$pow(2)$sum(dim = 2L)
      }
      if (log_smooth > 0) {
        fd      <- log_wts[, 2:C_] - log_wts[, 1:(C_ - 1L)]
        penalty <- penalty + log_smooth * fd$pow(2)$sum(dim = 2L)
      }
    }
    list(
      losses      = base_losses + penalty,
      base_losses = base_losses,
      exp_llr     = exp_llr,
      max_exp_llr = torch_amax(exp_llr, dim = 1L),
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
    buf_MT$copy_(log_pmf)$exp_()
    active_pmf_sum <- buf_MT$matmul(above_mask$to(dtype = dtype))
    buf_MRC$copy_(active_pmf_sum$log()$unsqueeze(3L))
    buf_MRC$add_(llr$unsqueeze(3L))
    buf_MRC$add_(log_comp_denoms$unsqueeze(2L))
    buf_MRC$sub_(llr_denom$unsqueeze(3L))
    grad_w <- -logsumexp_inplace_(buf_MRC, dim = 1L)$exp()

    if (lin_smooth > 0 || log_smooth > 0) {
      # Path-graph Laplacian penalty. `lin_smooth` adds an L2 penalty on
      # adjacent weight differences; `log_smooth` adds the same penalty on
      # log-weights. Both penalties encourage smoothness in the weight
      # distribution, which can help with convergence if the RIPr is indeed
      # smooth.
      # `lin_smooth` has a stronger effect on the high-density modes, while
      # `log_smooth` encourages smoothness in the tails.
      if (lin_smooth > 0) {
        fd <- weights[, 2:C_] - weights[, 1:(C_ - 1L)]
        lap_v <- torch_cat(list(
          -fd[, 1:1],
          fd[, 1:(C_ - 2L)] - fd[, 2:(C_ - 1L)],
          fd[, (C_ - 1L):(C_ - 1L)]
        ), dim = 2)
        grad_w <- grad_w + 2 * lin_smooth * lap_v
      }
      if (log_smooth > 0) {
        log_weights <- weights$log()
        fd <- log_weights[, 2:C_] - log_weights[, 1:(C_ - 1L)]
        lap_v <- torch_cat(list(
          -fd[, 1:1],
          fd[, 1:(C_ - 2L)] - fd[, 2:(C_ - 1L)],
          fd[, (C_ - 1L):(C_ - 1L)]
        ), dim = 2)
        grad_w <- grad_w + 2 * log_smooth * lap_v / weights
      }
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

  best_losses        <- torch_full(n_restarts, Inf, device = device, dtype = dtype)
  best_base_losses   <- torch_full(n_restarts, Inf, device = device, dtype = dtype)
  best_weights       <- torch_full(c(n_restarts, C_), 0, device = device, dtype = dtype)
  best_max_exp       <- torch_full(n_restarts, Inf, device = device, dtype = dtype)
  best_weights_max_exp <- torch_full(c(n_restarts, C_), 0, device = device, dtype = dtype)

  n_snapshots <- iters %/% track_interval
  losses_history <- torch_full(
    c(n_snapshots, n_restarts), Inf, device = device, dtype = dtype
  )
  tracked_history <- rep(NA_real_, n_snapshots)

  as_weight_tensor <- function(w) {
    if (inherits(w, "torch_tensor")) {
      w$to(device = device, dtype = dtype)$detach()
    } else {
      torch_tensor(w, device = device, dtype = dtype)
    }
  }

  if (!is.null(init_weights)) {
    iw <- as_weight_tensor(init_weights)
    if (!identical(as.integer(iw$shape), as.integer(c(n_restarts, C_)))) {
      stop(sprintf(
        "init_weights must have shape (%d, %d); got (%s)",
        n_restarts, C_, paste(iw$shape, collapse = ", ")
      ))
    }
  }

  if (use_softmax) {
    unconstrained_weights <- if (!is.null(init_weights)) {
      as_weight_tensor(init_weights)$log()
    } else {
      torch_randn(c(n_restarts, C_), device = device, dtype = dtype)
    }
    unconstrained_weights$requires_grad_(FALSE)
    step_fn <- optim(list(unconstrained_weights), fwd_fn = eval_weights, bwd_fn = compute_grad)
  } else {
    weights <- if (!is.null(init_weights)) {
      as_weight_tensor(init_weights)
    } else {
      rdirichlet(n_restarts, rep(1, C_)) |>
        torch_tensor(device = device, dtype = dtype)
    }
    weights$requires_grad_(FALSE)
    step_fn <- optim(list(weights), fwd_fn = eval_weights, bwd_fn = compute_grad)
  }

  es_global_best     <- Inf
  es_iters_no_improv <- 0L

  tryCatch(
    for (iter in seq_len(iters)) {
      if (use_softmax) weights <- nnf_softmax(unconstrained_weights, dim = 2)
      fwd <- eval_weights(weights$log())

      improved <- fwd$losses < best_losses
      best_losses      <- torch_where(improved, fwd$losses,      best_losses)
      best_base_losses <- torch_where(improved, fwd$base_losses, best_base_losses)
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

        if (!is.null(early_stopping_patience)) {
          global_best <- best_losses$min()$item()
          if (global_best < es_global_best * (1 - early_stopping_tol)) {
            es_global_best     <- global_best
            es_iters_no_improv <- 0L
          } else {
            es_iters_no_improv <- es_iters_no_improv + track_interval
            if (es_iters_no_improv >= early_stopping_patience) {
              emit_fn(sprintf(
                "Early stopping at iter %d (no improvement >= %.2g for %d iters).",
                iter, early_stopping_tol, early_stopping_patience
              ))
              break
            }
          }
        }

        if (!is.null(monitor_fn)) {
          monitor_fn(list(
            iter                 = iter,
            cur_loss             = as.numeric(fwd$losses),
            cur_loss_unreg       = as.numeric(fwd$base_losses),
            cur_max_exp          = as.numeric(fwd$max_exp_llr),
            best_loss            = as.numeric(best_losses),
            best_loss_unreg      = as.numeric(best_base_losses),
            best_max_exp         = as.numeric(best_max_exp),
            cur_weights          = as.array(weights),
            best_weights         = as.array(best_weights),
            best_weights_max_exp = as.array(best_weights_max_exp),
            exp_llr              = t(as.array(fwd$exp_llr)),
            tracked              = if (is.null(tracked)) NA_real_ else as.numeric(tracked)
          ))
        }
      }

      if (iter %% report_interval == 0) {
        reg_now <- (fwd$losses - fwd$base_losses)$min()$item()
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
    weights                  = as.array(best_weights),
    final_loss               = as.array(best_losses),
    weights_max_exp          = as.array(best_weights_max_exp),
    final_max_exp            = as.array(best_max_exp),
    loss_history             = as.array(losses_history),
    tracked_history          = as.array(tracked_history),
    expectation_profile      = as.array(eval_weights(best_weights$log())$exp_llr$t()),
    expectation_profile_max_exp = as.array(eval_weights(best_weights_max_exp$log())$exp_llr$t())
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
#' @param init_weights Optional numeric matrix or tensor of shape
#'   `(n_restarts, length(ws))` to use as the starting point instead of random
#'   initialisation. Accepts the `weights` or `weights_max_exp` tensor from a
#'   previous [run_ripr()] result directly.
#' @param lin_smooth Laplacian smoothness penalty for the weights. Default: 0 (no smoothing).
#' @param log_smooth Laplacian smoothness penalty for the log-weights. Default: 0 (no smoothing).
#' @param early_stopping_patience Stop early if no meaningful improvement is seen
#'   for this many consecutive iterations. `NULL` disables early stopping. Default: `NULL`.
#' @param early_stopping_tol Relative improvement threshold for early stopping.
#'   See [optimize_mixture_weights()]. Default: `1e-4`.
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
  init_weights = NULL,
  lin_smooth = 0,
  log_smooth = 0,
  early_stopping_patience = NULL,
  early_stopping_tol = 1e-4,
  emit_fn = message,
  monitor_fn = NULL
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
    init_weights = init_weights,
    lin_smooth = lin_smooth,
    log_smooth = log_smooth,
    early_stopping_patience = early_stopping_patience,
    early_stopping_tol = early_stopping_tol,
    emit_fn = emit_fn,
    monitor_fn = monitor_fn
  )
}
