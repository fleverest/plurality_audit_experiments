library(torch)

device <- if (cuda_is_available()) torch_device("cuda") else torch_device("cpu")
dtype <- torch_float64()

#' Custom matmul with 0 * -inf := 0
#'
#' @param A First tensor (m x k)
#' @param B Second tensor (k x n)
#' @return Result but with 0 * -inf = 0
#' @export
matmul_0_ninf <- function(A, B) {
  B_ninf <- torch_isneginf(B)
  B_safe <- torch_where(B_ninf, torch_zeros_like(B), B)
  result <- torch_matmul(A, B_safe)

  # Determine which positions should be -inf
  A_nonzero_int <- A$ne(0)$to(dtype = dtype)
  B_ninf_int <- B_ninf$to(dtype = dtype)

  ninf_contributions <- torch_matmul(A_nonzero_int, B_ninf_int)
  should_be_ninf <- ninf_contributions > 0

  result$masked_fill_(should_be_ninf, -Inf)
  result
}

#' Custom addition with -inf + any := -inf
#'
#' @param A First tensor
#' @param B Second tensor
#' @return `A + B` but with -inf + any = -inf
#' @export
add_ninf_any <- function(A, B) {
  A_ninf <- torch_isneginf(A)
  result <- A + B
  result$masked_fill_(A_ninf, -Inf)
  result
}

#' Generate all valid count vectors for a multinomial distribution with m categories on GPU
#'
#' @param n Total number of trials
#' @param m Number of categories (default: 3)
#' @param device Torch device to use (default: global device)
#' @param dtype Torch data type to use (default: global dtype)
#' @return Tensor of shape (M, m) containing all valid count vectors
#' @export
build_counts_tensor <- function(n, m = 3) {
  # create ranges on device
  ar <- torch_arange(0, n, device = device, dtype = dtype) # (n+1,)

  # make meshgrid for first m-1 categories
  grids <- torch_meshgrid(rep(list(ar), m - 1), "ij") # list of (m-1) tensors

  # stack all grids into a single tensor upfront
  stacked_grids <- torch_stack(grids, dim = m) # (n+1, n+1, ..., n+1, m-1)

  # sum the first m-1 categories
  sum_first_m1 <- stacked_grids$sum(dim = m) # (n+1, n+1, ..., n+1)
  mask <- sum_first_m1 <= n

  # compute last category
  last_cat <- n - sum_first_m1 # (n+1, n+1, ..., n+1)

  # reshape stacked_grids to flatten spatial dimensions
  spatial_size <- prod(stacked_grids$shape[1:(m - 1)])
  first_cats_flat <- stacked_grids$reshape(c(spatial_size, m - 1)) # ((n+1)^(m-1), m-1)

  # flatten mask and last_cat
  mask_flat <- mask$reshape(spatial_size)
  last_cat_flat <- last_cat$reshape(spatial_size)

  # select valid entries using boolean indexing (all tensors)
  first_cats_sel <- first_cats_flat$index_select(
    1,
    mask_flat$nonzero()$squeeze()
  )
  last_cat_sel <- last_cat_flat$masked_select(mask_flat)

  torch_cat(list(first_cats_sel, last_cat_sel$unsqueeze(2)), dim = 2) # (M, m)
}

#' Generate all valid count vectors for a multinomial distribution with m categories on CPU
#'
#' @param n Total number of trials
#' @param m Number of categories (default: 3)
#' @return Tensor of shape (M, m) containing all valid count vectors
#' @export
build_counts_tensor_cpu <- function(n, m = 3L) {
  M <- choose(n + m - 1, m - 1)
  X <- matrix(0L, nrow = M, ncol = m)

  # Recursive helper function to generate all combinations
  fill_counts <- function(row_idx, remaining, pos, current) {
    # Base case: we're at the last category
    if (pos == m) {
      X[row_idx, ] <<- c(current, remaining)
      return(row_idx + 1L)
    }
    # Recursive case: try all possible values for current position
    for (val in 0:remaining) {
      row_idx <- fill_counts(
        row_idx,
        remaining - val,
        pos + 1L,
        c(current, val)
      )
    }
    row_idx
  }
  fill_counts(1L, n, 1L, integer(0))
  torch::torch_tensor(X, device = device, dtype = dtype)
}


#' Compute log PMF of multinomial for multiple count vectors and multiple component probabilities
#'
#' @param X Tensor of shape (M, 3) containing M count vectors
#' @param log_comp_probs_t Tensor of shape (3, C) containing log probabilities of C components
#' @param n Total number of trials
#' @return Tensor of shape (M, C) containing log PMF values
#' @export
mnom_logpmf <- function(X, log_comp_probs_t, n) {
  const <- torch_lgamma(torch::torch_tensor(
    n + 1,
    device = device,
    dtype = dtype
  ))
  lgamma_sum <- torch_lgamma(X + 1)$sum(dim = 2, keepdim = TRUE) # (M,1)
  const - lgamma_sum + matmul_0_ninf(X, log_comp_probs_t) # (M,C)
}

#' Compute expected mixture likelihood ratio from log probabilities, E_theta[Q(X)/P_w(X)]
#'
#' @param n Total number of trials
#' @param log_theta Tensor of shape (3, T) containing log probabilities for theta (DGP probs)
#' @param log_q Tensor of shape (3,) containing log probabilities for q (numerator probs)
#' @param log_ws Tensor of shape (3, C) containing log probabilities for the mixture components (components of denominator probs)
#' @param log_wts Tensor of shape (C, W) containing the log-weights for the mixture components
#' @return Tensor of shape (T, B) containing expected values for each (theta, weights) pair
#' @export
mnom_exp_llr <- function(n, log_theta, log_q, log_ws, log_wts) {
  X <- build_counts_tensor(n) # (M,3)

  # Dims
  M_ <- X$size(1)
  C_ <- log_ws$size(2)
  W_ <- log_wts$size(2)
  T_ <- log_theta$size(2)

  # Compute log probabilities under theta
  log_pmf <- mnom_logpmf(X, log_theta, n) # (M, T)

  # Compute llr for each count vector (M, C)
  llr_num <- matmul_0_ninf(X, log_q)$unsqueeze(2)$expand(c(M_, W_)) # (M,W)
  llr_denom <- torch_logsumexp(
    matmul_0_ninf(X, log_ws)$unsqueeze(3)$expand(c(M_, C_, W_)) + # (M,C,W)
      log_wts$unsqueeze(1)$expand(c(M_, C_, W_)), # (M,C,W)
    dim = 2
  ) # (M, W)
  llr <- llr_num - llr_denom # (M, W)

  # Compute expected llr under theta
  log_exp_llr <- torch_logsumexp(
    add_ninf_any(
      log_pmf$unsqueeze(3)$expand(c(M_, T_, W_)),
      llr$unsqueeze(2)$expand(c(M_, T_, W_))
    ), # (M,T,W)
    dim = 1
  ) # (T, W)
  log_exp_llr$exp() # (T, W)
}

# Example usage:
n <- 10
# Two DGPs: theta1=(1/3, 1/3, 1/3), theta2=(1/2,1/2,0)
log_theta <- matrix(
  c(1 / 3, 1 / 3, 1 / 3, 1 / 2, 1 / 2, 0),
  nrow = 3,
  byrow = TRUE
) |>
  torch_tensor(device = device, dtype = dtype) |>
  torch_log() # (3,2)
# Numerator q = (0.5, 0.25, 0.25)
log_q <- torch_tensor(c(0.5, 0.25, 0.25), device = device, dtype = dtype) |>
  torch_log() # (3,)
# Five mixture components w1=(1/2,1/2,0), w2=(1/3,1/3,1/3), w3=(1/2,0,1/2), w4=(3/8, 3/8, 1/4), w5=(3/8, 1/4, 3/8)
log_ws <- matrix(
  c(
    1 / 2,
    1 / 2,
    0,
    1 / 3,
    1 / 3,
    1 / 3,
    1 / 2,
    0,
    1 / 2,
    3 / 8,
    3 / 8,
    1 / 4,
    3 / 8,
    1 / 4,
    3 / 8
  ),
  nrow = 5,
  byrow = TRUE
) |>
  t() |>
  torch_tensor(device = device, dtype = dtype) |>
  torch_log() # (3,5)
# choose(5+5-1, 5-1) different mixture weights
log_wts <- build_counts_tensor(5, 5) |>
  torch_div(5) |>
  torch_transpose(1, 2) |>
  torch_log() # (5,126)

result <- mnom_exp_llr(n, log_theta, log_q, log_ws, log_wts)
all_leq1 <- result$le(1)$all(dim = 1)


# ==============================================================================
# SIMULATIONS
# =============================================================================

# build the theta grid used in the script (parameterised number of points)
make_thetas <- function(points = 21) {
  if (!is.numeric(points) || length(points) != 1 || points < 3L) {
    stop("`points` must be a single numeric >= 3")
  }
  s1_len <- ceiling((points + 1) / 2)
  s2_len <- floor((points + 1) / 2)
  seq1 <- seq(1 / 2, 1 / 3, length.out = s1_len) # inclusive endpoints
  seq2 <- seq(1 / 3, 1 / 2, length.out = s2_len) # inclusive endpoints; we'll drop the first element
  seq2_tail <- if (length(seq2) > 1L) tail(seq2, -1) else numeric(0)

  c(
    lapply(seq1, function(theta1) c(theta1, theta1, 1 - 2 * theta1)),
    lapply(seq2_tail, function(theta1) c(theta1, 1 - 2 * theta1, theta1))
  )
}

#############################################################################
# Parallel gradient descent

rdirichlet <- function(n, alpha) {
  k <- length(alpha)
  samples <- matrix(0, nrow = n, ncol = k)
  for (i in seq_len(k)) {
    samples[, i] <- rgamma(n, shape = alpha[i], rate = 1)
  }
  samples <- samples / rowSums(samples)
  samples
}

optim_frank_wolfe <- optimizer(
  initialize = function(params, step = 2.0, alpha = 1.0, ...) {
    # step_size at iteration t: c / (t + c)
    defaults <- list(
      step = step,
      alpha = alpha
    )
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

          # Frank-Wolfe step: find vertex that minimizes <grad, s>
          # For simplex, this is a one-hot at argmin of gradient
          if (length(param$shape) == 1) {
            # Single simplex: (C,)
            best_idx <- grad$argmin()
            s <- torch_zeros_like(param)
            s[best_idx] <- 1.0
          } else {
            # Multiple simplexes: (R, C) - find argmin per row
            best_idx <- grad$argmin(dim = 2, keepdim = TRUE) # (R, 1)
            s <- torch_zeros_like(param)
            s$scatter_(2, best_idx, 1.0) # one-hot per row
          }

          # Adaptive step size: [c / (t + c)] ^ alpha
          gamma <- (group$step / (private$iteration + group$step))^group$alpha

          # Convex combination: w = (1-gamma)*w + gamma*s
          param$mul_(1 - gamma)
          param$add_(s, alpha = gamma)
        }
      }
    })

    loss
  },

  private = list(
    iteration = NULL
  )
)

optim_projected_gd <- optimizer(
  initialize = function(params, lr = 0.01, momentum = 0.0, eps = 1e-8, ...) {
    defaults <- list(
      lr = lr,
      momentum = momentum,
      eps = eps
    )
    super$initialize(params, defaults)

    # Initialize momentum buffers if needed
    if (momentum > 0) {
      private$momentum_buffer <- NULL
    }
  },

  step = function(closure = NULL) {
    param <- self$param_groups[[1]]$params[[1]]
    grad <- param$grad

    lr <- self$param_groups[[1]]$lr
    eps <- self$param_groups[[1]]$eps
    momentum <- self$param_groups[[1]]$momentum

    # Project gradient onto tangent space of simplex
    # Tangent space: {d : sum(d) = 0, d_i >= 0 if w_i = 0}

    # Multiple simplexes: (R, C)
    # 1. Remove component perpendicular to simplex (row-wise mean)
    grad_mean <- grad$mean(dim = 2, keepdim = TRUE)
    grad_projected <- grad - grad_mean

    # 2. Zero out gradients for components at boundary with positive gradient
    at_boundary <- param <= 0
    grad_projected <- torch_where(
      at_boundary & (grad_projected > 0),
      torch_zeros_like(grad_projected),
      grad_projected
    )

    # Apply momentum to projected gradient if enabled
    if (momentum > 0) {
      if (is.null(private$momentum_buffer)) {
        buf <- torch_zeros_like(grad_projected)
        private$momentum_buffer <- buf
      } else {
        buf <- private$momentum_buffer
      }
      buf$mul_(momentum)$add_(grad_projected)
      grad_projected <- buf
    }

    # Take gradient step with projected gradient, then project back to simplex
    param$add_(grad_projected, alpha = -lr)$clamp_(min = eps)
    param$div_(param$sum(dim = 2, keepdim = TRUE))
  },

  private = list(
    momentum_buffer = NULL
  )
)

lr_step <- function(op, step_size, gamma, ...) {
  torch::lr_step(op, step_size = step_size, gamma = gamma)
}

#' Find optimal weights with parallel random restarts
#'
#' @param n Total number of trials
#' @param log_theta Tensor of shape (3, T) - DGP probabilities
#' @param log_q Tensor of shape (3,) - numerator probabilities
#' @param log_ws Tensor of shape (3, C) - mixture component probabilities
#' @param n_restarts Number of parallel random restarts
#' @param optim_fn Optimizer function (e.g., optim_adam, optim_frank_wolfe)
#' @param tol Convergence tolerance
#' @param batch_size Batch size for logging
#' @param n_batches Number of batches for logging
#' @param eps Minimum weight to avoid numerical issues. Defaults to 0.0.
#' @param ... Additional arguments for optimizer
#' @return List with best result across all restarts
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
  ...
) {
  # Pre-compute things that don't depend on weights
  X <- build_counts_tensor(n) # (M, 3)
  M_ <- X$size(1)

  C_ <- log_ws$size(2)
  T_ <- log_theta$size(2)

  log_pmf <- mnom_logpmf(X, log_theta, n) # (M, T)
  llr_num <- matmul_0_ninf(X, log_q$unsqueeze(2))$squeeze(2) # (M,)
  log_comp_denoms <- matmul_0_ninf(X, log_ws) # (M, C)

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
  current_losses <- losses_history[1, ] # initialize

  # Initialize R parallel random starts: (R, C)
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

        # Compute LLR denominator for all restarts in parallel
        # log_comp_denoms: (M, C)
        # log_wts: (R, C)
        # Want: (M, R) - logsumexp over C dimension

        # Expand: (1, R, C) + (M, 1, C) -> (M, R, C)
        log_denoms_expanded <- add_ninf_any(
          log_comp_denoms$unsqueeze(2)$expand(c(M_, n_restarts, C_)),
          log_wts$unsqueeze(1)$expand(c(M_, n_restarts, C_))
        )

        llr_denom <- torch_logsumexp(log_denoms_expanded, dim = 3) # (M, R)

        # LLR for each count vector and each restart: (M, R)
        llr <- llr_num$unsqueeze(2)$expand(c(M_, n_restarts)) - llr_denom

        # Expected LLR under each theta for each restart
        # log_pmf: (M, T)
        # llr: (M, R)
        # Want: (T, R) for each restart

        # Expand: (M, T, 1) + (M, 1, R) -> (M, T, R)
        log_pmf_llr <- add_ninf_any(
          log_pmf$unsqueeze(3)$expand(c(M_, T_, n_restarts)),
          llr$unsqueeze(2)$expand(c(M_, T_, n_restarts))
        )

        log_exp_llr <- torch_logsumexp(log_pmf_llr, dim = 1) # (T, R)
        exp_llr <- log_exp_llr$exp() # (T, R)

        # Loss for each restart: max over thetas
        max_exp_llr <- exp_llr$max(dim = 1) # (R,)
        argmax_exp_llr <- max_exp_llr[[2]]
        max_exp_llr <- max_exp_llr[[1]]

        # Track history and best
        prev_losses <- current_losses
        current_losses <- max_exp_llr
        improved <- current_losses < best_losses
        best_losses <- torch_where(improved, current_losses, best_losses)
        best_weights[improved, ] <- weights[improved, ]
        if (iter %% track_freq == 0) {
          losses_history[iter / track_freq, ] <- current_losses
        }

        ## Compute gradients
        worst_log_pmf <- log_pmf$index_select(2, argmax_exp_llr) # (M, R)
        log_grad_terms <- (
          worst_log_pmf$unsqueeze(3) + # (M, R, 1)
            llr$unsqueeze(3) + # (M, R, 1)
            log_comp_denoms$unsqueeze(2) - # (M, 1, C)
            llr_denom$unsqueeze(3) # (M, R, 1)
        ) # (M, R, C)

        grad_w <- -log_grad_terms$logsumexp(dim = 1)$exp() # (R, C)
        if (use_softmax) {
          # Chain rule through softmax
          grad_u <- weights *
            (grad_w - (grad_w * weights)$sum(dim = 2, keepdim = TRUE))
        }
      })

      # Update weights
      if (use_softmax) {
        unconstrained_weights$grad <- grad_u
        optimizer$step()
      } else {
        weights$grad <- grad_w
        optimizer$step()
      }
    }

    scheduler$step()

    # Track best for each restart
    cat(sprintf(
      "Batch %d: best=%.8f, worst=%.8f, mean=%.8f, lr=%.8f\n",
      batch_idx,
      best_losses$min(),
      best_losses$max(),
      best_losses$mean(),
      optimizer$param_groups[[1]]$lr
    ))
    # Check convergence (all restarts)
    if (batch_idx > 1) {
      max_change <- (prev_losses - current_losses)$abs()$max()$item()
      if (max_change < tol) {
        cat(sprintf("Converged at batch %d\n", batch_idx))
        break
      }
    }
  }

  list(
    best_weights = best_weights,
    best_losses = best_losses,
    losses_history = losses_history
  )
}

run_simulation <- function(
  n,
  n_thetas,
  q,
  n_ws,
  n_restarts,
  optim_fn,
  batch_size,
  n_batches,
  tol,
  ...
) {
  log_theta <- make_thetas(n_thetas) |>
    do.call(what = rbind) |>
    torch_tensor(device = device, dtype = dtype) |>
    torch_transpose(1, 2) |>
    torch_log()
  log_q <- q |>
    torch_tensor(device = device, dtype = dtype) |>
    torch_log()
  log_ws <- make_thetas(n_ws) |>
    do.call(what = rbind) |>
    torch_tensor(device = device, dtype = dtype) |>
    torch_transpose(1, 2) |>
    torch_log()

  optimize_mixture_weights(
    n,
    log_theta,
    log_q,
    log_ws,
    n_restarts = n_restarts,
    optim_fn,
    batch_size = batch_size,
    n_batches = n_batches,
    tol = tol,
    ...
  )
}

set.seed(20251111)
torch_manual_seed(20251111)
n_thetas <- 1001
q <- c(7 / 16, 5 / 16, 4 / 16)
n_restarts <- 50
lr <- 1.0
optim_fn <- function(params, lr = 0.01, ...) {
  torch::optim_adam(params = params, lr = lr)
}
tol <- 1e-8


small_ns <- 1:5
for (n in small_ns) {
  res <- run_simulation(
    n = n,
    n_thetas = n_thetas,
    q = q,
    n_ws = n_thetas,
    n_restarts = 500,
    optim_fn = optim_fn,
    use_softmax = TRUE,
    lr = lr,
    batch_size = 1000,
    n_batches = 250,
    tol = tol,
    gamma = 0.95
  )
  out <- list(
    best_weights = res$best_weights |> as.array(),
    best_losses = res$best_losses |> as.array(),
    losses_history = res$losses_history |> as.array()
  )
  rm(res)
  gc()
  if (cuda_is_available()) {
    cuda_empty_cache()
  }
  saveRDS(out, file = sprintf("results_n%d.rds", n))
  rm(out)
}

big_ns <- c(10, 20, 50, 100)
for (n in big_ns) {
  res <- run_simulation(
    n = n,
    n_thetas = n_thetas,
    q = q,
    n_ws = n_thetas,
    n_restarts = 100,
    optim_fn = optim_fn,
    use_softmax = TRUE,
    lr = lr,
    batch_size = 1000,
    n_batches = 250,
    tol = tol,
    gamma = 0.95
  )
  out <- list(
    best_weights = res$best_weights |> as.array(),
    best_losses = res$best_losses |> as.array(),
    losses_history = res$losses_history |> as.array()
  )
  rm(res)
  gc()
  if (cuda_is_available()) {
    cuda_empty_cache()
  }
  saveRDS(out, file = sprintf("results_n%d.rds", n))
  rm(out)
}

# ==============================================================================
# PLOTTING FUNCTIONS
# ==============================================================================

library(ggplot2)
library(reshape2)
library(dplyr)

plot_results_weights <- function(results_weights, results_losses, top = Inf) {
  ##########################################################
  # Visualise the different resulting weights as a heatmap #
  weights_df <- results_weights |>
    as.matrix() |>
    as.data.frame()
  colnames(weights_df) <- as.character(seq_len(ncol(weights_df)))

  weights_df$restart <- seq_len(nrow(weights_df))

  weights_melt <- melt(
    weights_df,
    variable.name = "Component",
    value.name = "Weight",
    id.vars = "restart"
  )
  loss_order <- order(results_losses)
  weights_melt |> # Filter to top `top` restarts by loss
    filter(restart %in% (loss_order |> head(top))) |>
    mutate(restart = factor(restart, levels = loss_order)) |>
    ggplot(aes(x = Component, y = restart, fill = Weight)) +
    geom_tile() +
    scale_fill_viridis_c() +
    labs(
      title = "Optimized Mixture Weights Across Random Restarts (Ordered by Loss)",
      x = "Mixture Component",
      y = "Loss (max expectation) achieved"
    ) +
    theme_minimal() +
    scale_y_discrete(
      limits = rev(loss_order) |> tail(top) |> factor(),
      labels = function(x) results_losses[as.numeric(as.character(x))]
    )
}

plot_results_loss_history <- function(results_loss_history) {
  # Plot of loss histories for each restart
  losses_history_df <- as.data.frame(results_loss_history)
  colnames(losses_history_df) <- paste0(
    "Restart",
    seq_len(ncol(losses_history_df))
  )
  losses_history_df$Iteration <- seq_len(nrow(losses_history_df))
  losses_melt <- melt(
    losses_history_df,
    id.vars = "Iteration",
    variable.name = "Restart",
    value.name = "Loss"
  )
  ggplot(losses_melt, aes(x = Iteration, y = Loss, color = Restart)) +
    geom_line(alpha = 0.7) +
    labs(
      title = "Loss Histories for Each Random Restart",
      x = "Iteration",
      y = "Max Expected LLR"
    ) +
    theme_minimal() +
    theme(legend.position = "none")
}


files <- list.files(pattern = "*.rds$")

for (f in files) {
  res <- readRDS(f)
  p1 <- plot_results_weights(res$best_weights, res$best_losses, top = 50)
  ggsave(
    filename = paste0("weights_", f, ".png"),
    plot = p1,
    width = 8,
    height = 6
  )

  p2 <- plot_results_loss_history(log(1 - res$losses_history))
  ggsave(
    filename = paste0("loss_history_", f, ".png"),
    plot = p2,
    width = 8,
    height = 6
  )

  rm(res)
  gc()
}

# Analyse the resulting weights for different n
thetas <- make_thetas(n_thetas)
for (f in files) {
  res <- readRDS(f)
  cat(sprintf("Results for %s:\n", f))
  cat(sprintf("Best loss: %.8f\n", min(res$best_losses)))
  left_theta <- thetas[[which.max(res$best_losses)]]
  cat(sprintf("Theta with largest weight: %.8f\n", thetas[[which.max()]]))
  cat(sprintf("Best weights (top 10):\n"))
  print(res$best_weights[which.min(res$best_losses), ] |> order() |> tail(10))
  cat("\n")
}
res <- readRDS("results_n50.rds")
res$best_weights[which.min(res$best_losses), ] |> order() |> tail(10)
res$losses_history |> dim()
plot_results_loss_history(res$losses_history)
