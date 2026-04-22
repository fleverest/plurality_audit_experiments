box::use(
  torch[torch_tensor, torch_logsumexp],
  ripr / torch_settings[device, dtype],
  ripr / tensor_ops[matmul_0_ninf, add_ninf_any],
  ripr / multinomial[build_counts_tensor, mnom_logpmf],
  ripr / grids[make_simplex_grid]
)

log_sum_exp <- function(x) {
  m <- max(x[is.finite(x)])
  m + log(sum(exp(x - m)))
}

# Assemble a full-length log-weight vector from per-half log-weights.
# When n_components is even the midpoint (index floor(n/2) == ceiling(n/2)) is
# shared by both halves; its contributions are log-added.
assemble_boundary_log_weights <- function(
  n_components,
  log_w_left,
  log_w_right
) {
  left_idx <- seq_len(floor(n_components / 2))
  right_idx <- seq(ceiling(n_components / 2), n_components)

  full <- rep(-Inf, n_components)
  full[left_idx] <- log_w_left

  overlap <- intersect(left_idx, right_idx)
  unique_right <- setdiff(right_idx, overlap)

  if (length(overlap) > 0L) {
    overlap_pos_in_right <- which(right_idx %in% overlap)
    full[overlap] <- log(
      exp(full[overlap]) + exp(log_w_right[overlap_pos_in_right])
    )
    full[unique_right] <- log_w_right[-overlap_pos_in_right]
  } else {
    full[right_idx] <- log_w_right
  }

  full
}

#' Fit Beta distributions to the left and right halves of the boundary weights
#'
#' @param weights Matrix (restarts x components) of discrete mixture weights.
#' @return Matrix (restarts x 6): left_shape1, left_shape2, right_shape1,
#'   right_shape2, left_mass, right_mass.
#' @export
smooth_ripr_beta <- function(weights) {
  n_restarts <- nrow(weights)
  n_components <- ncol(weights)

  beta_params <- matrix(0, nrow = n_restarts, ncol = 6L)
  colnames(beta_params) <- c(
    "left_shape1",
    "left_shape2",
    "right_shape1",
    "right_shape2",
    "left_mass",
    "right_mass"
  )

  for (i in seq_len(n_restarts)) {
    left_w <- weights[i, seq_len(floor(n_components / 2))]
    right_w <- weights[i, seq(ceiling(n_components / 2), n_components)]

    left_pos <- seq(0, 1, length.out = length(left_w))
    right_pos <- seq(0, 1, length.out = length(right_w))

    left_w_norm <- left_w / sum(left_w)
    right_w_norm <- right_w / sum(right_w)

    left_mean <- sum(left_w_norm * left_pos)
    right_mean <- sum(right_w_norm * right_pos)
    left_var <- sum(left_w_norm * (left_pos - left_mean)^2)
    right_var <- sum(right_w_norm * (right_pos - right_mean)^2)

    concentration_left <- left_mean * (1 - left_mean) / left_var - 1
    concentration_right <- right_mean * (1 - right_mean) / right_var - 1

    beta_params[i, ] <- c(
      left_mean * concentration_left,
      (1 - left_mean) * concentration_left,
      right_mean * concentration_right,
      (1 - right_mean) * concentration_right,
      sum(left_w),
      sum(right_w)
    )
  }

  beta_params
}

#' Fit Normal distributions to the left and right halves of the boundary weights
#'
#' @param weights Matrix (restarts x components) of discrete mixture weights.
#' @return Matrix (restarts x 6): left_mean, left_sd, right_mean, right_sd,
#'   left_mass, right_mass.
#' @export
smooth_ripr_normal <- function(weights) {
  n_restarts <- nrow(weights)
  n_components <- ncol(weights)

  normal_params <- matrix(0, nrow = n_restarts, ncol = 6L)
  colnames(normal_params) <- c(
    "left_mean",
    "left_sd",
    "right_mean",
    "right_sd",
    "left_mass",
    "right_mass"
  )

  for (i in seq_len(n_restarts)) {
    left_w <- weights[i, seq_len(floor(n_components / 2))]
    right_w <- weights[i, seq(ceiling(n_components / 2), n_components)]

    left_pos <- seq(0, 1, length.out = length(left_w))
    right_pos <- seq(0, 1, length.out = length(right_w))

    left_w_norm <- left_w / sum(left_w)
    right_w_norm <- right_w / sum(right_w)

    left_mean <- sum(left_w_norm * left_pos)
    right_mean <- sum(right_w_norm * right_pos)

    normal_params[i, ] <- c(
      left_mean,
      sqrt(sum(left_w_norm * (left_pos - left_mean)^2)),
      right_mean,
      sqrt(sum(right_w_norm * (right_pos - right_mean)^2)),
      sum(left_w),
      sum(right_w)
    )
  }

  normal_params
}

#' Log-weights for a beta-approximated mixture over boundary grid points
#'
#' Evaluates dbeta at n_components equally-spaced positions in [0,1], separately
#' for the left and right halves, normalises each half in log-space, and scales
#' by left_mass / right_mass.
#'
#' @param n_components Integer — number of boundary components (ncol of the weights
#'   matrix passed to [smooth_ripr_beta()]).
#' @param beta_params Named numeric vector: left_shape1, left_shape2, right_shape1,
#'   right_shape2, left_mass, right_mass — typically a row or colMeans of the matrix
#'   returned by [smooth_ripr_beta()].
#' @return Numeric vector of length n_components: log mixture weights.
#' @export
beta_boundary_log_weights <- function(n_components, beta_params) {
  n_left <- floor(n_components / 2)
  n_right <- n_components - ceiling(n_components / 2) + 1L

  log_w_left <- dbeta(
    seq(0, 1, length.out = n_left),
    beta_params[["left_shape1"]],
    beta_params[["left_shape2"]],
    log = TRUE
  )
  log_w_right <- dbeta(
    seq(0, 1, length.out = n_right),
    beta_params[["right_shape1"]],
    beta_params[["right_shape2"]],
    log = TRUE
  )

  log_w_left <- log_w_left -
    log_sum_exp(log_w_left) +
    log(beta_params[["left_mass"]])
  log_w_right <- log_w_right -
    log_sum_exp(log_w_right) +
    log(beta_params[["right_mass"]])

  assemble_boundary_log_weights(n_components, log_w_left, log_w_right)
}


#' Log-weights for a normal-approximated mixture over boundary grid points
#'
#' @param n_components Integer — number of boundary components.
#' @param normal_params Named numeric vector: left_mean, left_sd, right_mean, right_sd,
#'   left_mass, right_mass — from [smooth_ripr_normal()].
#' @return Numeric vector of length n_components: log mixture weights.
#' @export
normal_boundary_log_weights <- function(n_components, normal_params) {
  n_left <- floor(n_components / 2)
  n_right <- n_components - ceiling(n_components / 2) + 1L

  log_w_left <- dnorm(
    seq(0, 1, length.out = n_left),
    normal_params[["left_mean"]],
    normal_params[["left_sd"]],
    log = TRUE
  )
  log_w_right <- dnorm(
    seq(0, 1, length.out = n_right),
    normal_params[["right_mean"]],
    normal_params[["right_sd"]],
    log = TRUE
  )

  log_w_left <- log_w_left -
    log_sum_exp(log_w_left) +
    log(normal_params[["left_mass"]])
  log_w_right <- log_w_right -
    log_sum_exp(log_w_right) +
    log(normal_params[["right_mass"]])

  assemble_boundary_log_weights(n_components, log_w_left, log_w_right)
}

#' Expected mixture likelihood ratio profile under a given set of log boundary weights
#'
#' Computes E_theta[Q(X) / P_w(X)] for each DGP theta in the grid, where P_w is
#' the mixture defined by log_boundary_weights over the component grid ws.
#'
#' @param n Total number of trials per observation.
#' @param log_boundary_weights Numeric vector of length C — log mixture weights over
#'   the boundary component grid, e.g. from [beta_boundary_log_weights()] or
#'   [normal_boundary_log_weights()].
#' @param thetas R list of numeric vectors — DGP grid (same format as [run_ripr()]).
#' @param q Numeric vector — numerator distribution Q.
#' @param ws R list of numeric vectors — boundary component grid (same as [run_ripr()]).
#' @return Numeric vector of length T: E_theta[Q(X)/P_w(X)] for each theta.
#' @export
ripr_expectation_profile <- function(n, log_boundary_weights, thetas, q, ws) {
  to_log_tensor <- function(prob_list) {
    do.call(rbind, prob_list) |>
      torch_tensor(device = device, dtype = dtype) |>
      (\(x) x$transpose(1, 2))() |>
      (\(x) x$log())()
  }

  log_theta <- to_log_tensor(thetas)
  log_q <- torch_tensor(q, device = device, dtype = dtype)$log()
  log_ws <- to_log_tensor(ws)

  X <- build_counts_tensor(n)
  M_ <- X$size(1)
  C_ <- log_ws$size(2)
  T_ <- log_theta$size(2)

  log_pmf <- mnom_logpmf(X, log_theta, n) # (M, T)
  llr_num <- matmul_0_ninf(X, log_q$unsqueeze(2))$squeeze(2) # (M,)
  log_comp_denoms <- matmul_0_ninf(X, log_ws) # (M, C)

  log_wts <- torch_tensor(
    log_boundary_weights,
    device = device,
    dtype = dtype
  ) # (C,)

  log_denom <- torch_logsumexp(
    add_ninf_any(
      log_comp_denoms,
      log_wts$unsqueeze(1)$expand(c(M_, C_))
    ),
    dim = 2
  ) # (M,)

  llr <- llr_num - log_denom # (M,)

  as.numeric(
    torch_logsumexp(
      add_ninf_any(log_pmf, llr$unsqueeze(2)$expand(c(M_, T_))),
      dim = 1
    )$exp()
  ) # (T,)
}

# The rest of this file is a work in progress for comparing fitted continuous distributions to the discrete weights, and visualizing the fit. This is not currently used in the main RIPR workflow, but may be useful for future development and diagnostics.
if (!is.null(weights)) {
  plot_weights_and_beta <- function(weights, log = FALSE) {
    beta_params <- smooth_ripr_beta(weights) |> apply(2L, mean) # average parameters across restarts for plotting
    mean_weights <- weights |>
      apply(2L, mean)
    nweights <- length(mean_weights)
    left_shape1 <- beta_params[1L]
    left_shape2 <- beta_params[2L]
    right_shape1 <- beta_params[3L]
    right_shape2 <- beta_params[4L]

    # Plot high resolution continuous beta distributions against binned discrete weights
    x <- seq(0, 1, length.out = 10000)
    left_beta_density <- dbeta(x, left_shape1, left_shape2)
    left_discrete_density <- mean_weights[
      1:floor(nweights / 2)
    ] /
      sum(mean_weights[1:floor(nweights / 2)])
    right_beta_density <- dbeta(x, right_shape1, right_shape2)
    right_discrete_density <- mean_weights[
      ceiling(nweights / 2):nweights
    ] /
      sum(mean_weights[ceiling(nweights / 2):nweights])

    ggplot() +
      geom_segment(
        # Vertical segments, scaled up to match the area under the Beta density for better visualization
        aes(
          x = seq(0, 1, length.out = length(left_discrete_density)),
          xend = seq(0, 1, length.out = length(left_discrete_density)),
          y = 0,
          yend = left_discrete_density *
            max(left_beta_density) /
            max(left_discrete_density)
        ),
        color = "red",
        size = 1
      ) +
      geom_line(
        aes(x = x, y = left_beta_density),
        color = "blue",
        size = 1
      ) +
      labs(
        title = "Left Side: Beta vs Discrete Weights",
        x = "Weight",
        y = "Density"
      ) +
      theme_minimal() +
      (if (log) {
        scale_y_continuous(trans = "log10", limits = c(1e-10, NA))
      } else {
        NULL
      })
  }

  plot_weights_and_beta(result_5$weights, log = TRUE)

  # Clamp weights and refit:
  clamped_weights <- result_5$weights |>
    apply(2L, function(w) {
      w[w < 1e-6] <- 0
      w
    })
  plot_weights_and_beta(clamped_weights, log = TRUE)

  # This is much harder, but let's compute the expectations of the likelihood ratios under the fitted Beta distribution and compare to the expectation under the discrete weights, to see how well the fitted distribution captures the key quantity of interest for RIPR.
  library(torch)
  box::use(
    ripr / torch_settings[device, dtype],
    ripr / tensor_ops[matmul_0_ninf, add_ninf_any],
    ripr / multinomial[build_counts_tensor, mnom_logpmf],
    ripr / grids[make_simplex_grid]
  )

  weights <- result_5$weights
  n <- 5

  resolution <- 10001L
  thetas <- make_simplex_grid(resolution)
  q <- c(7 / 16, 5 / 16, 4 / 16)
  ws <- make_simplex_grid(resolution)

  beta_params <- colMeans(smooth_ripr_beta(weights))
  exp_profile_beta <- ripr_expectation_profile(
    n = n,
    log_boundary_weights = beta_boundary_log_weights(resolution, beta_params),
    thetas = thetas,
    q = q,
    ws = ws
  )
  max(exp_profile_beta)
  which.max(exp_profile_beta)

  beta_params_clamped <- weights |>
    apply(2L, function(w) {
      w[w < 1e-9] <- 0
      w
    }) |>
    smooth_ripr_beta() |>
    colMeans()
  exp_profile_beta_clamped <- ripr_expectation_profile(
    n = n,
    log_boundary_weights = beta_boundary_log_weights(
      resolution,
      beta_params_clamped
    ),
    thetas = thetas,
    q = q,
    ws = ws
  )
  max(exp_profile_beta_clamped)
  which.max(exp_profile_beta_clamped)

  normal_params <- colMeans(smooth_ripr_normal(weights))
  exp_profile_normal <- ripr_expectation_profile(
    n = n,
    log_boundary_weights = normal_boundary_log_weights(
      resolution,
      normal_params
    ),
    thetas = thetas,
    q = q,
    ws = ws
  )
  max(exp_profile_normal)
  which.max(exp_profile_normal)

  ws <- make_simplex_grid(length(mean_weights))
  exp_profile_discrete <- ripr_expectation_profile(
    n = n,
    log_boundary_weights = log(weights[1, ]),
    thetas = thetas,
    q = q,
    ws = ws
  )
  max(exp_profile_discrete) # should be higher than both fitted distributions, since they are approximations to the discrete weights

  # Try clamping weights to zero to see if it improves the fit in the tails
  clamped_mean_weights <- mean_weights
  clamped_mean_weights[clamped_mean_weights < 1e-10] <- 0
  clamed_mean_weights <- clamped_mean_weights / sum(clamped_mean_weights) # renormalise after clamping
  exp_profile_clamped <- ripr_expectation_profile(
    n = n,
    log_boundary_weights = log(clamped_mean_weights),
    thetas = thetas,
    q = q,
    ws = ws
  )
  max(exp_profile_clamped)

  rank(c(
    max(exp_profile_discrete),
    max(exp_profile_clamped),
    max(exp_profile_beta),
    max(exp_profile_normal)
  ))
}

for (res in c(
  result_1,
  result_2,
  result_3,
  result_4,
  result_5,
  result_10,
  result_20,
  result_50,
  result_100
)) {
  beta_params <- colMeans(smooth_ripr_beta(res$weights))
  print(beta_params)
}

smooth_ripr_beta(weights) |> colMeans()
