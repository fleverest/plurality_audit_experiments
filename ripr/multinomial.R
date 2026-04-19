box::use(
  torch[
    torch_arange,
    torch_meshgrid,
    torch_stack,
    torch_cat,
    torch_lgamma,
    torch_tensor,
    torch_logsumexp
  ],
  ripr / torch_settings[device, dtype],
  ripr / tensor_ops[matmul_0_ninf, add_ninf_any]
)

#' All valid count vectors for a multinomial with m categories (GPU)
#'
#' Enumerates every integer vector (x_1, ..., x_m) with x_i >= 0 and
#' sum(x_i) = n using a GPU meshgrid, then filters to valid entries.
#'
#' @param n Total number of trials.
#' @param m Number of categories. Default: 3.
#' @return Tensor of shape (M, m) where M = choose(n + m - 1, m - 1).
#' @export
build_counts_tensor <- function(n, m = 3) {
  ar <- torch_arange(0, n, device = device, dtype = dtype)
  grids <- torch_meshgrid(rep(list(ar), m - 1), "ij")
  stacked <- torch_stack(grids, dim = m)

  sum_first_m1 <- stacked$sum(dim = m)
  mask <- sum_first_m1 <= n
  last_cat <- n - sum_first_m1

  spatial_size <- prod(stacked$shape[1:(m - 1)])
  first_cats_flat <- stacked$reshape(c(spatial_size, m - 1))
  mask_flat <- mask$reshape(spatial_size)
  last_cat_flat <- last_cat$reshape(spatial_size)

  first_cats_sel <- first_cats_flat$index_select(
    1,
    mask_flat$nonzero()$squeeze()
  )
  last_cat_sel <- last_cat_flat$masked_select(mask_flat)

  torch_cat(list(first_cats_sel, last_cat_sel$unsqueeze(2)), dim = 2)
}

#' All valid count vectors for a multinomial with m categories (CPU)
#'
#' Recursive CPU alternative to [build_counts_tensor()]; useful when the
#' meshgrid approach runs out of memory for large n or m.
#'
#' @param n Total number of trials.
#' @param m Number of categories. Default: 3.
#' @return Tensor of shape (M, m) placed on the global device.
#' @export
build_counts_tensor_cpu <- function(n, m = 3L) {
  M <- choose(n + m - 1, m - 1)
  X <- matrix(0L, nrow = M, ncol = m)

  fill_counts <- function(row_idx, remaining, pos, current) {
    if (pos == m) {
      X[row_idx, ] <<- c(current, remaining)
      return(row_idx + 1L)
    }
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
  # NOTE: despite the _cpu name, the tensor is placed on the global device
  torch_tensor(X, device = device, dtype = dtype)
}

#' Log PMF of a multinomial for M count vectors and C component distributions
#'
#' Computes log Multinomial(x; n, p_c) for every (x, p_c) pair via a single
#' matrix multiplication in log space.
#'
#' @param X Tensor of shape (M, m) — count vectors from [build_counts_tensor()].
#' @param log_comp_probs_t Tensor of shape (m, C) — log category probabilities
#'   for each of C component distributions.
#' @param n Total number of trials.
#' @return Tensor of shape (M, C) of log PMF values.
#' @export
mnom_logpmf <- function(X, log_comp_probs_t, n) {
  const <- torch_lgamma(torch_tensor(n + 1, device = device, dtype = dtype))
  lgamma_sum <- torch_lgamma(X + 1)$sum(dim = 2, keepdim = TRUE) # (M, 1)
  const - lgamma_sum + matmul_0_ninf(X, log_comp_probs_t) # (M, C)
}

#' Expected mixture likelihood ratio E_theta[Q(X) / P_w(X)]
#'
#' For every pair of DGP theta and weight vector w, computes the expected
#' likelihood ratio of the numerator Q against the mixture denominator P_w.
#' Used to evaluate candidate weight vectors without running an optimiser.
#'
#' @param n Total number of trials.
#' @param log_theta Tensor of shape (m, T) — log probabilities for T candidate
#'   data-generating processes.
#' @param log_q Tensor of shape (m,) — log probabilities of the numerator Q.
#' @param log_ws Tensor of shape (m, C) — log probabilities of C mixture
#'   component distributions.
#' @param log_wts Tensor of shape (C, W) — log mixture weights for W candidate
#'   weight vectors.
#' @return Tensor of shape (T, W) — expected LLR for each (theta, weights) pair.
#' @export
mnom_exp_llr <- function(n, log_theta, log_q, log_ws, log_wts) {
  X <- build_counts_tensor(n)
  M_ <- X$size(1)
  C_ <- log_ws$size(2)
  W_ <- log_wts$size(2)
  T_ <- log_theta$size(2)

  log_pmf <- mnom_logpmf(X, log_theta, n)
  llr_num <- matmul_0_ninf(X, log_q)$unsqueeze(2)$expand(c(M_, W_))
  llr_denom <- torch_logsumexp(
    matmul_0_ninf(X, log_ws)$unsqueeze(3)$expand(c(M_, C_, W_)) +
      log_wts$unsqueeze(1)$expand(c(M_, C_, W_)),
    dim = 2
  )
  llr <- llr_num - llr_denom

  log_exp_llr <- torch_logsumexp(
    add_ninf_any(
      log_pmf$unsqueeze(3)$expand(c(M_, T_, W_)),
      llr$unsqueeze(2)$expand(c(M_, T_, W_))
    ),
    dim = 1
  )
  log_exp_llr$exp()
}

if (FALSE) {
  # Sanity check: all expected LLRs should be <= 1 under the true weights
  n <- 10
  log_theta <- matrix(
    c(1 / 3, 1 / 3, 1 / 3, 1 / 2, 1 / 2, 0),
    nrow = 3,
    byrow = TRUE
  ) |>
    torch_tensor(device = device, dtype = dtype) |>
    torch_log()
  log_q <- torch_tensor(c(0.5, 0.25, 0.25), device = device, dtype = dtype) |>
    torch_log()
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
    torch_log()
  log_wts <- build_counts_tensor(5, 5) |>
    torch_div(5) |>
    torch_transpose(1, 2) |>
    torch_log()
  result <- mnom_exp_llr(n, log_theta, log_q, log_ws, log_wts)
  all_leq1 <- result$le(1)$all(dim = 1)
}
