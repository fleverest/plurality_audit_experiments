box::use(
  torch[
    torch_arange,
    torch_meshgrid,
    torch_stack,
    torch_cat,
    torch_lgamma,
    torch_tensor
  ],
  ripr / torch_settings[device, dtype],
  ripr / tensor_ops[matmul_0_ninf]
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
