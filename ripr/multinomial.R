box::use(
  utils[combn],
  torch[
    torch_lgamma,
    torch_tensor
  ],
  ripr / torch_settings[device, dtype],
  ripr / tensor_ops[matmul_0_ninf]
)

#' Log multinomial coefficient log(n! / prod x_i!) per count vector
#' @param X Tensor of shape (M, m) of count vectors.
#' @param n Total number of trials.
#' @return Tensor of shape (M,).
#' @export
log_multinom_coef <- function(X, n) {
  torch_lgamma(torch_tensor(n + 1, device = device, dtype = dtype)) -
    torch_lgamma(X + 1)$sum(dim = 2L)
}


#' All valid count vectors for a multinomial with m categories
#'
#' Enumerates every integer vector (x_1, ..., x_m) with x_i >= 0 and
#' sum(x_i) = n via stars and bars: each count vector corresponds to a choice
#' of m - 1 bar positions among n + m - 1 slots, so the enumeration costs
#' O(M * m) — proportional to the output — rather than the (n+1)^(m-1) dense
#' grid it would take to mesh over the first m - 1 coordinates. Enumeration
#' runs on the host; only the final (M, m) tensor is moved to the device.
#'
#' @param n Total number of trials.
#' @param m Number of categories. Default: 3.
#' @return Tensor of shape (M, m) where M = choose(n + m - 1, m - 1).
#' @export
build_counts_tensor <- function(n, m = 3) {
  counts <- if (m == 1L) {
    matrix(as.integer(n), nrow = 1L)
  } else {
    bars <- combn(n + m - 1L, m - 1L) # (m - 1, M) increasing bar positions
    boundaries <- rbind(0L, bars, n + m)
    t(diff(boundaries) - 1L) # x_i = b_i - b_{i-1} - 1, with b_0 = 0, b_m = n + m
  }
  torch_tensor(counts, device = device, dtype = dtype)
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
  log_multinom_coef(X, n)$unsqueeze(2) + matmul_0_ninf(X, log_comp_probs_t) # (M, C)
}

#' Multinomial likelihood interface for RIPr
#'
#' Constructs a likelihood interface object for the multinomial distribution
#' with `n` draws over `K` categories. All heavy computation (log-PMF
#' evaluation, score function) runs on the torch device specified by
#' [ripr/torch_settings].
#'
#' The returned list is the likelihood interface expected by [run_ripr()]:
#' \describe{
#'   \item{`support_tensor`}{`(M, K)` GPU tensor of all outcome count vectors
#'     with positive probability under at least one boundary point.}
#'   \item{`M`}{Integer. Number of outcomes in the support.}
#'   \item{`log_pmf(theta)`}{Function. Takes a length-`K` R vector `theta` and
#'     returns an `(M,)` GPU tensor of `log p_theta(x)` for each outcome.}
#'   \item{`log_pmf_batch(theta_mat)`}{Function. Takes a `(K, N)` R matrix and
#'     returns an `(M, N)` GPU tensor of log-PMFs for `N` parameter vectors.}
#'   \item{`score(theta)`}{Function. Takes a length-`K` R vector `theta` and
#'     returns an `(M, K)` GPU tensor of `d log p_theta(x) / d theta_l =
#'     x_l / theta_l - n` for each outcome and coordinate.}
#'   \item{`n`}{Integer. The sample size passed to `make_multinomial_likelihood`.}
#'   \item{`K`}{Integer. The number of categories.}
#' }
#'
#' @param n Integer. Total number of draws (ballot count).
#' @param K Integer. Number of categories (candidates).
#' @return A named list implementing the likelihood interface for [run_ripr()].
#' @seealso [run_ripr()], [run_plurality_ripr()]
#' @export
make_multinomial_likelihood <- function(n, K) {
  X_tensor <- build_counts_tensor(n, K)
  X_mat_gpu <- X_tensor$to(device = device, dtype = dtype)
  M <- X_mat_gpu$size(1L)

  log_base_gpu <- log_multinom_coef(X_mat_gpu, n) # (M,)

  log_pmf <- function(theta) {
    lt <- torch_tensor(log(theta), device = device, dtype = dtype)
    matmul_0_ninf(X_mat_gpu, lt) + log_base_gpu
  }

  log_pmf_batch <- function(theta_mat) {
    lt <- torch_tensor(log(theta_mat), device = device, dtype = dtype)
    matmul_0_ninf(X_mat_gpu, lt) + log_base_gpu$unsqueeze(2L)
  }

  score <- function(theta) {
    theta_gpu <- torch_tensor(theta, device = device, dtype = dtype)
    X_mat_gpu / theta_gpu$unsqueeze(1L) - n
  }

  list(
    support_tensor = X_mat_gpu,
    M = M,
    log_pmf = log_pmf,
    log_pmf_batch = log_pmf_batch,
    score = score,
    n = n,
    K = K
  )
}
