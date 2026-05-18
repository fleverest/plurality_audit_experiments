box::use(
  torch[
    torch_tensor,
    torch_lgamma
  ],
  ripr / torch_settings[device, dtype],
  ripr / tensor_ops[matmul_0_ninf],
  ripr / multinomial[build_counts_tensor]
)

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

  log_base_gpu <- torch_lgamma(torch_tensor(
    n + 1L,
    device = device,
    dtype = dtype
  )) -
    torch_lgamma(X_mat_gpu + 1)$sum(dim = 2L)

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
