box::use(
  torch[torch_isneginf, torch_where, torch_zeros_like, torch_matmul],
  ripr / torch_settings[dtype]
)

#' Matrix multiplication treating 0 × -Inf as 0
#'
#' Standard `torch_matmul` propagates -Inf into any row that multiplies it,
#' even with a zero weight. This variant zeroes those contributions instead,
#' which is the correct behaviour when the zero weight means "exclude this
#' component" in log-space mixture computations.
#'
#' @param A Tensor of shape (m, k).
#' @param B Tensor of shape (k, n), may contain -Inf entries.
#' @return Tensor of shape (m, n) with 0 × -Inf = 0.
#' @export
matmul_0_ninf <- function(A, B) {
  B_ninf <- torch_isneginf(B)
  B_safe <- torch_where(B_ninf, torch_zeros_like(B), B)
  result <- torch_matmul(A, B_safe)

  A_nonzero_int <- A$ne(0)$to(dtype = dtype)
  B_ninf_int <- B_ninf$to(dtype = dtype)
  ninf_contributions <- torch_matmul(A_nonzero_int, B_ninf_int)

  result$masked_fill_(ninf_contributions > 0, -Inf)
  result
}

#' Element-wise addition treating -Inf + x as -Inf
#'
#' Prevents log-space underflow from being silently cancelled when adding a
#' finite correction to a -Inf log-probability.
#'
#' @param A First tensor. Positions where `A` is -Inf remain -Inf in output.
#' @param B Second tensor.
#' @return Tensor equal to `A + B` except where `A` is -Inf.
#' @export
add_ninf_any <- function(A, B) {
  A_ninf <- torch_isneginf(A)
  result <- A + B
  result$masked_fill_(A_ninf, -Inf)
  result
}
