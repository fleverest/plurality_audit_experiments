box::use(
  torch[torch_isneginf, torch_where, torch_zeros_like, torch_matmul, torch_amax],
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

#' In-place element-wise addition treating -Inf + x as -Inf
#'
#' Fills `out` with `A + B`, masking positions where `A` is -Inf back to -Inf.
#' `A` and `B` may be smaller tensors that broadcast into `out`; the mask is
#' computed from `A` before expansion, avoiding a large intermediate bool tensor.
#'
#' @param out Pre-allocated output tensor. Modified in-place.
#' @param A First tensor (unexpanded). -Inf positions are preserved.
#' @param B Second tensor (unexpanded).
#' @return `out`, invisibly.
#' @export
add_ninf_any_ <- function(out, A, B, A_ninf = NULL) {
  out$copy_(A)
  out$add_(B)
  out$masked_fill_(if (is.null(A_ninf)) torch_isneginf(A) else A_ninf, -Inf)
  invisible(out)
}

#' In-place numerically-stable logsumexp along `dim`
#'
#' Computes `logsumexp(buf, dim)` without allocating a full-sized temporary.
#' `buf` is modified in-place (contains shifted exp values after the call) and
#' must not be read again until overwritten. Where all inputs along `dim` are
#' -Inf the result is -Inf.
#'
#' @param buf Tensor to reduce. Modified in-place.
#' @param dim Integer dimension to reduce over.
#' @return Tensor of the reduced logsumexp values (one fewer dimension than `buf`).
#' @export
logsumexp_inplace_ <- function(buf, dim) {
  mx <- torch_amax(buf, dim = dim, keepdim = TRUE)
  safe_mx <- torch_where(torch_isneginf(mx), torch_zeros_like(mx), mx)
  buf$sub_(safe_mx)$exp_()
  buf$sum(dim = dim)$log()$add_(safe_mx$squeeze(dim))
}
