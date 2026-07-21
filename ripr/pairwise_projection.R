box::use(
  stats[pbeta],
  S7[S7_inherits],
  ripr /
    mixture[
      discrete_simplex_mixture,
      truncated_dirichlet,
      log_pmf,
      n_categories
    ],
  ripr / multinomial[make_multinomial_likelihood]
)

# Pairwise-null baseline e-variables for plurality audits.
#
# The `pairwise_projection` class in ripr/mixture.R gives the analytic RIPr
# P*_j of a numerator Q onto each pairwise null H_0^(j) = {theta_1 <= theta_j}.
# This module provides the resulting baseline e-variables and their exact
# log-growth: the per-pair e-variable Q / P*_j, the relaxed 1-D
# truncated-Beta variant, and the combined min-over-j e-variable, which is
# valid for the union null H_0 = union_j H_0^(j).

as_count_matrix <- function(X) {
  if (inherits(X, "torch_tensor")) {
    return(as.matrix(as.array(X$cpu())))
  }
  if (is.null(dim(X))) matrix(X, nrow = 1L) else as.matrix(X)
}

logsumexp <- function(v) {
  m <- max(v)
  if (!is.finite(m)) {
    return(m)
  }
  m + log(sum(exp(v - m)))
}

#' log P*_j over the full multinomial support, reusing a precomputed log_q
#'
#' Every (m, N - m) split of each support row is itself a support row, so the
#' marginal over the (1, j)-split is a grouped logsumexp of `log_q` — Q's pmf
#' need not be re-evaluated per pair (it costs one numerical integral per row
#' for truncated_dirichlet numerators). Equivalent to
#' `log_pmf(pairwise_projection(Q, j, n), support)` when `X_cpu` is the full
#' support and `log_q = log_pmf(Q, X_cpu)`.
#'
#' @param log_q Numeric vector: `log_pmf(Q, X_cpu)`.
#' @param X_cpu Full-support count matrix of shape (M, K).
#' @param j Integer in `2:K`. The competing candidate defining the null.
#' @return Numeric vector of length M with `log P*_j(x)` per row.
#' @export
pairwise_log_pstar <- function(log_q, X_cpu, j) {
  x1 <- X_cpu[, 1L]
  N <- x1 + X_cpu[, j]
  merged <- X_cpu
  merged[, 1L] <- N
  merged[, j] <- 0
  key <- apply(merged, 1L, paste, collapse = "_")
  grp <- match(key, key)
  f <- match(grp, unique(grp)) # dense ids 1..G, first-occurrence order
  log_marg_by_rep <- vapply(
    split(log_q_stacked, factor(stacked_grp, levels = rep_idx)),
    logsumexp_num,
    numeric(1L)
  )
  log_marg <- log_marg_by_rep[match(grp, rep_idx)]
  log_marg + lchoose(N, x1) - N * log(2)
}

#' Log e-values of the relaxed 1-D pairwise test (truncated-Beta prior)
#'
#' For a Dirichlet(alpha) prior truncated to the pairwise alternative
#' `{theta_1 > theta_j}` only (rather than the full plurality wedge H_1),
#' Dirichlet neutrality makes the split `p = theta_1 / (theta_1 + theta_j)`
#' a `Beta(alpha_1, alpha_j)` truncated to `(1/2, 1)`, independent of the
#' merged coordinates. The e-variable against `H_0^{(j)}` then collapses to
#' one dimension:
#'
#'   E_j(x) = 2^N B(a_1 + x_1, a_j + N - x_1) / B(a_1, a_j)
#'            * P(Beta(a_1 + x_1, a_j + N - x_1) > 1/2) / P(Beta(a_1, a_j) > 1/2)
#'
#' with `N = x_1 + x_j`. This coincides with the same-prior projection
#' e-value `Q / P*_j` when Q is a point mass (or K = 2), but differs for
#' K >= 3 priors truncated to the full wedge H_1.
#'
#' @param X Count matrix / tensor of shape (M, K).
#' @param alpha Numeric vector of Dirichlet concentration parameters.
#' @param j Integer in `2:K`. The competing candidate defining the null.
#' @return Numeric vector of length M with `log E_j(x)` per row.
#' @export
relaxed_pairwise_log_e <- function(X, alpha, j) {
  X_cpu <- as_count_matrix(X)
  x1 <- X_cpu[, 1L]
  N <- x1 + X_cpu[, j]
  a1 <- alpha[1L]
  aj <- alpha[j]
  N *
    log(2) +
    lbeta(a1 + x1, aj + N - x1) -
    lbeta(a1, aj) +
    pbeta(0.5, a1 + x1, aj + N - x1, lower.tail = FALSE, log.p = TRUE) -
    pbeta(0.5, a1, aj, lower.tail = FALSE, log.p = TRUE)
}

#' Exact log-growth of the pairwise-null baseline e-variables
#'
#' Computes, by exact summation over the full multinomial support, the
#' log-growth `E_Q[log E]` of the pairwise baseline e-variables for the
#' plurality null `H_0 = union_j {theta_1 <= theta_j}`:
#'
#'   * `pairwise_ripr`: `E = min_j Q / P*_j` with `P*_j` the same-prior
#'     pairwise RIPr ([pairwise_projection()]). The pointwise minimum of
#'     per-pair e-variables is a valid e-variable for the union null.
#'   * `pairwise_beta`: the same combination using the relaxed 1-D
#'     truncated-Beta e-values ([relaxed_pairwise_log_e()]). Computed when Q
#'     is a `truncated_dirichlet` (from its alpha) or a single point mass
#'     (where it coincides with `pairwise_ripr`); `NA` otherwise.
#'
#' Both growth rates are expectations under the same Q used by the GRO
#' method, so they are directly comparable to `E_Q[log Q / P_W]` from the
#' full-null RIPr optimiser.
#'
#' @param Q A `simplex_mixture` numerator (mixing measure supported on H_1).
#' @param n Integer multinomial sample size.
#' @return data.frame with columns `variant` (`"pairwise_ripr"` /
#'   `"pairwise_beta"`), `pair` (competing candidate `j`, or `NA` for the
#'   combined min-over-j e-variable), and `growth` (`E_Q[log E]`).
#' @export
pairwise_baseline_growth <- function(Q, n) {
  K <- n_categories(Q)
  likelihood <- make_multinomial_likelihood(n, K)
  support <- likelihood$support_tensor
  X_cpu <- as_count_matrix(support)
  log_q <- as.numeric(log_pmf(Q, support)$cpu())
  q_mass <- exp(log_q)

  alpha <- if (S7_inherits(Q, truncated_dirichlet)) Q@alpha else NULL
  is_point_mass <- S7_inherits(Q, discrete_simplex_mixture) &&
    ncol(Q@atoms) == 1L

  growth_under_q <- function(log_e) {
    sum(ifelse(q_mass > 0, q_mass * log_e, 0))
  }

  log_e_ripr <- lapply(2:K, function(j) {
    log_q - pairwise_log_pstar(log_q, X_cpu, j)
  })
  log_e_beta <- lapply(2:K, function(j) {
    if (!is.null(alpha)) {
      relaxed_pairwise_log_e(X_cpu, alpha, j)
    } else if (is_point_mass) {
      log_e_ripr[[j - 1L]]
    } else {
      NULL
    }
  })

  growth_rows <- function(variant, log_e_list) {
    if (any(vapply(log_e_list, is.null, logical(1L)))) {
      return(data.frame(
        variant = variant,
        pair = NA_integer_,
        growth = NA_real_
      ))
    }
    combined <- do.call(pmin, log_e_list)
    data.frame(
      variant = variant,
      pair = c(NA_integer_, 2:K),
      growth = c(
        growth_under_q(combined),
        vapply(log_e_list, growth_under_q, numeric(1L))
      )
    )
  }

  rbind(
    growth_rows("pairwise_ripr", log_e_ripr),
    growth_rows("pairwise_beta", log_e_beta)
  )
}
