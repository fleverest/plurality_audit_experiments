box::use(
  ripr / pairwise_projection[pairwise_baseline_growth]
)

#' Closed-form pairwise baseline growth for one (n, Q) combination, tagged
#' with `Q_name` and `n` for downstream joins.
#' @export
build_pairwise_baseline <- function(q_info) {
  df <- pairwise_baseline_growth(q_info$Q, q_info$n)
  df$Q_name <- q_info$name
  df$n <- q_info$n
  df
}

#' Closed-form pairwise baseline growth across a dense n grid, for one Q.
#' Returns `NULL` when the full multinomial support would be too large to
#' enumerate (the resulting curve just ends early for that Q).
#' @export
build_growth_vs_n <- function(q_info, n) {
  if (choose(n + q_info$K - 1, q_info$K - 1) > 5e5) {
    return(NULL)
  }
  df <- pairwise_baseline_growth(q_info$Q, n)
  df$Q_name <- q_info$name
  df$n <- n
  df
}
