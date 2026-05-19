box::use(
  seqan[update, reset, value]
)

#' Run a martingale sequentially over a matrix of observation sequences
#'
#' Resets `martingale_stat` before each sequence, feeds the full sequence via
#' `update()`, and extracts the per-step e-process values. The initial value of
#' 1 (before any observations) is dropped; the returned matrix has one row per
#' observation.
#'
#' @param martingale_stat A `Test` object (UITest, InfimumMartingale, BatchRIPr, …).
#' @param sequences Integer matrix of shape (max_n × sim_reps). Each column is
#'   one observation sequence.
#' @return Numeric matrix of shape (max_n × sim_reps) — e-process values at
#'   each observation step, one column per sequence.
#' @export
run_martingale <- function(martingale_stat, sequences) {
  seq_len <- nrow(sequences)
  n_seqs <- ncol(sequences)
  mat <- matrix(0, nrow = seq_len + 1L, ncol = n_seqs)
  for (i in seq_len(n_seqs)) {
    reset(martingale_stat)
    suppressMessages(update(martingale_stat, sequences[, i]))
    mat[, i] <- value(martingale_stat, Inf)
  }
  mat
}
