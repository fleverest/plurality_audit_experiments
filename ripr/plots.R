box::use(
  ggplot2[
    ggplot,
    aes,
    geom_tile,
    geom_line,
    scale_fill_viridis_c,
    labs,
    theme_minimal,
    theme,
    scale_y_discrete
  ],
  reshape2[melt],
  dplyr[filter, mutate]
)

#' Heatmap of optimised mixture weights across restarts, ordered by loss
#'
#' Rows are restarts sorted by ascending loss (best at top); columns are mixture
#' components. Useful for spotting whether the optimiser converges to a single
#' solution or finds a family of local optima.
#'
#' @param results_weights Matrix of shape (R, C) — mixture weights per restart.
#' @param results_losses Numeric vector of length R — best loss per restart.
#' @param top Show only the `top` lowest-loss restarts. Default: all restarts.
#' @return A ggplot object.
#' @export
plot_results_weights <- function(results_weights, results_losses, top = Inf) {
  weights_df <- as.data.frame(as.matrix(results_weights))
  colnames(weights_df) <- as.character(seq_len(ncol(weights_df)))
  weights_df$restart <- seq_len(nrow(weights_df))

  weights_melt <- melt(
    weights_df,
    variable.name = "Component",
    value.name = "Weight",
    id.vars = "restart"
  )
  loss_order <- order(results_losses)

  weights_melt |>
    filter(restart %in% utils::head(loss_order, top)) |>
    mutate(restart = factor(restart, levels = loss_order)) |>
    ggplot(aes(x = Component, y = restart, fill = Weight)) +
    geom_tile() +
    scale_fill_viridis_c() +
    labs(
      title = "Optimized Mixture Weights Across Random Restarts (Ordered by Loss)",
      x = "Mixture Component",
      y = "Loss (max expectation) achieved"
    ) +
    theme_minimal() +
    scale_y_discrete(
      limits = factor(utils::tail(rev(loss_order), top)),
      labels = function(x) results_losses[as.numeric(as.character(x))]
    )
}

#' Line plot of loss convergence per restart over optimisation iterations
#'
#' Each line is one restart; the y-axis is whatever transformation of the
#' tracked loss the caller passes in (typically `log(1 - losses_history)` to
#' show progress toward the lower bound).
#'
#' @param results_loss_history Matrix of shape (iterations/100, R) — loss
#'   snapshots recorded every 100 iterations during optimisation.
#' @return A ggplot object.
#' @export
plot_results_loss_history <- function(results_loss_history) {
  losses_df <- as.data.frame(results_loss_history)
  colnames(losses_df) <- paste0("Restart", seq_len(ncol(losses_df)))
  losses_df$Iteration <- seq_len(nrow(losses_df))

  melt(
    losses_df,
    id.vars = "Iteration",
    variable.name = "Restart",
    value.name = "Loss"
  ) |>
    ggplot(aes(x = Iteration, y = Loss, color = Restart)) +
    geom_line(alpha = 0.7) +
    labs(
      title = "Loss Histories for Each Random Restart",
      x = "Iteration",
      y = "Max Expected LLR"
    ) +
    theme_minimal() +
    theme(legend.position = "none")
}
