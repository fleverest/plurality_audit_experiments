box::use(
  ggplot2[
    ggplot,
    aes,
    geom_raster,
    geom_line,
    geom_point,
    geom_segment,
    scale_fill_gradientn,
    scale_x_continuous,
    scale_x_discrete,
    scale_y_discrete,
    scale_y_log10,
    labs,
    theme_minimal,
    theme
  ],
  reshape2[melt],
  dplyr[filter, mutate]
)

#' Lollipop plot of boundary mixture weights
#'
#' Plots the weight assigned to each atom in the output of [run_boundary_ripr()].
#' The x-axis is the boundary parameter s in [0, 1], which traces the null
#' boundary from (1/2, 1/2, 0) through (1/3, 1/3, 1/3) to (1/2, 0, 1/2).
#'
#' @param boundary_result Output list from [run_boundary_ripr()].
#' @return A ggplot object.
#' @export
plot_boundary_weights <- function(boundary_result, cdf = FALSE) {
  # Invert null_boundary_3: recover s from atom theta vector.
  atom_to_s <- function(theta) {
    if (theta[2L] >= theta[3L]) 1.5 - 3 * theta[1L]  # segment 1: theta = (t,t,1-2t)
    else                        3 * theta[1L] - 0.5   # segment 2: theta = (t,1-2t,t)
  }

  s <- vapply(boundary_result$ws_list, atom_to_s, numeric(1L))
  w <- boundary_result$weights

  df <- data.frame(s = s, weight = w)

  if (cdf) {
    df <- df[order(df$s), ]
    df$weight <- cumsum(df$weight)
  }

  p <- ggplot(df, aes(x = s, y = weight))
  p <- if (cdf) p + geom_line(colour = "steelblue") + geom_point(size = 1, colour = "steelblue")
       else     p + geom_segment(aes(xend = s, yend = 0), linewidth = 0.4, colour = "steelblue") +
                    geom_point(size = 1.5, colour = "steelblue")

  p +
    scale_x_continuous(
      breaks = c(0, 0.5, 1),
      labels = c("(1/2,1/2,0)", "(1/3,1/3,1/3)", "(1/2,0,1/2)")
    ) +
    labs(
      x = "Boundary atom",
      y = if (cdf) "Cumulative weight" else "Mixture weight",
      title = sprintf("Boundary mixture (%d atoms)", boundary_result$n_atoms)
    ) +
    theme_minimal()
}

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
    geom_raster() +
    scale_fill_gradientn(
      colours = c("#440154", "#3b528b", "#21918c", "#5ec962", "#fde725")
    ) +
    labs(
      title = "Optimized Mixture Weights Across Random Restarts (Ordered by Loss)",
      x = "Mixture Component",
      y = "Loss (max expectation) achieved"
    ) +
    theme_minimal() +
    scale_y_discrete(
      limits = factor(utils::tail(rev(loss_order), top)),
      labels = function(x) {
        formatC(
          results_losses[as.numeric(as.character(x))] - 1,
          digits = 3,
          format = "e"
        )
      }
    ) +
    scale_x_discrete(
      breaks = function(x) {
        x[round(seq(1, length(x), length.out = min(5L, length(x))))]
      }
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
    ggplot(aes(x = Iteration, y = Loss - 1, group = Restart)) +
    geom_line(alpha = 0.7) +
    labs(
      title = "Loss Histories for Each Random Restart",
      x = "Iteration",
      y = "Max Expected LLR"
    ) +
    theme_minimal() +
    theme(legend.position = "none") +
    scale_y_log10()
}

#' Heatmap of expectation profile across DGPs per restart, ordered by loss
#'
#' Rows are restarts sorted by ascending loss (best at top); columns are DGP
#' indices. Fill shows E_theta[Q(X)/P_w(X)] at the final weights for that restart.
#'
#' @param results_expectation_profile Matrix of shape (R, T) — expectation per
#'   restart and DGP.
#' @param results_losses Numeric vector of length R — best loss per restart.
#' @param top Show only the `top` lowest-loss restarts. Default: all restarts.
#' @return A ggplot object.
#' @export
plot_results_expectation_profile <- function(
  results_expectation_profile,
  results_losses,
  top = Inf
) {
  profile_df <- as.data.frame(as.matrix(results_expectation_profile))
  colnames(profile_df) <- seq_len(ncol(profile_df))
  profile_df$restart <- seq_len(nrow(profile_df))

  profile_melt <- melt(
    profile_df,
    id.vars = "restart",
    variable.name = "Theta",
    value.name = "Expectation"
  )
  profile_melt$Theta <- as.integer(as.character(profile_melt$Theta))

  loss_order <- order(results_losses)

  profile_melt |>
    filter(restart %in% utils::head(loss_order, top)) |>
    mutate(restart = factor(restart, levels = loss_order)) |>
    ggplot(aes(x = Theta, y = restart, fill = Expectation)) +
    geom_raster() +
    scale_fill_gradientn(
      colours = c("#440154", "#3b528b", "#21918c", "#5ec962", "#fde725")
    ) +
    labs(
      title = "Expectation Profile Across DGPs per Restart (Ordered by Loss)",
      x = "DGP Index",
      y = "Loss achieved"
    ) +
    theme_minimal() +
    scale_x_discrete(
      breaks = function(x) {
        x[round(seq(1, length(x), length.out = min(5L, length(x))))]
      }
    ) +
    scale_y_discrete(
      limits = factor(utils::tail(rev(loss_order), top)),
      labels = function(x) {
        formatC(
          results_losses[as.numeric(as.character(x))] - 1,
          digits = 3,
          format = "e"
        )
      }
    )
}
