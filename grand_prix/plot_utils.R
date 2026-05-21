box::use(
  dplyr[bind_rows, mutate],
  tibble[tibble],
  ggplot2[
    ggplot,
    aes,
    geom_line,
    geom_hline,
    facet_wrap,
    vars,
    scale_y_log10,
    scale_linetype_manual,
    theme_minimal,
    labs
  ]
)

#' Gather branched martingale targets into a long-format tibble of rep-means
#'
#' @param universal_inference Branched list from the `universal_inference` target.
#' @param infimum_martingale Branched list from the `infimum_martingale` target.
#' @param ripr_sequence Branched list from the `ripr_sequence` target.
#' @param batch_ripr Branched list from the `batch_ripr` target.
#' @return Tibble: step, mean_value, Q_name, dgp_name, martingale, batch_size,
#'   q_mode, q_family ("Point"/"Dirichlet"), K (integer).
#' @export
build_sim_long <- function(
  universal_inference,
  infimum_martingale,
  ripr_sequence,
  batch_ripr
) {
  entry_to_tbl <- function(entry, martingale_type) {
    n_steps <- nrow(entry$values)
    tibble(
      step = seq_len(n_steps) - 1L,
      mean_value = rowMeans(entry$values),
      Q_name = entry$Q_name,
      dgp_name = entry$dgp_name,
      martingale = martingale_type,
      batch_size = if (!is.null(entry$batch_size)) {
        entry$batch_size[[1L]]
      } else {
        NA_integer_
      }
    )
  }

  ui_tbl <- bind_rows(lapply(
    do.call(c, universal_inference),
    entry_to_tbl,
    martingale_type = "UITest"
  ))
  im_tbl <- bind_rows(lapply(
    do.call(c, infimum_martingale),
    entry_to_tbl,
    martingale_type = "InfimumMartingale"
  ))
  rs_tbl <- bind_rows(lapply(
    do.call(c, ripr_sequence),
    entry_to_tbl,
    martingale_type = "RIPrSequence"
  ))
  batch_tbl <- bind_rows(lapply(
    do.call(c, batch_ripr),
    function(e) entry_to_tbl(e, "BatchRIPr")
  ))

  bind_rows(ui_tbl, im_tbl, rs_tbl, batch_tbl) |>
    mutate(
      q_mode = sub("^(point|dirichlet)_", "", Q_name),
      q_family = ifelse(grepl("^point_", Q_name), "Point", "Dirichlet"),
      K = nchar(q_mode)
    )
}

#' Plot mean e-process values for one DGP
#'
#' Colour = martingale type, linetype = Point vs Dirichlet Q, facets = Q mode.
#'
#' @param sim_long Tibble from [build_sim_long()].
#' @param dgp_name Character. DGP name, e.g. "point_754".
#' @param K Integer. Number of candidates.
#' @param alpha Numeric. Significance level; draws a dotted rejection line.
#' @return A ggplot object.
#' @export
plot_mean_martingales <- function(sim_long, dgp_name, K, alpha = 0.05) {
  plot_data <- sim_long[sim_long$dgp_name == dgp_name & sim_long$K == K, ]
  plot_data[is.na(plot_data$batch_size), "batch_size"] <- 0L

  ggplot(
    plot_data,
    aes(
      x = step,
      y = mean_value,
      colour = martingale,
      linetype = q_family,
      group = interaction(q_family, martingale, batch_size)
    )
  ) +
    geom_line() +
    facet_wrap(vars(q_mode)) +
    scale_y_log10() +
    scale_linetype_manual(
      values = c("Point" = "solid", "Dirichlet" = "dashed")
    ) +
    theme_minimal() +
    labs(
      title = sprintf("DGP: %s  (K = %d)", dgp_name, K),
      x = "Step",
      y = "Mean e-process (log scale)",
      colour = "Martingale",
      linetype = "Alternative Q"
    )
}
