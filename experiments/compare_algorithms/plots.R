box::use(
  ggplot2[
    ggplot,
    aes,
    geom_line,
    geom_point,
    geom_hline,
    facet_grid,
    labs,
    theme_minimal,
    scale_y_continuous,
    scale_y_log10,
    coord_cartesian,
    theme,
    element_text
  ],
  dplyr[filter, group_by, mutate, arrange, ungroup],
)

#' ggplot theme shared by every plot in this experiment.
#' @export
theme_larger <- function() {
  theme_minimal() +
    theme(
      legend.position = "bottom",
      axis.title = element_text(size = 18),
      axis.text = element_text(size = 12),
      plot.title = element_text(size = 20, hjust = 0.5),
      strip.text = element_text(size = 15),
      legend.text = element_text(size = 12)
    )
}

#' One pairwise-baseline "growth vs batch size n" panel (per_batch or
#' per_ballot, depending on the pre-scaled `growth` column in `df`).
growth_vs_n_panel <- function(df, y_lab, title) {
  ggplot(df, aes(x = n, y = growth, color = variant_label)) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "grey50") +
    geom_line(linewidth = 0.8) +
    geom_point(size = 1.2) +
    facet_grid(K_name ~ mixture_type, scales = "free_y") +
    labs(color = NULL, x = "Batch size n", y = y_lab, title = title) +
    theme_larger()
}

#' Pairwise baseline growth (per-batch and per-ballot) vs batch size n, one
#' line per (K, mixture type, baseline variant).
#' @export
build_growth_vs_n_plots <- function(pairwise_growth_vs_n) {
  variant_map <- c(
    pairwise_ripr = "Pairwise RIPr (same prior)",
    pairwise_beta = "Pairwise Beta (relaxed prior)"
  )
  df <- pairwise_growth_vs_n |>
    filter(is.na(pair), !is.na(growth)) |>
    mutate(
      K_name = paste0(nchar(sub(".*_", "", Q_name)) - 1L, "-simplex"),
      mixture_type = ifelse(
        grepl("^dirichlet", Q_name),
        "Dirichlet",
        "Point mass"
      ),
      variant_label = unname(variant_map[variant])
    )

  list(
    per_batch = growth_vs_n_panel(
      df,
      "E[log E]",
      "Pairwise baseline: log-growth per batch vs batch size"
    ),
    per_ballot = growth_vs_n_panel(
      df |> mutate(growth = growth / n),
      "E[log E] / n",
      "Pairwise baseline: log-growth per ballot vs batch size"
    )
  )
}

#' One algo-comparison time series panel (KL, gap, or best-log-growth),
#' filtered to `algos` and faceted by (K, n) x mixture type.
algo_comparison_panel <- function(
  algos_df_with_runmax,
  algos,
  include,
  yvar,
  ylab,
  title,
  log_y = FALSE,
  baseline_df = NULL,
  growth_ymin = NULL
) {
  df <- if (include) {
    algos_df_with_runmax |> filter(algo %in% algos)
  } else {
    algos_df_with_runmax |> filter(!algo %in% algos)
  }
  p <- ggplot(df, aes(x = elapsed_s, y = .data[[yvar]], color = algo_label)) +
    geom_line(linewidth = 0.8)
  if (!is.null(baseline_df)) {
    p <- p +
      geom_hline(
        data = baseline_df,
        aes(yintercept = growth, linetype = variant_label),
        color = "grey30",
        linewidth = 0.6
      )
  }
  p <- p +
    facet_grid(
      K_name + n ~ mixture_type,
      scales = if (include) "free_y" else "free"
    )
  p <- if (log_y) {
    p + scale_y_log10(expand = c(0, 0))
  } else {
    p + scale_y_continuous(expand = c(0, 0))
  }
  if (!is.null(growth_ymin)) {
    p <- p + coord_cartesian(ylim = c(growth_ymin, NA))
  }
  p <- p +
    labs(color = NULL, x = "Runtime (s)", y = ylab, title = title) +
    theme_larger()
  if (!is.null(baseline_df)) {
    p <- p + labs(linetype = NULL)
  }
  p
}

#' The six algo-comparison panels: FW-variants-only and "rest" (hybrids +
#' pure EM + the line-search reference) versions of KL-vs-ULB, FW gap, and
#' best-log-growth-so-far, each faceted by (K, n) x mixture type.
#' @export
build_algo_comparison_plots <- function(algos_df, pairwise_baselines) {
  fw_variants <- c(
    "fw_pairwise",
    "fw_line-search",
    "fw_vanilla",
    "fw_fully-corrective"
  )

  # Running max of gr (log-growth-rate lower bound, already computed by
  # run_ripr) across each algo/Q/n run — spans multiple independent target
  # instances for em_only (one per random-init sweep step), so this
  # cross-run cummax can't be produced inside a single run_ripr call and
  # still has to happen here.
  algos_df_with_runmax <- algos_df |>
    group_by(algo, Q_name, n, mixture_type, K) |>
    arrange(elapsed_s) |>
    mutate(
      best_log_growth = cummax(gr),
      K_name = paste0(K, "-simplex")
    ) |>
    ungroup()

  # Pairwise baselines: one growth value per (Q, n, variant), drawn as
  # horizontal reference lines on the growth plots.
  variant_map <- c(
    pairwise_ripr = "Pairwise RIPr (same prior)",
    pairwise_beta = "Pairwise Beta (relaxed prior)"
  )
  baseline_df <- pairwise_baselines |>
    filter(is.na(pair), !is.na(growth)) |>
    mutate(
      K_name = paste0(nchar(sub(".*_", "", Q_name)) - 1L, "-simplex"),
      mixture_type = ifelse(
        grepl("^dirichlet", Q_name),
        "Dirichlet",
        "Point mass"
      ),
      variant_label = unname(variant_map[variant])
    )
  # Extend the growth plots' y floor to keep negative baselines visible.
  growth_ymin <- min(-0.05, min(baseline_df$growth, na.rm = TRUE) - 0.05)

  list(
    FW_KL_min_ULB = algo_comparison_panel(
      algos_df_with_runmax,
      fw_variants,
      include = TRUE,
      yvar = "log_adj",
      ylab = "log(KL - ULB)",
      title = "FW variants: convergence comparison"
    ),
    FW_gap = algo_comparison_panel(
      algos_df_with_runmax,
      fw_variants,
      include = TRUE,
      yvar = "gap",
      ylab = "FW Gap",
      title = "FW variants: duality gap",
      log_y = TRUE
    ),
    FW_GR = algo_comparison_panel(
      algos_df_with_runmax,
      fw_variants,
      include = TRUE,
      yvar = "best_log_growth",
      ylab = "Best E[log E]",
      title = "FW variants: best log-growth-rate over time",
      baseline_df = baseline_df,
      growth_ymin = growth_ymin
    ),
    rest_KL_min_ULB = algo_comparison_panel(
      algos_df_with_runmax,
      c("fw_pairwise", "fw_vanilla", "fw_fully-corrective"),
      include = FALSE,
      yvar = "log_adj",
      ylab = "log(KL - ULB)",
      title = "Algorithm comparison: convergence (log scale)"
    ),
    rest_gap = algo_comparison_panel(
      algos_df_with_runmax,
      c("fw_pairwise", "fw_vanilla", "fw_fully-corrective"),
      include = FALSE,
      yvar = "gap",
      ylab = "FW Gap",
      title = "Algorithm comparison: duality gap",
      log_y = TRUE
    ),
    rest_GR = algo_comparison_panel(
      algos_df_with_runmax,
      c("fw_pairwise", "fw_vanilla", "fw_fully-corrective"),
      include = FALSE,
      yvar = "best_log_growth",
      ylab = "Best E[log E]",
      title = "Algorithm comparison: best log-growth-rate over time",
      baseline_df = baseline_df,
      growth_ymin = growth_ymin
    )
  )
}
