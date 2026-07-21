box::use(
  dplyr[filter, summarise, group_by, left_join, mutate, select]
)

# Human-readable labels for Q's and algo/config names, shared by the
# wrangling and plotting steps.

#' @export
q_map <- c(
  dirichlet_321 = "2-simplex (Dirichlet)",
  dirichlet_4321 = "3-simplex (Dirichlet)",
  dirichlet_54321 = "4-simplex (Dirichlet)",
  dirichlet_654321 = "5-simplex (Dirichlet)",
  point_321 = "2-simplex (point mass)",
  point_4321 = "3-simplex (point mass)",
  point_54321 = "4-simplex (point mass)",
  point_654321 = "5-simplex (point mass)"
)

#' @export
algo_map <- c(
  "fw_line-search" = "Line-search FW",
  fw_pairwise = "Pairwise FW",
  fw_vanilla = "Vanilla FW",
  "fw_fully-corrective" = "Fully-corrective FW",
  em_only = "Pure EM (random initialisation)",
  "hybrid_line-search" = "Line-search FW + EM refinement",
  hybrid_pairwise = "Pairwise FW + EM refinement",
  "hybrid_fully-corrective" = "Fully-corrective FW + EM refinement"
)

#' Stack every (config, Q) run's history into one long data frame, labelled
#' with human-readable Q/algo names, K (parsed from the Q label), mixture
#' type, and the KL gap to the best-seen `kl_ulb` for that (Q, n) — used to
#' plot log(KL - ULB) on a common, always-positive scale per facet.
#' @export
build_algos_df <- function(algo_comparison) {
  algos_df <- do.call(rbind, lapply(algo_comparison, `[[`, "history"))
  y_limits <- algos_df |>
    group_by(Q_name, n) |>
    summarise(ymin = max(kl_ulb), .groups = "drop")
  algos_df |>
    left_join(y_limits, by = c("Q_name", "n")) |>
    mutate(
      Q_label = unname(ifelse(
        Q_name %in% names(q_map),
        q_map[Q_name],
        Q_name
      )),
      K = as.integer(sub("(\\d+).*", "\\1", Q_label)),
      mixture_type = ifelse(
        grepl("Dirichlet", Q_label),
        "Dirichlet",
        "Point mass"
      ),
      algo_label = unname(ifelse(
        algo %in% names(algo_map),
        algo_map[algo],
        algo
      )),
      adj = kl_after_em - ymin,
      log_adj = log10(adj)
    )
}

#' Best growth-rate lower bound (`gr`, already computed by run_ripr) seen by
#' any algo for each (Q, n), alongside the closed-form pairwise RIPr/Beta
#' baselines for comparison.
#' @export
build_growth_summary <- function(algos_df, pairwise_baselines) {
  best_gro <- algos_df |>
    group_by(Q_name, n) |>
    summarise(
      gro_best = max(gr, na.rm = TRUE),
      .groups = "drop"
    )
  combined <- pairwise_baselines |> filter(is.na(pair))
  best_gro |>
    left_join(
      combined |>
        filter(variant == "pairwise_ripr") |>
        select(Q_name, n, pairwise_ripr = growth),
      by = c("Q_name", "n")
    ) |>
    left_join(
      combined |>
        filter(variant == "pairwise_beta") |>
        select(Q_name, n, pairwise_beta = growth),
      by = c("Q_name", "n")
    )
}
