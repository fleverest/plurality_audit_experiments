algo_comparison_plan <- function() {
  list(
    # -------------------------------------------------------------------------
    # Algorithm comparison experiment: pure EM vs pure FW (all variants) vs hybrid.
    # Pure EM sweeps random initialisations (1–20 atoms); pure FW and hybrid run
    # all fw_iters steps with gap_tol = 0 to prevent early stopping.
    # -------------------------------------------------------------------------
    tar_target(
      algo_comparison_n,
      10L
    ),

    tar_target(
      algo_comparison_maxiters,
      100L
    ),

    tar_target(
      algo_comparison_Q,
      {
        box::use(ripr / mixture[discrete_simplex_mixture, truncated_dirichlet])
        list(
          list(
            name = "point_321",
            K = 3L,
            Q = discrete_simplex_mixture(as.matrix(3:1 / sum(3:1)), 1)
          ),
          list(
            name = "dirichlet_321",
            K = 3L,
            Q = truncated_dirichlet(3:1)
          ),
          list(
            name = "point_4321",
            K = 4L,
            Q = discrete_simplex_mixture(
              as.matrix(4:1 / sum(4:1)),
              1
            )
          ),
          list(
            name = "dirichlet_4321",
            K = 4L,
            Q = truncated_dirichlet(4:1)
          ),
          list(
            name = "point_54321",
            K = 5L,
            Q = discrete_simplex_mixture(
              as.matrix(5:1 / sum(5:1)),
              1
            )
          ),
          list(
            name = "dirichlet_54321",
            K = 5L,
            Q = truncated_dirichlet(5:1)
          ),
          list(
            name = "point_654321",
            K = 6L,
            Q = discrete_simplex_mixture(
              as.matrix(6:1 / sum(6:1)),
              1
            )
          ),
          list(
            name = "dirichlet_654321",
            K = 6L,
            Q = truncated_dirichlet(6:1)
          )
        )
      }
    ),

    tar_target(
      algo_comparison_configs,
      c(
        lapply(seq_len(algo_comparison_maxiters), function(l) {
          list(
            name = "em_only",
            fw_iters = 0L,
            em_iters = 2000L,
            init = l,
            kl_atol = 1e-12,
            kl_rtol = 1e-10,
            verbose = TRUE
          )
        }),
        lapply(c("pairwise", "linesearch", "standard"), function(v) {
          list(
            name = paste0("fw_", v),
            fw_iters = algo_comparison_maxiters,
            em_iters = 0L,
            fw_variant = v,
            gap_tol = 0,
            init = NULL,
            verbose = TRUE
          )
        }),
        list(list(
          name = "hybrid_linesearch",
          fw_iters = algo_comparison_maxiters,
          em_iters = 10L,
          fw_variant = "linesearch",
          gap_tol = 0,
          init = NULL,
          verbose = TRUE
        )),
        list(list(
          name = "hybrid_pairwise",
          fw_iters = algo_comparison_maxiters,
          em_iters = 10L,
          fw_variant = "pairwise",
          gap_tol = 0,
          init = NULL,
          verbose = TRUE
        ))
      )
    ),

    tar_target(
      algo_comparison,
      {
        box::use(
          ripr / plurality_ripr[run_plurality_ripr],
          ripr / plurality_geometry[full_plurality_face_descriptors]
        )
        q_info <- algo_comparison_Q[[1L]]
        cfg <- algo_comparison_configs[[1L]]
        args <- c(
          list(
            q = q_info$Q,
            n = algo_comparison_n,
            face_descriptors = full_plurality_face_descriptors(q_info$K),
            checkpoint_iters = 0L:cfg$fw_iters
          ),
          cfg[setdiff(names(cfg), c("name"))]
        )
        result <- do.call(run_plurality_ripr, args)
        # Rewrite iter for em_only to reflect the number of atoms rather than steps.
        if (cfg$name == "em_only") {
          result$history$iter <- cfg$init
        }
        result$history$Q_name <- q_info$name
        result$history$algo <- cfg$name
        list(history = result$history, checkpoints = result$checkpoints)
      },
      pattern = cross(algo_comparison_configs, algo_comparison_Q),
      iteration = "list"
    ),

    tar_target(
      algos_df,
      {
        box::use(dplyr[filter, summarise, group_by, left_join, mutate])
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
        algo_map <- c(
          fw_linesearch = "Line-search FW",
          fw_pairwise = "Pairwise FW",
          fw_standard = "Standard FW",
          em_only = "Pure EM (random initialisation)",
          hybrid_linesearch = "Line-search FW + EM refinement",
          hybrid_pairwise = "Pairwise FW + EM refinement"
        )
        algos_df <- do.call(rbind, lapply(algo_comparison, `[[`, "history"))
        y_limits <- algos_df |>
          group_by(Q_name) |>
          summarise(ymin = max(kl_ulb), .groups = "drop")
        algos_df <- algos_df |>
          left_join(y_limits, by = "Q_name") |>
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

        algos_df
      }
    ),

    tar_target(
      algo_comparison_plots,
      {
        box::use(
          ggplot2[
            ggplot,
            aes,
            geom_line,
            facet_grid,
            labs,
            theme_minimal,
            scale_y_continuous,
            scale_y_log10,
            coord_cartesian,
            theme,
            element_text
          ],
          dplyr[
            filter,
            summarise,
            group_by,
            left_join,
            mutate,
            arrange,
            ungroup
          ],
        )

        # Prepare data with running max of log-growth-rate
        algos_df_with_runmax <- algos_df |>
          group_by(algo, Q_name, mixture_type, K) |>
          arrange(elapsed_s) |>
          mutate(
            log_growth_rate = kl_after_em - log(1 + gap),
            best_log_growth = cummax(log_growth_rate),
            K_name = paste0(K, "-simplex")
          ) |>
          ungroup()

        # Theme with larger fonts
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

        list(
          FW_KL_min_ULB = algos_df_with_runmax |>
            filter(
              algo %in% c("fw_pairwise", "fw_linesearch", "fw_standard")
            ) |>
            ggplot(aes(x = elapsed_s, y = log_adj, color = algo_label)) +
            geom_line(linewidth = 0.8) +
            facet_grid(
              K_name ~ mixture_type,
              scales = "free_y"
            ) +
            scale_y_continuous(expand = c(0, 0)) +
            labs(
              color = NULL,
              x = "Runtime (s)",
              y = "log(KL - ULB)",
              title = "FW variants: convergence comparison"
            ) +
            theme_larger(),

          FW_gap = algos_df_with_runmax |>
            filter(
              algo %in% c("fw_pairwise", "fw_linesearch", "fw_standard")
            ) |>
            ggplot(aes(x = elapsed_s, y = gap, color = algo_label)) +
            geom_line(linewidth = 0.8) +
            facet_grid(
              K_name ~ mixture_type,
              scales = "free_y"
            ) +
            scale_y_log10(expand = c(0, 0)) +
            labs(
              color = NULL,
              x = "Runtime (s)",
              y = "FW Gap",
              title = "FW variants: duality gap"
            ) +
            theme_larger(),

          FW_GR = algos_df_with_runmax |>
            filter(
              algo %in% c("fw_pairwise", "fw_linesearch", "fw_standard")
            ) |>
            ggplot(aes(
              x = elapsed_s,
              y = best_log_growth,
              color = algo_label
            )) +
            geom_line(linewidth = 0.8) +
            facet_grid(
              K_name ~ mixture_type,
              scales = "free_y"
            ) +
            scale_y_continuous(expand = c(0, 0)) +
            coord_cartesian(ylim = c(-0.05, NA)) +
            labs(
              color = NULL,
              x = "Runtime (s)",
              y = "Best E[log E]",
              title = "FW variants: best log-growth-rate over time"
            ) +
            theme_larger(),

          rest_KL_min_ULB = algos_df_with_runmax |>
            filter(!algo %in% c("fw_pairwise", "fw_standard")) |>
            ggplot(aes(x = elapsed_s, y = log_adj, color = algo_label)) +
            geom_line(linewidth = 0.8) +
            facet_grid(
              K_name ~ mixture_type,
              scales = "free"
            ) +
            scale_y_continuous(expand = c(0, 0)) +
            labs(
              color = NULL,
              x = "Runtime (s)",
              y = "log(KL - ULB)",
              title = "Algorithm comparison: convergence (log scale)"
            ) +
            theme_larger(),

          rest_gap = algos_df_with_runmax |>
            filter(!algo %in% c("fw_pairwise", "fw_standard")) |>
            ggplot(aes(x = elapsed_s, y = gap, color = algo_label)) +
            geom_line(linewidth = 0.8) +
            facet_grid(
              K_name ~ mixture_type,
              scales = "free"
            ) +
            scale_y_log10(expand = c(0, 0)) +
            labs(
              color = NULL,
              x = "Runtime (s)",
              y = "FW Gap",
              title = "Algorithm comparison: duality gap"
            ) +
            theme_larger(),

          rest_GR = algos_df_with_runmax |>
            filter(!algo %in% c("fw_pairwise", "fw_standard")) |>
            ggplot(aes(
              x = elapsed_s,
              y = best_log_growth,
              color = algo_label
            )) +
            geom_line(linewidth = 0.8) +
            facet_grid(
              K_name ~ mixture_type,
              scales = "free"
            ) +
            scale_y_continuous(expand = c(0, 0)) +
            coord_cartesian(ylim = c(-0.05, NA)) +
            labs(
              color = NULL,
              x = "Runtime (s)",
              y = "Best E[log E]",
              title = "Algorithm comparison: best log-growth-rate over time"
            ) +
            theme_larger()
        )
      }
    )
  )
}
