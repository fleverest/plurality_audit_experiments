algo_comparison_plan <- function() {
  list(
    # -------------------------------------------------------------------------
    # Algorithm comparison experiment: pure EM vs pure FW (all variants) vs hybrid.
    # Pure EM sweeps random initialisations (1–20 atoms); pure FW and hybrid run
    # all fw_iters steps with gap_tol = 0 to prevent early stopping.
    # -------------------------------------------------------------------------
    tar_target(
      algo_comparison_n_values,
      c(10L, 50L)
    ),

    tar_target(
      algo_comparison_maxiters,
      100L
    ),

    tar_target(
      algo_comparison_Q,
      {
        box::use(experiments / compare_algorithms / design[build_Q_list])
        build_Q_list()
      }
    ),

    tar_target(
      algo_comparison_nQ,
      {
        box::use(experiments / compare_algorithms / design[build_nQ])
        build_nQ(algo_comparison_n_values, algo_comparison_Q)
      }
    ),

    tar_target(
      algo_comparison_configs,
      {
        box::use(experiments / compare_algorithms / design[build_configs])
        build_configs(algo_comparison_maxiters)
      }
    ),

    tar_target(
      algo_comparison,
      {
        box::use(experiments / compare_algorithms / run[run_one])
        run_one(algo_comparison_nQ[[1L]], algo_comparison_configs[[1L]])
      },
      pattern = cross(algo_comparison_configs, algo_comparison_nQ),
      iteration = "list"
    ),

    tar_target(
      algos_df,
      {
        box::use(experiments / compare_algorithms / wrangle[build_algos_df])
        build_algos_df(algo_comparison)
      }
    ),

    # -------------------------------------------------------------------------
    # Pairwise baseline growth as a function of the batch size n. Closed-form,
    # so a dense n grid is cheap; combinations whose full multinomial support
    # would be too large to enumerate are skipped (their curves end early).
    # -------------------------------------------------------------------------
    tar_target(
      pairwise_growth_n_grid,
      c(1:10, 12L, 15L, 20L, 25L, 30L, 40L, 50L)
    ),

    tar_target(
      pairwise_growth_vs_n,
      {
        box::use(
          experiments / compare_algorithms / baselines[build_growth_vs_n]
        )
        build_growth_vs_n(algo_comparison_Q[[1L]], pairwise_growth_n_grid)
      },
      pattern = cross(algo_comparison_Q, pairwise_growth_n_grid)
    ),

    tar_target(
      pairwise_growth_vs_n_plots,
      {
        box::use(
          experiments / compare_algorithms / plots[build_growth_vs_n_plots]
        )
        build_growth_vs_n_plots(pairwise_growth_vs_n)
      }
    ),

    tar_target(
      growth_summary,
      {
        box::use(
          experiments / compare_algorithms / wrangle[build_growth_summary]
        )
        build_growth_summary(algos_df, pairwise_baselines)
      }
    ),

    tar_target(
      algo_comparison_plots,
      {
        box::use(
          experiments / compare_algorithms / plots[build_algo_comparison_plots]
        )
        build_algo_comparison_plots(algos_df, pairwise_baselines)
      }
    )
  )
}
