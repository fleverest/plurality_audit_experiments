library(targets)
library(crew)
library(crew.cluster)
library(nanonext)


# Second IP address is reachable from other compute nodes on M3
is_m3 <- grepl("m3", Sys.info()[["nodename"]])
gpu_controller <- if (!is_m3) {
  crew_controller_local(name = "gpu", workers = 1L)
} else {
  crew_controller_slurm(
    name = "gpu",
    host = ip_addr()[2L],
    workers = 2L, # QOSMaxGRESPerUser is 2 for me
    options_cluster = crew_options_slurm(
      partition = "gpu",
      time_minutes = 240L,
      log_output = "logs/crew_gpu_%A_%a.out",
      log_error = "logs/crew_gpu_%A_%a.err",
      script_lines = c(
        "#SBATCH --gres=gpu:A100:1",
        "module load r cuda/12.6"
      )
    )
  )
}

cpu_controller <- if (!is_m3) {
  crew_controller_local(name = "cpu", workers = 4L)
} else {
  crew_controller_slurm(
    name = "cpu",
    host = ip_addr()[2L],
    workers = 95L, # My QOS allows 100 jobs, but leave some headroom for the controller and gpu jobs, plus an interactive job or two.
    seconds_idle = 120L,
    options_cluster = crew_options_slurm(
      partition = "comp", # your CPU partition
      cpus_per_task = 1L,
      memory_gigabytes_per_cpu = 10, # MaxTRESPU is 1TB total
      time_minutes = 60L,
      log_output = "logs/crew_cpu_%A_%a.out",
      log_error = "logs/crew_cpu_%A_%a.err",
      script_lines = "module load r"
    )
  )
}

tar_option_set(
  seed = 20260516L,
  controller = crew_controller_group(cpu_controller, gpu_controller),
  resources = tar_resources(
    crew = tar_resources_crew(controller = "cpu")
  )
)


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
    50L
  ),

  tar_target(
    algo_comparison_Q,
    {
      box::use(ripr / mixture[discrete_simplex_mixture, truncated_dirichlet])
      list(
        list(
          name = "point_543",
          K = 3L,
          Q = discrete_simplex_mixture(as.matrix(5:3 / sum(5:3)), 1)
        ),
        list(
          name = "dirichlet_543",
          K = 3L,
          Q = truncated_dirichlet(5:3)
        ),
        list(
          name = "point_5432",
          K = 4L,
          Q = discrete_simplex_mixture(
            as.matrix(5:2 / sum(5:2)),
            1
          )
        ),
        list(
          name = "dirichlet_5432",
          K = 4L,
          Q = truncated_dirichlet(5:2)
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
          Q = truncated_dirichlet(c(6, 5, 4, 3, 2, 1))
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
          em_iters = 1000L,
          init = l,
          kl_atol = 1e-12,
          kl_rtol = 1e-9
        )
      }),
      lapply(c("pairwise", "linesearch", "standard"), function(v) {
        list(
          name = paste0("fw_", v),
          fw_iters = algo_comparison_maxiters,
          em_iters = 0L,
          fw_variant = v,
          gap_tol = 0,
          init = NULL
        )
      }),
      list(list(
        name = "hybrid",
        fw_iters = algo_comparison_maxiters,
        em_iters = 10L, # Arbitrary but illustrates improvement over pure variants.
        fw_variant = "pairwise",
        gap_tol = 0,
        init = NULL
      ))
    )
  ),

  tar_target(
    algo_comparison,
    {
      box::use(ripr / ripr_experiments[run_plurality_ripr])
      q_info <- algo_comparison_Q[[1L]]
      cfg <- algo_comparison_configs[[1L]]
      args <- c(
        list(q = q_info$Q, n = algo_comparison_n),
        cfg[setdiff(names(cfg), c("name"))]
      )
      result <- do.call(run_plurality_ripr, args)
      # Rewrite iter for em_only to reflect the number of atoms rather than steps.
      if (cfg$name == "em_only") {
        result$history$iter <- cfg$init
      }
      result$history$Q_name <- q_info$name
      result$history$algo <- cfg$name
      result$history
    },
    pattern = cross(algo_comparison_configs, algo_comparison_Q),
    iteration = "list"
  ),

  tar_target(
    algos_df,
    do.call(rbind, algo_comparison)
  ),

  # -------------------------------------------------------------------------
  # Compute the RIPr (optimal P_W) for each Q alternative and batch size.
  # This is the most computationally intensive step, so we do it first and on
  # the GPU if available. The resulting P_W's are then used in the BatchRIPr
  # target below, which runs on the CPU but is much faster.
  # -------------------------------------------------------------------------
  # tar_target(
  #   max_n,
  #   100L
  # ),
  # tar_target(n_vals, as.list(seq_len(max_n))),
  # # Alternative distributions Q used as the numerator in each martingale.
  # # Both point and Dirichlet mixture variants are included for comparison.
  # # InfimumMartingale only runs for point alternatives (single-atom mixture_mnom).
  # tar_target(
  #   sim_Q,
  #   {
  #     box::use(ripr / mixture[discrete_simplex_mixture, truncated_dirichlet])
  #     list(
  #       list(
  #         name = "point_754",
  #         K = 3L,
  #         Q = discrete_simplex_mixture(as.matrix(c(7 / 16, 5 / 16, 4 / 16)), 1)
  #       ),
  #       list(
  #         name = "dirichlet_754",
  #         K = 3L,
  #         Q = truncated_dirichlet(c(7, 5, 4))
  #       ),
  #       list(
  #         name = "point_855",
  #         K = 3L,
  #         Q = discrete_simplex_mixture(as.matrix(c(8 / 18, 5 / 18, 5 / 18)), 1)
  #       ),
  #       list(
  #         name = "dirichlet_855",
  #         K = 3L,
  #         Q = truncated_dirichlet(c(8, 5, 5))
  #       ),
  #       list(
  #         name = "point_7531",
  #         K = 4L,
  #         Q = discrete_simplex_mixture(
  #           as.matrix(c(7 / 16, 5 / 16, 3 / 16, 1 / 16)),
  #           1
  #         )
  #       ),
  #       list(
  #         name = "dirichlet_7531",
  #         K = 4L,
  #         Q = truncated_dirichlet(c(7, 5, 3, 1))
  #       ),
  #       list(
  #         name = "point_8444",
  #         K = 4L,
  #         Q = discrete_simplex_mixture(
  #           as.matrix(c(8 / 20, 4 / 20, 4 / 20, 4 / 20)),
  #           1
  #         )
  #       ),
  #       list(
  #         name = "dirichlet_8444",
  #         K = 4L,
  #         Q = truncated_dirichlet(c(8, 4, 4, 4))
  #       )
  #     )
  #   }
  # ),
  # tar_target(
  #   sim_ripr_optimal,
  #   {
  #     q_opt <- sim_Q[[1L]]
  #     n <- n_vals[[1L]]
  #     result <- run_plurality_ripr(
  #       n = n,
  #       q = q_opt$Q,
  #       fw_iters = 25L,
  #       em_iters = 10L,
  #       verbose = TRUE
  #     )
  #     list(
  #       Q_name = q_opt$name,
  #       K = q_opt$K,
  #       Q = q_opt$Q,
  #       P_W = result$mixture,
  #       e_ratio = result$gap
  #     )
  #   },
  #   pattern = cross(sim_Q, n_vals),
  #   iteration = "list"
  # ),

  # # --------------------------------------------------------------------------
  # # Grand Prix: comparison of UITest, InfimumMartingale, BatchRIPr martingales.
  # # Mirrors grand_prix/simulations.R with a proper targets pipeline.
  # # --------------------------------------------------------------------------

  # # Scenarios: true data-generating distributions. Sequences are 100 draws from
  # # multiomial(1, q).
  # tar_target(
  #   sim_scenario,
  #   list(
  #     list(name = "point_754", q = c(7 / 16, 5 / 16, 4 / 16), K = 3L),
  #     list(name = "point_855", q = c(8 / 18, 5 / 18, 5 / 18), K = 3L),
  #     list(name = "point_7531", q = c(7 / 16, 5 / 16, 3 / 16, 1 / 16), K = 4L),
  #     list(name = "point_8444", q = c(8 / 20, 4 / 20, 4 / 20, 4 / 20), K = 4L)
  #   )
  # ),

  # # 1000L sequences per scenario.
  # tar_target(
  #   sim_reps,
  #   1000L
  # ),
  # # (sim_reps, max_n) matrices of simulated sequences for each scenario.
  # tar_target(
  #   sim_seqs,
  #   c(
  #     sim_scenario[[1L]],
  #     list(
  #       data = replicate(
  #         sim_reps,
  #         sample(
  #           seq_len(sim_scenario[[1L]]$K),
  #           size = max_n,
  #           replace = TRUE,
  #           prob = sim_scenario[[1L]]$q
  #         )
  #       )
  #     )
  #   ),
  #   pattern = map(sim_scenario),
  #   iteration = "list"
  # ),

  # # UITest (Universal Inference) — one branch per scenario, inner lapply over
  # # matching Q alternatives (same K). Returns a list of {Q_name, dgp_name,
  # # values (max_n x sim_reps matrix)}, one entry per matching Q.
  # tar_target(
  #   universal_inference,
  #   {
  #     ss <- sim_seqs
  #     lapply(
  #       Filter(function(sq) sq$K == ss$K, sim_Q),
  #       function(sq) {
  #         list(
  #           Q_name = sq$name,
  #           dgp_name = ss$name,
  #           values = run_martingale(UITest(sq$Q, 0.05), ss$data)
  #         )
  #       }
  #     )
  #   },
  #   pattern = map(sim_seqs),
  #   iteration = "list"
  # ),

  # # InfimumMartingale — same structure as universal_inference.
  # tar_target(
  #   infimum_martingale,
  #   {
  #     ss <- sim_seqs
  #     lapply(
  #       Filter(function(sq) sq$K == ss$K, sim_Q),
  #       function(sq) {
  #         list(
  #           Q_name = sq$name,
  #           dgp_name = ss$name,
  #           values = run_martingale(InfimumMartingale(sq$Q, 0.05), ss$data)
  #         )
  #       }
  #     )
  #   },
  #   pattern = map(sim_seqs),
  #   iteration = "list"
  # ),

  # # RIPrSequence — uncorrected Y_t = Q(X^t) / P_W^t(X^t), same structure as
  # # universal_inference. P_W^t is looked up from sim_ripr_optimal by n.
  # tar_target(
  #   ripr_sequence,
  #   {
  #     ss <- sim_seqs
  #     lapply(
  #       Filter(function(sq) sq$K == ss$K, sim_Q),
  #       function(sq) {
  #         q_name <- sq$name
  #         # Function to retrieve the appropriate P_W and e_ratio from sim_ripr_optimal for a given n.
  #         ripr_fn <- function(n) {
  #           entry <- Filter(
  #             function(r) r$Q_name == q_name && r$P_W@n == n,
  #             sim_ripr_optimal
  #           )[[1L]]
  #           list(mixture = entry$P_W, e_ratio = entry$e_ratio)
  #         }
  #         list(
  #           Q_name = sq$name,
  #           dgp_name = ss$name,
  #           values = run_martingale(RIPrSequence(sq$Q, ripr_fn, 0.05), ss$data)
  #         )
  #       }
  #     )
  #   },
  #   pattern = map(sim_seqs),
  #   iteration = "list"
  # ),

  # # BatchRIPr — one branch per scenario, nested lapply over matching Q
  # # alternatives and batch sizes. Returns a nested list of {Q_name, dgp_name,
  # # batch_size, values (max_n x sim_reps matrix)}.
  # tar_target(
  #   batch_sizes,
  #   list(1L, 2L) #, 5L, 10L, 25L)
  # ),
  # tar_target(
  #   batch_ripr,
  #   {
  #     ss <- sim_seqs
  #     bs <- batch_sizes
  #     lapply(
  #       Filter(function(sq) sq$K == ss$K, sim_Q),
  #       function(sq) {
  #         ripr_entry <- Filter(
  #           function(r) r$Q_name == sq$name && r$P_W@n == bs,
  #           sim_ripr_optimal
  #         )[[1L]]
  #         list(
  #           Q_name = sq$name,
  #           dgp_name = ss$name,
  #           batch_size = bs,
  #           values = run_martingale(
  #             BatchRIPr(sq$Q, ripr_entry$P_W, ripr_entry$e_ratio, 0.05),
  #             ss$data
  #           )
  #         )
  #       }
  #     )
  #   },
  #   pattern = cross(sim_seqs, batch_sizes),
  #   iteration = "list"
  # ),

  # # -------------------------------------------------------------------------
  # # Plots: mean e-process vs step, one plot per DGP.
  # # Colour = martingale type, linetype = Point/Dirichlet Q, facets = Q mode.
  # # -------------------------------------------------------------------------
  # tar_target(
  #   sim_long,
  #   build_sim_long(
  #     universal_inference,
  #     infimum_martingale,
  #     ripr_sequence,
  #     batch_ripr
  #   )
  # ),

  # tar_target(sim_plot_754, plot_mean_martingales(sim_long, "point_754", 3L)),
  # tar_target(sim_plot_855, plot_mean_martingales(sim_long, "point_855", 3L)),
  # tar_target(sim_plot_7531, plot_mean_martingales(sim_long, "point_7531", 4L)),
  # tar_target(sim_plot_8444, plot_mean_martingales(sim_long, "point_8444", 4L))

  NULL
)
