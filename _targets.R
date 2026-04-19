library(targets)
library(crew.cluster)

box::use(
  ripr /
    plots[
      plot_results_weights,
      plot_results_loss_history,
      plot_results_expectation_profile
    ],
  ripr / grids[make_simplex_grid],
  ripr / experiment_fns[run_ripr_target]
)

gpu_controller <- crew_controller_slurm(
  name = "gpu",
  host = nanonext::ip_addr()[2L], # Second IP address is reachable from other compute nodes on M3
  workers = 2L, # QOSMaxGRESPerUser is 2 for me
  options_cluster = crew_options_slurm(
    partition = "gpu",
    time_minutes = 120L,
    log_output = "logs/crew_%A_%a.out",
    log_error = "logs/crew_%A_%a.err",
    script_lines = c(
      "#SBATCH --gres=gpu:A100:1",
      "module load r cuda/12.6"
    )
  )
)

tar_option_set(
  seed = 20251111,
  controller = gpu_controller
)

n_config <- data.frame(
  n = c(1:5, 10, 20, 50, 100),
  n_restarts = c(rep(100L, 5), rep(50L, 4))
)

list(
  # Lightweight targets run locally in the orchestrator process
  tar_target(q, c(7 / 16, 5 / 16, 4 / 16), deployment = "main"),
  tar_target(thetas, make_simplex_grid(1001), deployment = "main"),
  tar_target(ws, make_simplex_grid(501), deployment = "main"),

  tarchetypes::tar_map(
    values = n_config,
    names = "n",
    # result targets are dispatched to SLURM GPU jobs
    tar_target(result, {
      box::use(ripr / experiment_fns[run_ripr_target])
      run_ripr_target(n, n_restarts, thetas, q, ws)
    }),
    # plot/analysis targets are cheap — run in the orchestrator
    tar_target(
      plot_weights,
      plot_results_weights(result$weights, result$final_loss, top = 50),
      deployment = "main"
    ),
    tar_target(
      plot_loss_history,
      plot_results_loss_history(log(1 - result$loss_history)),
      deployment = "main"
    ),
    tar_target(
      plot_expectation_profile,
      plot_results_expectation_profile(
        result$expectation_profile,
        result$final_loss,
        top = 50
      ),
      deployment = "main"
    )
  )
)
