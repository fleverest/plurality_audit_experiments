library(targets)
library(crew.cluster)

box::use(
  ripr / mixture[point_mnom, dirichlet_mnom],
  ripr / simplex_utils[simplex_lattice],
  ripr / ripr_experiments[run_plurality_ripr],
  # These S7 modules aren't working for some reason.
  #  grand_prix / ui_mart[UITest],
  #  grand_prix / im_mart[InfimumMartingale],
  #  grand_prix / ripr_mart[BatchRIPr]
  grand_prix / martingale_sequences[run_martingale],
  grand_prix / plot_utils[build_sim_long, plot_mean_martingales]
)
source("grand_prix/ui_mart.R")
source("grand_prix/im_mart.R")
source("grand_prix/ripr_mart.R")

# Second IP address is reachable from other compute nodes on M3
is_m3 <- grepl("m3", Sys.info()[["nodename"]])
gpu_controller <- if (!is_m3) {
  NULL
} else {
  crew_controller_slurm(
    name = "gpu",
    host = nanonext::ip_addr()[2L],
    workers = 2L, # QOSMaxGRESPerUser is 2 for me
    options_cluster = crew_options_slurm(
      partition = "gpu",
      time_minutes = 240L,
      log_output = "logs/crew_%A_%a.out",
      log_error = "logs/crew_%A_%a.err",
      script_lines = c(
        "#SBATCH --gres=gpu:A100:1",
        "module load r cuda/12.6"
      )
    )
  )
}

tar_option_set(
  seed = 20260516L,
  controller = gpu_controller
)


# Define the alternatives of interest.
simplex_k3_alt <- simplex_lattice(3, 50L) / 50L
simplex_k3_alt <- t(simplex_k3_alt[
  simplex_k3_alt[, 1L] > simplex_k3_alt[, 2L] &
    simplex_k3_alt[, 1L] > simplex_k3_alt[, 3L],
])
simplex_k4_alt <- simplex_lattice(4, 25L) / 25L
simplex_k4_alt <- t(simplex_k4_alt[
  simplex_k4_alt[, 1L] > simplex_k4_alt[, 2L] &
    simplex_k4_alt[, 1L] > simplex_k4_alt[, 3L] &
    simplex_k4_alt[, 1L] > simplex_k4_alt[, 4L],
])

list(
  tar_target(
    max_n,
    100L
  ),

  tar_target(n_vals, as.list(seq_len(max_n))),

  # Alternative distributions Q used as the numerator in each martingale.
  # Both point and Dirichlet mixture variants are included for comparison.
  # InfimumMartingale only runs for point alternatives (single-atom mixture_mnom).
  tar_target(
    sim_Q,
    list(
      list(
        name = "point_754",
        K = 3L,
        Q = point_mnom(q = c(7 / 16, 5 / 16, 4 / 16))
      ),
      list(
        name = "dirichlet_754",
        K = 3L,
        Q = dirichlet_mnom(alpha = c(7, 5, 4), atoms = simplex_k3_alt)
      ),
      list(
        name = "point_855",
        K = 3L,
        Q = point_mnom(q = c(8 / 18, 5 / 18, 5 / 18))
      ),
      list(
        name = "dirichlet_855",
        K = 3L,
        Q = dirichlet_mnom(alpha = c(8, 5, 5), atoms = simplex_k3_alt)
      ),
      list(
        name = "point_7531",
        K = 4L,
        Q = point_mnom(q = c(7 / 16, 5 / 16, 3 / 16, 1 / 16))
      ),
      list(
        name = "dirichlet_7531",
        K = 4L,
        Q = dirichlet_mnom(alpha = c(7, 5, 3, 1), atoms = simplex_k4_alt)
      ),
      list(
        name = "point_8444",
        K = 4L,
        Q = point_mnom(q = c(8 / 20, 4 / 20, 4 / 20, 4 / 20))
      ),
      list(
        name = "dirichlet_8444",
        K = 4L,
        Q = dirichlet_mnom(alpha = c(8, 4, 4, 4), atoms = simplex_k4_alt)
      )
    )
  ),

  # -------------------------------------------------------------------------
  # Compute the RIPr optimal P_W for each Q alternative and batch size.
  # This is the most computationally intensive step, so we do it first and on
  # the GPU if available. The resulting P_W's are then used in the BatchRIPr
  # target below, which runs on the CPU but is much faster.
  # -------------------------------------------------------------------------
  tar_target(
    sim_ripr_optimal,
    {
      q_opt <- sim_Q[[1L]]
      n <- n_vals[[1L]]
      result <- run_plurality_ripr(
        n = n,
        q = q_opt$Q,
        max_atoms_added = 50L,
        n_em_iter = 25L,
        verbose = TRUE
      )
      list(
        Q_name = q_opt$name,
        K = q_opt$K,
        Q = q_opt$Q,
        P_W = result$mixture,
        e_ratio = result$E_star
      )
    },
    pattern = cross(sim_Q, n_vals),
    iteration = "list"
  ),

  # --------------------------------------------------------------------------
  # Grand Prix: comparison of UITest, InfimumMartingale, BatchRIPr martingales.
  # Mirrors grand_prix/simulations.R with a proper targets pipeline.
  # --------------------------------------------------------------------------

  # Scenarios: true data-generating distributions. Sequences are 100 draws from
  # multiomial(1, q).
  tar_target(
    sim_scenario,
    list(
      list(name = "point_754", q = c(7 / 16, 5 / 16, 4 / 16), K = 3L),
      list(name = "point_855", q = c(8 / 18, 5 / 18, 5 / 18), K = 3L),
      list(name = "point_7531", q = c(7 / 16, 5 / 16, 3 / 16, 1 / 16), K = 4L),
      list(name = "point_8444", q = c(8 / 20, 4 / 20, 4 / 20, 4 / 20), K = 4L)
    )
  ),

  # 1000L sequences per scenario.
  tar_target(
    sim_reps,
    1000L
  ),
  # (sim_reps, max_n) matrices of simulated sequences for each scenario.
  tar_target(
    sim_seqs,
    c(
      sim_scenario[[1L]],
      list(
        data = replicate(
          sim_reps,
          sample(
            seq_len(sim_scenario[[1L]]$K),
            size = max_n,
            replace = TRUE,
            prob = sim_scenario[[1L]]$q
          )
        )
      )
    ),
    pattern = map(sim_scenario),
    iteration = "list"
  ),

  # UITest (Universal Inference) — one branch per scenario, inner lapply over
  # matching Q alternatives (same K). Returns a list of {Q_name, dgp_name,
  # values (max_n x sim_reps matrix)}, one entry per matching Q.
  tar_target(
    universal_inference,
    {
      ss <- sim_seqs
      lapply(
        Filter(function(sq) sq$K == ss$K, sim_Q),
        function(sq) {
          list(
            Q_name = sq$name,
            dgp_name = ss$name,
            values = run_martingale(UITest(sq$Q, 0.05), ss$data)
          )
        }
      )
    },
    pattern = map(sim_seqs),
    iteration = "list"
  ),

  # InfimumMartingale — same structure as universal_inference.
  tar_target(
    infimum_martingale,
    {
      ss <- sim_seqs
      lapply(
        Filter(function(sq) sq$K == ss$K, sim_Q),
        function(sq) {
          list(
            Q_name = sq$name,
            dgp_name = ss$name,
            values = run_martingale(InfimumMartingale(sq$Q, 0.05), ss$data)
          )
        }
      )
    },
    pattern = map(sim_seqs),
    iteration = "list"
  ),

  # RIPrSequence — uncorrected Y_t = Q(X^t) / P_W^t(X^t), same structure as
  # universal_inference. P_W^t is looked up from sim_ripr_optimal by n.
  tar_target(
    ripr_sequence,
    {
      ss <- sim_seqs
      lapply(
        Filter(function(sq) sq$K == ss$K, sim_Q),
        function(sq) {
          q_name <- sq$name
          # Function to retrieve the appropriate P_W and e_ratio from sim_ripr_optimal for a given n.
          ripr_fn <- function(n) {
            entry <- Filter(
              function(r) r$Q_name == q_name && r$P_W@n == n,
              sim_ripr_optimal
            )[[1L]]
            list(mixture = entry$P_W, e_ratio = entry$e_ratio)
          }
          list(
            Q_name = sq$name,
            dgp_name = ss$name,
            values = run_martingale(RIPrSequence(sq$Q, ripr_fn, 0.05), ss$data)
          )
        }
      )
    },
    pattern = map(sim_seqs),
    iteration = "list"
  ),

  # BatchRIPr — one branch per scenario, nested lapply over matching Q
  # alternatives and batch sizes. Returns a nested list of {Q_name, dgp_name,
  # batch_size, values (max_n x sim_reps matrix)}.
  tar_target(
    batch_sizes,
    list(1L, 5L, 10L)
  ),
  tar_target(
    batch_ripr,
    {
      ss <- sim_seqs
      bs <- batch_sizes
      lapply(
        Filter(function(sq) sq$K == ss$K, sim_Q),
        function(sq) {
          ripr_entry <- Filter(
            function(r) r$Q_name == sq$name && r$P_W@n == bs,
            sim_ripr_optimal
          )[[1L]]
          list(
            Q_name = sq$name,
            dgp_name = ss$name,
            batch_size = bs,
            values = run_martingale(
              BatchRIPr(sq$Q, ripr_entry$P_W, ripr_entry$e_ratio, 0.05),
              ss$data
            )
          )
        }
      )
    },
    pattern = cross(sim_seqs, batch_sizes),
    iteration = "list"
  ),

  # -------------------------------------------------------------------------
  # Plots: mean e-process vs step, one plot per DGP.
  # Colour = martingale type, linetype = Point/Dirichlet Q, facets = Q mode.
  # -------------------------------------------------------------------------
  tar_target(
    sim_long,
    build_sim_long(
      universal_inference,
      infimum_martingale,
      ripr_sequence,
      batch_ripr
    )
  ),

  tar_target(sim_plot_754, plot_mean_martingales(sim_long, "point_754", 3L)),
  tar_target(sim_plot_855, plot_mean_martingales(sim_long, "point_855", 3L)),
  tar_target(sim_plot_7531, plot_mean_martingales(sim_long, "point_7531", 4L)),
  tar_target(sim_plot_8444, plot_mean_martingales(sim_long, "point_8444", 4L))
)
