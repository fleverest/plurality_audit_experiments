library(targets)
library(crew.cluster)

box::use(
  ripr / mixture[point_mnom, dirichlet_mnom],
  ripr / grids[simplex_lattice],
  ripr / ripr_colgen_boundary[run_boundary_ripr]
)

# Second IP address is reachable from other compute nodes on M3
is_m3 <- grepl("m3", Sys.info()[["nodename"]])
gpu_controller <- if (!is_m3) NULL else crew_controller_slurm(
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

tar_option_set(
  seed = 20260516L,
  controller = if (is_m3) gpu_controller else NULL
)


# Define the alternatives of interest.
simplex_k3_alt <- simplex_lattice(3, 50L) / 50L
simplex_k3_alt <- t(simplex_k3_alt[simplex_k3_alt[, 1L] > simplex_k3_alt[, 2L] & simplex_k3_alt[, 1L] > simplex_k3_alt[, 3L], ])
simplex_k4_alt <- simplex_lattice(4, 25L) / 25L
simplex_k4_alt <- t(simplex_k4_alt[simplex_k4_alt[, 1L] > simplex_k4_alt[, 2L] & simplex_k4_alt[, 1L] > simplex_k4_alt[, 3L] & simplex_k4_alt[, 1L] > simplex_k4_alt[, 4L], ])

list(
  tar_target(
    alternative,
    list(
      point_754 = point_mnom(q = c(7 / 16, 5 / 16, 4 / 16)),
      point_844 = point_mnom(q = c(8 / 16, 4 / 16, 4 / 16)),
      dirichlet_754 = dirichlet_mnom(alpha = c(7, 5, 4), atoms = simplex_k3_alt),
      dirichlet_844 = dirichlet_mnom(alpha = c(8, 4, 4), atoms = simplex_k3_alt),
      point_7531 = point_mnom(q = c(7 / 16, 5 / 16, 3 / 16, 1 / 16)),
      point_8222 = point_mnom(q = c(8 / 16, 2 / 16, 2 / 16, 2 / 16)),
      dirichlet_7531 = dirichlet_mnom(alpha = c(7, 5, 3, 1), atoms = simplex_k4_alt),
      dirichlet_8222 = dirichlet_mnom(alpha = c(8, 2, 2, 2), atoms = simplex_k4_alt)
    ),
    iteration = "list"
  ),
  tar_target(ns, seq_len(50L)),
  tar_target(
    ripr_boundary,
    run_boundary_ripr(n = ns, q = alternative),
    pattern   = cross(alternative, ns),
    iteration = "list"
  )
)
