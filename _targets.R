library(targets)
library(crew.cluster)

box::use(
  ripr / mixture[point_mnom, dirichlet_mnom],
  ripr / grids[simplex_lattice],
  ripr / ripr_colgen_boundary[run_boundary_ripr]
)

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
  controller = if (is_m3) gpu_controller else NULL
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
    alternative,
    list(
      list(name = "point_754", Q = point_mnom(q = c(7 / 16, 5 / 16, 4 / 16))),
      list(name = "point_844", Q = point_mnom(q = c(8 / 16, 4 / 16, 4 / 16))),
      list(
        name = "dirichlet_754",
        Q = dirichlet_mnom(alpha = c(7, 5, 4), atoms = simplex_k3_alt)
      ),
      list(
        name = "dirichlet_844",
        Q = dirichlet_mnom(alpha = c(8, 4, 4), atoms = simplex_k3_alt)
      ),
      list(
        name = "point_7531",
        Q = point_mnom(q = c(7 / 16, 5 / 16, 3 / 16, 1 / 16))
      ),
      list(
        name = "point_8222",
        Q = point_mnom(q = c(8 / 16, 2 / 16, 2 / 16, 2 / 16))
      ),
      list(
        name = "dirichlet_7531",
        Q = dirichlet_mnom(alpha = c(7, 5, 3, 1), atoms = simplex_k4_alt)
      ),
      list(
        name = "dirichlet_8222",
        Q = dirichlet_mnom(alpha = c(8, 2, 2, 2), atoms = simplex_k4_alt)
      )
    )
  ),
  tar_target(n, as.list(seq_len(50L))),
  tar_target(
    ripr_boundary,
    c(
      list(alt_name = alternative[[1L]]$name, n = n[[1L]]),
      run_boundary_ripr(
        n = n[[1L]],
        q = alternative[[1L]]$Q,
        atoms_per_face = 200L
      )
    ),
    pattern = cross(alternative, n),
    iteration = "list"
  ),
  tar_target(
    grand_prix_sims,
    list(
      list(
        name = "p_754",
        replicate(
          10000,
          sample(
            1:3,
            size = 200,
            replace = TRUE,
            prob = c(7 / 16, 5 / 16, 4 / 16)
          )
        )
      ),
      list(
        name = "p_844",
        replicate(
          10000,
          sample(
            1:3,
            size = 200,
            replace = TRUE,
            prob = c(8 / 16, 4 / 16, 4 / 16)
          )
        )
      )
    )
  )
)
