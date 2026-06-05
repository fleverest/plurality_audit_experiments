box::use(
  stats[rgamma],
  ripr / mixture[discrete_simplex_mixture, n_categories],
  ripr /
    plurality_geometry[plurality_face_descriptors],
  ripr / multinomial[make_multinomial_likelihood],
  ripr / ripr_optimiser[run_ripr, fw_gap]
)

#' Sample random atoms on the null boundary faces via uniform Dirichlet draws.
#'
#' Atoms are distributed cyclically across faces (first atom on face 1, second
#' on face 2, ..., wrapping around). Useful for constructing `init_atoms` /
#' `init_atom_faces` to pass to [run_ripr()], e.g. for a pure-EM experiment.
#'
#' @param face_descriptors List of face descriptor closures.
#' @param n_atoms Integer. Total number of atoms to generate.
#' @return Named list with:
#'   - `atoms`: K × n_atoms numeric matrix (each column is one atom).
#'   - `faces`: integer vector of length n_atoms with 1-based face indices.
#' @export
sample_dirichlet_atoms <- function(face_descriptors, n_atoms) {
  n_faces <- length(face_descriptors)
  face_idx <- ((seq_len(n_atoms) - 1L) %% n_faces) + 1L
  atom_list <- lapply(seq_len(n_atoms), function(i) {
    fd <- face_descriptors[[face_idx[i]]]
    alpha <- rgamma(fd$n_vertices, shape = 1)
    fd$parametrise(alpha / sum(alpha))
  })
  list(atoms = do.call(cbind, atom_list), faces = face_idx)
}

#' RIPr optimiser for K-candidate plurality audits (multinomial)
#'
#' Convenience wrapper around [run_ripr()] for the standard plurality audit
#' setting. Constructs the plurality null boundary and multinomial likelihood,
#' resolves `init` into `init_atoms` / `init_atom_faces`, then delegates to
#' [run_ripr()] via `...`.
#'
#' @param n Integer. Total ballot count.
#' @param q A `simplex_mixture` — the numerator distribution Q. `K` is inferred
#'   from `n_categories(q)`.
#' @param fw_iters Integer. Number of Frank-Wolfe iterations (passed to
#'   [run_ripr()]). Set to 0L for pure EM.
#' @param em_iters Integer. Max EM iterations per outer step (passed to
#'   [run_ripr()]). Set to 0L to skip EM.
#' @param init Controls atom initialisation:
#'   - `NULL` (default): one atom per face projected from q's mode (or mean if
#'     mode is NA).
#'   - A positive integer `l`: generates `l` random uniform-Dirichlet atoms
#'     across faces via [sample_dirichlet_atoms()].
#'   - A named list with `atoms` (K × l matrix) and `faces` (integer vector):
#'     uses these directly.
#' @param ... Further arguments passed to [run_ripr()] (e.g. `fw_variant`,
#'   `gap_tol`, `verbose`, …).
#' @return List with:
#'   - `mixture`: a `discrete_simplex_mixture` with atoms on the null boundary.
#'   - `history`: the `outer_history` data.frame from [run_ripr()].
#'   - `kl_trace`: the per-step KL trace data.frame from [run_ripr()].
#'   - `E_star`: terminal FW gap value (max E_theta[Q / P_w]).
#'   - `theta_star`: terminal maximising theta.
#'   - `converged`: `TRUE` if the FW gap fell below `gap_tol`.
#' @export
run_plurality_ripr <- function(n, q, fw_iters, em_iters, init = NULL, ...) {
  K <- n_categories(q)
  face_descriptors <- plurality_face_descriptors(K)
  likelihood <- make_multinomial_likelihood(n, K)

  if (is.null(init)) {
    q_ref <- if (anyNA(q@mode)) q@mean else q@mode
    init_data <- list(
      atoms = do.call(cbind, lapply(face_descriptors, function(fd) fd$init_point(q_ref))),
      faces = seq_along(face_descriptors)
    )
  } else if (is.numeric(init) && length(init) == 1L) {
    init_data <- sample_dirichlet_atoms(face_descriptors, as.integer(init))
  } else if (is.list(init)) {
    init_data <- init
  } else {
    stop("`init` must be NULL, a positive integer, or a list with `atoms` and `faces`.")
  }

  result <- run_ripr(
    face_descriptors = face_descriptors,
    likelihood = likelihood,
    q = q,
    init_atoms = init_data$atoms,
    init_atom_faces = init_data$faces,
    fw_iters = fw_iters,
    em_iters = em_iters,
    ...
  )

  list(
    mixture = discrete_simplex_mixture(
      atoms = do.call(cbind, result$atoms),
      weights = result$weights,
      n = n
    ),
    history = result$history,
    kl_trace = result$kl_trace,
    gap = result$gap,
    oracle_theta = result$oracle_theta,
    converged = result$converged
  )
}
