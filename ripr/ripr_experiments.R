box::use(
  ripr / mixture[mixture_mnom],
  ripr /
    plurality_geometry[plurality_face_descriptors],
  ripr / multinomial_likelihood[make_multinomial_likelihood],
  ripr / ripr_optimiser[run_ripr]
)

#' RIPr optimiser for K-candidate plurality audits (multinomial)
#'
#' Convenience wrapper around [run_ripr()] for the standard plurality audit
#' setting: constructs plurality null boundary face descriptors via
#' [plurality_face_descriptors()] and a multinomial likelihood via
#' [make_multinomial_likelihood()], runs the Frank-Wolfe + EM optimiser, and
#' returns the result as a `mixture_mnom`.
#'
#' @param n Integer. Total ballot count.
#' @param q A `mixture_mnom` — the numerator distribution Q. Use [point_mnom()]
#'   for a single point alternative or [dirichlet_mnom()] for a grid-weighted
#'   Dirichlet prior. `K` is inferred from `nrow(q@atoms)`.
#' @param max_atoms_added Integer. Maximum atoms added (beyond initialisation).
#'   Default: 50.
#' @param oracle_grid Integer. Grid density for the per-face oracle. Total
#'   lattice points per face ~ `oracle_grid^(n_vertices - 1)`. Default: 200.
#' @param n_em_iter Integer. Maximum EM refinement iterations per Frank-Wolfe
#'   step. EM stops early when the KL decrease drops below `tol`. Default: 3.
#' @param verbose Logical. Print per-iteration progress. Default: `TRUE`.
#' @return List with:
#'   - `mixture`: a `mixture_mnom` with atoms on the null boundary.
#'   - `history`: list of per-iteration info (`theta_stars`, `E_ratio`, `kl`).
#'   - `converged`: `TRUE` if the validity condition was met.
#' @export
run_plurality_ripr <- function(
  n,
  q,
  max_atoms = 50L,
  oracle_grid = 200L,
  n_em_iter = 3L,
  verbose = TRUE
) {
  K <- nrow(q@atoms)
  face_descriptors <- plurality_face_descriptors(K)
  likelihood <- make_multinomial_likelihood(n, K)

  result <- run_ripr(
    face_descriptors = face_descriptors,
    likelihood = likelihood,
    q = q,
    max_atoms = max_atoms,
    oracle_grid = oracle_grid,
    n_em_iter = n_em_iter,
    verbose = verbose
  )

  list(
    mixture = mixture_mnom(
      atoms = do.call(cbind, result$atoms),
      weights = result$weights,
      n = n
    ),
    history = result$outer_history,
    kl_trace = result$kl_trace,
    E_star = result$E_star,
    converged = result$converged
  )
}
