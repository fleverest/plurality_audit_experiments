box::use(
  ripr / mixture[mixture_mnom],
  ripr /
    plurality_geometry[
      plurality_face_descriptors,
      plurality_face_descriptors_unrestricted
    ],
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
#' @param atoms_per_face Integer. Maximum atoms added per boundary face.
#'   Default: 50.
#' @param oracle_grid Integer. Grid density for the per-face oracle. Total
#'   lattice points per face ~ `oracle_grid^(n_vertices - 1)`. Default: 200.
#' @param reweight_maxit Integer. Maximum mirror descent iterations for the
#'   weight reweighting step. Default: 1000.
#' @param n_em_iter Integer. Maximum EM refinement iterations per Frank-Wolfe
#'   step. EM stops early when the KL decrease drops below `tol`. Default: 3.
#' @param tol Numeric. Convergence tolerance: stop when
#'   `max_theta E_theta[Q/P_W] <= 1 + tol`. Default: `1e-4`.
#' @param verbose Logical. Print per-iteration progress. Default: `TRUE`.
#' @return List with:
#'   - `mixture`: a `mixture_mnom` with atoms on the null boundary.
#'   - `history`: list of per-iteration info (`theta_stars`, `E_ratio`, `kl`).
#'   - `converged`: `TRUE` if the validity condition was met.
#' @export
run_plurality_ripr <- function(
  n,
  q,
  atoms_per_face = 50L,
  oracle_grid = 200L,
  reweight_maxit = 1000L,
  n_em_iter = 3L,
  tol = 1e-4,
  verbose = TRUE,
  boundary = TRUE
) {
  K <- nrow(q@atoms)
  if (boundary) {
    face_descriptors <- plurality_face_descriptors(K)
  } else {
    face_descriptors <- plurality_face_descriptors_unrestricted(K)
  }
  likelihood <- make_multinomial_likelihood(n, K)

  result <- run_ripr(
    face_descriptors = face_descriptors,
    likelihood = likelihood,
    q = q,
    atoms_per_face = atoms_per_face,
    oracle_grid = oracle_grid,
    reweight_maxit = reweight_maxit,
    n_em_iter = n_em_iter,
    tol = tol,
    verbose = verbose
  )

  list(
    mixture = mixture_mnom(
      atoms = do.call(cbind, result$atoms),
      weights = result$weights,
      n = n
    ),
    history = result$history,
    converged = result$converged
  )
}
