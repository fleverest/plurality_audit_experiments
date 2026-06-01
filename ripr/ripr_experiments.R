box::use(
  ripr / mixture[discrete_simplex_mixture, n_categories],
  ripr /
    plurality_geometry[plurality_face_descriptors],
  ripr / multinomial[make_multinomial_likelihood],
  ripr / ripr_optimiser[run_ripr, fw_gap]
)

#' RIPr optimiser for K-candidate plurality audits (multinomial)
#'
#' Convenience wrapper around [run_ripr()] for the standard plurality audit
#' setting: constructs plurality null boundary face descriptors via
#' [plurality_face_descriptors()] and a multinomial likelihood via
#' [make_multinomial_likelihood()], runs the Frank-Wolfe + EM optimiser, and
#' returns the result as a `discrete_simplex_mixture`.
#'
#' @param n Integer. Total ballot count.
#' @param q A `simplex_mixture` — the numerator distribution Q. `K` is inferred
#'   from `n_categories(q)`.
#' @param max_atoms_added Integer. Maximum atoms added (beyond initialisation).
#'   Default: 50.
#' @param n_seeds Integer. Random Dirichlet seeds per face for the oracle. Default: 200.
#' @param kl_atol Absolute tolerance for EM convergence. Default 1e-12.
#' @param kl_rtol Relative tolerance for EM convergence. Default 1e-6.
#' @param gap_tol Numeric. Outer loop stops when the expected likelihood ratio
#' falls below this threshold (plus 1L). Default: 1e-6.
#' @param em_iters Integer. Maximum EM refinement iterations per Frank-Wolfe
#'   step. EM stops early when the KL decrease drops below
#'   `kl_atol` + `kl_rtol * abs(KL)`. Default: 3.
#' @param verbose Logical. Print per-iteration progress. Default: `TRUE`.
#' @return List with:
#'   - `mixture`: a `discrete_simplex_mixture` with atoms on the null boundary.
#'   - `history`: list of per-iteration info (`theta_stars`, `E_ratio`, `kl`).
#'   - `converged`: `TRUE` if the validity condition was met.
#' @export
run_plurality_ripr <- function(
  n,
  q,
  max_atoms = NULL,
  n_seeds = 200L,
  em_iters = 3L,
  kl_atol = 1e-12,
  kl_rtol = 1e-6,
  gap_tol = 1e-6,
  verbose = TRUE
) {
  K <- n_categories(q)
  face_descriptors <- plurality_face_descriptors(K)
  likelihood <- make_multinomial_likelihood(n, K)

  result <- run_ripr(
    face_descriptors = face_descriptors,
    likelihood = likelihood,
    q = q,
    max_atoms = max_atoms,
    n_seeds = n_seeds,
    em_iters = em_iters,
    kl_atol = kl_atol,
    kl_rtol = kl_rtol,
    gap_tol = gap_tol,
    verbose = verbose
  )

  list(
    mixture = discrete_simplex_mixture(
      atoms = do.call(cbind, result$atoms),
      weights = result$weights,
      n = n
    ),
    history = result$outer_history,
    kl_trace = result$kl_trace,
    E_star = result$E_star,
    theta_star = result$theta_star,
    converged = result$converged
  )
}
