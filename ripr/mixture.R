box::use(
  S7[new_class, new_generic, method, class_numeric, class_integer],
  torch[torch_tensor, torch_logsumexp],
  ripr / torch_settings[device, dtype],
  ripr / multinomial[mnom_logpmf]
)

#' Finite mixture of multinomial distributions
#'
#' Represents Q = sum_j w_j * Multinomial(theta_j) as a pair of atoms and
#' weights. Used both as the alternative hypothesis Q and as the null boundary
#' mixture P_w returned by the RIPr optimiser.
#'
#' @slot atoms Numeric matrix of shape (K, M) — columns are probability vectors.
#' @slot weights Numeric vector of length M summing to 1.
mixture_mnom <- new_class(
  "mixture_mnom",
  properties = list(
    atoms   = class_numeric,
    weights = class_numeric
  ),
  validator = function(self) {
    if (!is.matrix(self@atoms))
      return("`atoms` must be a matrix")
    if (ncol(self@atoms) != length(self@weights))
      return("`weights` must have one entry per column of `atoms`")
    if (abs(sum(self@weights) - 1) > 1e-9)
      return("`weights` must sum to 1")
    NULL
  }
)

#' Log PMF of a mixture_mnom for all multinomial outcomes of size n
#'
#' @param mixture A `mixture_mnom` object.
#' @param X Tensor of shape (N, K), e.g. count vectors from [mnom_logpmf()].
#' @param n Total number of trials.
#' @return Tensor of shape (N,) with log Q(x) for each row x of X.
log_pmf <- new_generic("log_pmf", "mixture")

method(log_pmf, mixture_mnom) <- function(mixture, X, n) {
  log_atoms_t <- torch_tensor(log(mixture@atoms), device = device, dtype = dtype)
  log_wts     <- torch_tensor(log(mixture@weights), device = device, dtype = dtype)
  # (N, M) + (M,) broadcast → logsumexp over M → (N,)
  torch_logsumexp(mnom_logpmf(X, log_atoms_t, n) + log_wts, dim = 2L)
}

#' Construct a point-mass mixture_mnom
#'
#' @param q Numeric vector of length K summing to 1.
#' @return A `mixture_mnom` with a single atom.
#' @export
point_mnom <- function(q) {
  mixture_mnom(atoms = matrix(q, ncol = 1L), weights = 1)
}

#' Construct a mixture_mnom from a grid of atoms with Dirichlet weights
#'
#' Evaluates the Dirichlet(alpha) density at each column of `atoms` and
#' normalises to produce mixture weights. Atoms with zero probability under
#' the prior (any component = 0) are assigned zero weight.
#'
#' @param alpha Numeric vector of the K Dirichlet concentration parameters.
#' @param atoms Numeric matrix of shape (K, M) of grid points in the simplex
#'   (e.g. points in H_1).
#' @return A `mixture_mnom` with an approximate Dirichlet density over the given atoms.
#' @export
dirichlet_mnom <- function(alpha, atoms) {
  # log Dir(alpha)(theta) up to the shared normalising constant
  log_unnorm <- colSums((alpha - 1) * log(atoms))
  log_unnorm[!is.finite(log_unnorm)] <- -Inf
  # normalise in log space for numerical stability
  log_weights <- log_unnorm - log(sum(exp(log_unnorm - max(log_unnorm[is.finite(log_unnorm)]))))
  mixture_mnom(atoms = atoms, weights = exp(log_weights))
}
