# Face descriptors for the K-candidate plurality null hypothesis boundary.
#
# The null hypothesis H_0 = { theta in Delta^{K-1} : theta_1 <= theta_j for
# some j != 1 } is a union of K-1 pieces, one per competing candidate j. This
# file provides three alternative geometric descriptions of those pieces, all
# built on the same `parametrise`/`jacobian`/`pinv`/`init_point` interface
# expected by [run_ripr()]:
#
#   * `boundary_face_descriptors()` (the reduced/exposed boundary): face
#     j is the exposed boundary piece where theta_1 = theta_j >= theta_k for
#     all other k. This is the minimal set needed for KL-projection, since the
#     projection of any point onto H_0 lands where the winning tie is also the
#     runner-up among all other candidates.
#   * `full_plurality_face_descriptors()` (the literal half-space union): face
#     j is the full closed half-space { theta_j >= theta_1 }, with no
#     restriction relative to the other candidates. The union over j of these
#     half-spaces is exactly H_0 itself, redundancy included.
#   * `tie_simplex_face_descriptors()` (the union of tie facets): face j is
#     the full simplex facet { theta_1 = theta_j }, i.e. the hyperplane slice
#     without the runner-up restriction. The union over j of these facets is
#     the set of all ties involving candidate 1, a strict subset of H_0.
#
# Each face is a convex polytope; `build_face_descriptors()` turns any vertex
# matrix into the descriptor list via SVD-based pseudo-inversion.

#' Project a point onto a plurality null boundary face
#'
#' Moves from `q` towards the standard basis vector `e_j` along a straight
#' line until `theta_1 = theta_j`, landing on face `j` of the plurality null
#' boundary. Used to initialise the first atom on each face.
#'
#' @param j Integer. Index of the face (competitor candidate), in `2:K`.
#' @param q Numeric vector of length `K` summing to 1, with `q[1] > q[j]`
#'   (i.e. `q` is strictly inside H_1).
#' @return Numeric vector of length `K` on face `j`.
project_to_face <- function(j, q) {
  lambda <- (q[1L] - q[j]) / (1 + q[1L] - q[j])
  theta <- (1 - lambda) * q
  theta[j] <- theta[j] + lambda
  theta
}

#' Enumerate the vertices of a plurality null boundary face
#'
#' Returns all `2^{K-2}` vertices of face `j` as columns of a `K x 2^{K-2}`
#' matrix. Each vertex corresponds to a subset `S` of `{2,...,K} \ {j}`: the
#' vertex has `theta_1 = theta_j = theta_k = 1/(|S|+2)` for `k` in `S`, and
#' zero elsewhere.
#'
#' @param j Integer. Face index (competing candidate), in `2:K`.
#' @param K Integer. Total number of candidates.
#' @return Numeric matrix of dimension `K x 2^{K-2}`. Columns are the vertex
#'   vectors; each column sums to 1.
boundary_face_vertices <- function(j, K) {
  others <- setdiff(seq_len(K)[-1L], j)
  n_others <- length(others)
  subsets <- lapply(0L:(2L^n_others - 1L), function(mask) {
    others[as.logical(intToBits(mask)[seq_len(n_others)])]
  })
  vertices <- lapply(subsets, function(S) {
    v <- numeric(K)
    v[c(1L, j, S)] <- 1 / (length(S) + 2L)
    v
  })
  do.call(cbind, vertices)
}

#' Enumerate the vertices of a full plurality half-space face
#'
#' Returns the `K` vertices of the closed half-space `{theta_j >= theta_1}`
#' as columns of a `K x K` matrix: the standard basis vector `e_k` for every
#' `k != 1` (mass entirely on candidate `k`), plus the tie vertex
#' `(e_1 + e_j) / 2`.
#'
#' @param j Integer. Face index (competing candidate), in `2:K`.
#' @param K Integer. Total number of candidates.
#' @return Numeric matrix of dimension `K x K`. Columns are the vertex
#'   vectors; each column sums to 1.
full_plurality_face_vertices <- function(j, K) {
  basis_vertices <- lapply(setdiff(seq_len(K), 1L), function(k) {
    v <- numeric(K)
    v[k] <- 1
    v
  })
  tie <- numeric(K)
  tie[c(1L, j)] <- 0.5
  do.call(cbind, c(basis_vertices, list(tie)))
}

#' Enumerate the vertices of a plurality tie-facet face
#'
#' Returns the `K-1` vertices of the full simplex facet `{theta_1 = theta_j}`
#' as columns of a `K x (K-1)` matrix: the tie vertex `(e_1 + e_j) / 2`, plus
#' the standard basis vector `e_k` for every `k` not in `{1, j}`.
#'
#' @param j Integer. Face index (competing candidate), in `2:K`.
#' @param K Integer. Total number of candidates.
#' @return Numeric matrix of dimension `K x (K-1)`. Columns are the vertex
#'   vectors; each column sums to 1.
tie_simplex_face_vertices <- function(j, K) {
  tie <- numeric(K)
  tie[c(1L, j)] <- 0.5
  basis_vertices <- lapply(setdiff(seq_len(K), c(1L, j)), function(k) {
    v <- numeric(K)
    v[k] <- 1
    v
  })
  do.call(cbind, c(list(tie), basis_vertices))
}

#' Build face descriptors from a per-face vertex enumerator
#'
#' Shared machinery behind [boundary_face_descriptors()],
#' [full_plurality_face_descriptors()] and [tie_simplex_face_descriptors()]:
#' turns a `vertex_fn(j, K)` (a `K x n_vertices` matrix of face vertices) into
#' the list of closures expected by [run_ripr()].
#'
#' A face descriptor is a list with components:
#' \describe{
#'   \item{`parametrise(alpha)`}{Maps `alpha` in `Delta^{n_vertices-1}` to a
#'     point `theta` on the face via the convex combination of vertices.}
#'   \item{`parametrise_batch(alpha_mat)`}{Batched version: `alpha_mat` is an
#'     `N x n_vertices` matrix; returns a `K x N` matrix of theta vectors.}
#'   \item{`jacobian`}{The `K x n_vertices` vertex matrix, i.e the
#'     constant Jacobian of `parametrise` with respect to `alpha`.}
#'   \item{`pinv`}{The `n_vertices x K` left pseudo-inverse of the Jacobian, for
#'    mapping gradients in `theta` space back to `alpha` space.}
#'   \item{`n_vertices`}{Integer. Number of vertices of the face.}
#'   \item{`init_point(q)`}{Returns an initial atom on this face by projecting
#'     the point `q` onto the face via [project_to_face()].}
#'   \item{`face_index`}{Integer `j`: index of the competing candidate.}
#' }
#'
#' @param K Integer. Number of candidates. Must be >= 2.
#' @param vertex_fn Function of `(j, K)` returning a `K x n_vertices` matrix
#'   of face-vertex columns summing to 1.
#' @return List of `K-1` face descriptors, indexed by competing candidate
#'   `j = 2, ..., K`.
#' @seealso [run_ripr()], [project_to_face()]
build_face_descriptors <- function(K, vertex_fn) {
  lapply(2L:K, function(j) {
    V <- vertex_fn(j, K)
    svd_V <- svd(V)
    tol <- max(dim(V)) * .Machine$double.eps * max(svd_V$d)
    pos <- svd_V$d > tol
    V_pinv <- svd_V$v[, pos, drop = FALSE] %*%
      (t(svd_V$u[, pos, drop = FALSE]) / svd_V$d[pos])
    list(
      parametrise = function(alpha) as.vector(V %*% alpha),
      parametrise_batch = function(alpha_mat) V %*% t(alpha_mat),
      jacobian = V,
      pinv = V_pinv,
      n_vertices = ncol(V),
      init_point = function(q) project_to_face(j, q),
      face_index = j
    )
  })
}

#' Face descriptors for the K-candidate plurality null boundary (reduced)
#'
#' Constructs a list of face descriptors — one per face of the plurality null
#' hypothesis boundary — for use with [run_ripr()]. Face `j` is the exposed
#' boundary piece where candidate `j` ties with candidate 1 and dominates
#' every other candidate: `theta_1 = theta_j >= theta_k` for all other `k`.
#' This is the minimal (non-redundant) description of the H_0 boundary; see
#' [full_plurality_face_descriptors()] and [tie_simplex_face_descriptors()]
#' for two alternative, redundant descriptions of the same null hypothesis.
#'
#' @param K Integer. Number of candidates. Must be >= 2.
#' @return List of `K-1` face descriptors, indexed by competing candidate
#'   `j = 2, ..., K`. See [build_face_descriptors()] for the descriptor
#'   structure.
#' @seealso [run_ripr()], [boundary_face_vertices()], [build_face_descriptors()]
#' @export
boundary_face_descriptors <- function(K) {
  build_face_descriptors(K, boundary_face_vertices)
}

#' Face descriptors for the full plurality null half-space union
#'
#' Constructs a list of face descriptors describing H_0 as the literal union
#' of half-spaces `bigcup_{j != 1} {theta_j >= theta_1}`. Face `j` is the
#' full closed half-space `{theta_j >= theta_1}`, with no restriction on the
#' other candidates — unlike [boundary_face_descriptors()], candidate `j`
#' need not dominate the remaining candidates on this face. The union over
#' `j` still equals H_0 exactly, but each individual face is larger (and the
#' faces overlap), so the FW oracle searches a bigger region per face.
#'
#' @param K Integer. Number of candidates. Must be >= 2.
#' @return List of `K-1` face descriptors, indexed by competing candidate
#'   `j = 2, ..., K`. See [build_face_descriptors()] for the descriptor
#'   structure.
#' @seealso [run_ripr()], [full_plurality_face_vertices()],
#'   [build_face_descriptors()]
#' @export
full_plurality_face_descriptors <- function(K) {
  build_face_descriptors(K, full_plurality_face_vertices)
}

#' Face descriptors for the union of pairwise tie facets
#'
#' Constructs a list of face descriptors describing the (strict) subset of
#' H_0's boundary given by `bigcup_{j != 1} {theta_1 = theta_j}`: the union
#' of full simplex facets where candidate `j` ties with candidate 1. Face `j`
#' is the entire facet `{theta_1 = theta_j}`, with no restriction that `j`
#' dominate the other candidates (unlike [boundary_face_descriptors()], which
#' further restricts to the exposed sub-region of this facet).
#'
#' @param K Integer. Number of candidates. Must be >= 2.
#' @return List of `K-1` face descriptors, indexed by competing candidate
#'   `j = 2, ..., K`. See [build_face_descriptors()] for the descriptor
#'   structure.
#' @seealso [run_ripr()], [tie_simplex_face_vertices()],
#'   [build_face_descriptors()]
#' @export
tie_simplex_face_descriptors <- function(K) {
  build_face_descriptors(K, tie_simplex_face_vertices)
}
