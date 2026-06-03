# Face descriptors for the K-candidate plurality null hypothesis boundary.
#
# The null hypothesis H_0 = { theta in Delta^{K-1} : theta_1 <= theta_j for
# some j != 1 } has a boundary partitioned into K-1 faces. Face j is the set
# where theta_1 = theta_j >= theta_k for all other k. Each face is a convex
# polytope whose 2^{K-2} vertices correspond to subsets S of {2,...,K}\{j}:
# the vertex for S has theta_1 = theta_j = theta_k = 1/(|S|+2) for k in S,
# and zero elsewhere. The empty subset gives the pure-pair vertex
# (theta_1 = theta_j = 1/2).

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
face_vertices <- function(j, K) {
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

#' Face descriptors for the K-candidate plurality null boundary
#'
#' Constructs a list of face descriptors — one per face of the plurality null
#' hypothesis boundary — for use with [run_ripr()]. Each descriptor packages
#' the geometry of one face (the set where candidate `j` ties with the winner)
#' as a collection of closures.
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
#'   \item{`n_vertices`}{Integer. Number of vertices (`2^{K-2}`).}
#'   \item{`init_point(q)`}{Returns an initial atom on this face by projecting
#'     the point `q` onto the face via [project_to_face()].}
#'   \item{`face_index`}{Integer `j`: index of the competing candidate.}
#' }
#'
#' @param K Integer. Number of candidates. Must be >= 2.
#' @return List of `K-1` face descriptors, indexed by competing candidate
#'   `j = 2, ..., K`.
#' @seealso [run_ripr()], [project_to_face()], [face_vertices()]
#' @export
plurality_face_descriptors <- function(K) {
  lapply(2L:K, function(j) {
    V <- face_vertices(j, K)
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
