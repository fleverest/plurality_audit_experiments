box::use(
  ripr / mixture[discrete_simplex_mixture, truncated_dirichlet]
)

# Experiment design: the Q alternatives, the (n, Q) grid, and the
# FW-variant / EM configs swept by the algorithm-comparison experiment.

#' The 8 numerator alternatives Q used throughout the comparison: a point
#' mass and a truncated Dirichlet at K = 3..6, both skewed toward candidate 1.
#' @export
build_Q_list <- function() {
  list(
    list(
      name = "point_321",
      K = 3L,
      Q = discrete_simplex_mixture(as.matrix(3:1 / sum(3:1)), 1)
    ),
    list(
      name = "dirichlet_321",
      K = 3L,
      Q = truncated_dirichlet(3:1)
    ),
    list(
      name = "point_4321",
      K = 4L,
      Q = discrete_simplex_mixture(
        as.matrix(4:1 / sum(4:1)),
        1
      )
    ),
    list(
      name = "dirichlet_4321",
      K = 4L,
      Q = truncated_dirichlet(4:1)
    ),
    list(
      name = "point_54321",
      K = 5L,
      Q = discrete_simplex_mixture(
        as.matrix(5:1 / sum(5:1)),
        1
      )
    ),
    list(
      name = "dirichlet_54321",
      K = 5L,
      Q = truncated_dirichlet(5:1)
    ),
    list(
      name = "point_654321",
      K = 6L,
      Q = discrete_simplex_mixture(
        as.matrix(6:1 / sum(6:1)),
        1
      )
    ),
    list(
      name = "dirichlet_654321",
      K = 6L,
      Q = truncated_dirichlet(6:1)
    )
  )
}

#' Cross batch sizes `n_values` with `Q_list`, dropping (n=50, K>=6): the
#' multinomial support size M = C(n+K-1,K-1) is already ~3.5M outcomes at
#' n=50,K=6 (and grows combinatorially from there), well past what fits in a
#' single worker's memory budget.
#' @export
build_nQ <- function(n_values, Q_list) {
  combos <- lapply(n_values, function(n) {
    lapply(Q_list, function(q_info) {
      if (n == 50L && q_info$K >= 6L) {
        NULL
      } else {
        c(list(n = n), q_info)
      }
    })
  })
  Filter(Negate(is.null), do.call(c, combos))
}

#' The full config sweep: `maxiters` random-init EM-only runs, one pure-FW
#' run per variant (pairwise / line-search / vanilla / fully-corrective), and
#' one FW-variant-plus-EM hybrid run each for line-search / pairwise /
#' fully-corrective.
#' @export
build_configs <- function(maxiters) {
  c(
    lapply(seq_len(maxiters), function(l) {
      list(
        name = "em_only",
        fw_iters = 0L,
        em_iters = 2000L,
        init = l,
        kl_atol = 1e-12,
        kl_rtol = 1e-10,
        verbose = TRUE
      )
    }),
    lapply(
      c("pairwise", "line-search", "vanilla", "fully-corrective"),
      function(v) {
        list(
          name = paste0("fw_", v),
          fw_iters = maxiters,
          em_iters = 0L,
          fw_variant = v,
          gap_tol = 0,
          init = NULL,
          verbose = TRUE
        )
      }
    ),
    list(list(
      name = "hybrid_line-search",
      fw_iters = maxiters,
      em_iters = 10L,
      fw_variant = "line-search",
      gap_tol = 0,
      init = NULL,
      verbose = TRUE
    )),
    list(list(
      name = "hybrid_pairwise",
      fw_iters = maxiters,
      em_iters = 10L,
      fw_variant = "pairwise",
      gap_tol = 0,
      init = NULL,
      verbose = TRUE
    )),
    list(list(
      name = "hybrid_fully-corrective",
      fw_iters = maxiters,
      em_iters = 10L,
      fw_variant = "fully-corrective",
      gap_tol = 0,
      init = NULL,
      verbose = TRUE
    ))
  )
}
