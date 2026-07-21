box::use(
  ripr /
    mixture[
      discrete_simplex_mixture,
      truncated_dirichlet,
      pairwise_projection,
      expected_likelihood_ratio,
      log_pmf
    ],
  ripr / multinomial[make_multinomial_likelihood],
  ripr /
    pairwise_projection[relaxed_pairwise_log_e, pairwise_log_pstar]
)

#' Pairwise-null baselines: closed-form RIPr onto each pairwise null
#' H_0^(j) = {theta_1 <= theta_j}, combined via min over j. See
#' ripr/pairwise_projection.R for the identities.
#'
#' Checks, per (K, n, j): pushforward exactness (analytic log_pmf agrees with
#' the pi_j-pushforward discrete mixture), the fast full-support path agrees
#' with the class implementation, the relaxed 1-D e-value and the discrete
#' RIPr e-value are both valid on the tie facet (ELR = 1) and inside the null
#' (ELR <= 1), and the point-mass case collapses to the closed-form BRAVO
#' wager.
#' @export
run_pairwise_validation <- function() {
  check <- function(cond, msg) {
    if (!all(cond)) {
      stop("pairwise_validation failed: ", msg)
    }
  }
  pi_j <- function(theta, j) {
    theta[c(1L, j)] <- (theta[1L] + theta[j]) / 2
    theta
  }
  # Base simplex points from which tie-facet / interior-null test points
  # are derived per (K, j).
  base_points <- function(K) {
    skewed <- exp(seq(1.5, 0, length.out = K))
    cbind(
      rep(1 / K, K),
      (K:1) / sum(K:1),
      (1:K) / sum(1:K),
      skewed / sum(skewed)
    )
  }

  rows <- list()
  for (setting in list(list(K = 3L, n = 6L), list(K = 4L, n = 5L))) {
    K <- setting$K
    n <- setting$n
    likelihood <- make_multinomial_likelihood(n, K)
    support <- likelihood$support_tensor

    atoms <- cbind(
      c(0.5, rep(0.5 / (K - 1), K - 1L)),
      (2 * K:1 + 1) / sum(2 * K:1 + 1)
    )
    wts <- c(0.6, 0.4)
    numerators <- list(
      disc = discrete_simplex_mixture(atoms, wts),
      dir = truncated_dirichlet(seq(K + 1, 2, length.out = K)),
      point = discrete_simplex_mixture(atoms[, 2L, drop = FALSE], 1)
    )

    for (j in 2:K) {
      # Exactness: analytic log_pmf == pi_j-pushforward discrete mixture.
      lp_disc <- as.numeric(log_pmf(
        pairwise_projection(numerators$disc, j, n),
        support
      )$cpu())
      lp_ref <- as.numeric(log_pmf(
        discrete_simplex_mixture(apply(atoms, 2L, pi_j, j = j), wts),
        support
      )$cpu())
      err_exact <- max(abs(lp_disc - lp_ref))
      check(err_exact < 1e-9, "pushforward exactness (discrete Q)")

      # Fast full-support path agrees with the class implementation.
      X_sup <- as.matrix(as.array(support$cpu()))
      log_q_disc <- as.numeric(log_pmf(numerators$disc, support)$cpu())
      check(
        max(abs(pairwise_log_pstar(log_q_disc, X_sup, j) - lp_disc)) <
          1e-9,
        "pairwise_log_pstar equivalence"
      )

      # Tie-facet and interior-null test points.
      bp <- base_points(K)
      ties <- apply(bp, 2L, pi_j, j = j)
      interior <- apply(bp, 2L, function(b) {
        s <- b[1L] + b[j]
        b[1L] <- 0.3 * s
        b[j] <- 0.7 * s
        b
      })

      errs <- vapply(
        names(numerators),
        function(nm) {
          Q <- numerators[[nm]]
          P <- pairwise_projection(Q, j, n)
          lp <- as.numeric(log_pmf(P, support)$cpu())
          check(
            abs(sum(exp(lp)) - 1) < 1e-8,
            paste("normalisation, Q =", nm)
          )
          # Theorem: E_phi[Q/P*] = 1 on the tie facet, <= 1 inside.
          elr_tie <- expected_likelihood_ratio(ties, n, Q, P)
          elr_int <- expected_likelihood_ratio(interior, n, Q, P)
          check(
            abs(elr_tie - 1) < 1e-6,
            paste("ELR = 1 on tie facet, Q =", nm)
          )
          check(
            elr_int <= 1 + 1e-8,
            paste("ELR <= 1 inside null, Q =", nm)
          )
          c(max(abs(elr_tie - 1)), max(elr_int) - 1)
        },
        numeric(2L)
      )

      # Point-mass collapse: e-value reduces to the BRAVO wager
      # 2^N p*^x1 (1-p*)^(N-x1) with p* = theta*_1 / (theta*_1 + theta*_j).
      theta_star <- atoms[, 2L]
      p_star <- theta_star[1L] / (theta_star[1L] + theta_star[j])
      X_cpu <- as.matrix(as.array(support$cpu()))
      x1 <- X_cpu[, 1L]
      N <- x1 + X_cpu[, j]
      log_e_point <- as.numeric(log_pmf(
        numerators$point,
        support
      )$cpu()) -
        as.numeric(log_pmf(
          pairwise_projection(numerators$point, j, n),
          support
        )$cpu())
      log_e_bravo <- N * log(2) + x1 * log(p_star) + (N - x1) * log(1 - p_star)
      err_bravo <- max(abs(log_e_point - log_e_bravo))
      check(err_bravo < 1e-9, "point-mass BRAVO collapse")

      # Relaxed 1-D e-values are valid: sum_x P_phi(x) E_j(x) = 1 on
      # ties, <= 1 inside the null.
      log_e_beta <- relaxed_pairwise_log_e(X_cpu, numerators$dir@alpha, j)
      elr_beta <- function(phi_mat) {
        apply(phi_mat, 2L, function(phi) {
          lp_phi <- as.numeric(likelihood$log_pmf(phi)$cpu())
          sum(exp(lp_phi + log_e_beta))
        })
      }
      check(
        abs(elr_beta(ties) - 1) < 1e-6,
        "relaxed e-value: ELR = 1 on tie facet"
      )
      check(
        elr_beta(interior) <= 1 + 1e-8,
        "relaxed e-value: ELR <= 1 inside null"
      )

      rows[[length(rows) + 1L]] <- data.frame(
        K = K,
        n = n,
        j = j,
        err_pushforward = err_exact,
        err_elr_tie = max(errs[1L, ]),
        max_elr_interior_excess = max(errs[2L, ]),
        err_bravo = err_bravo
      )
    }
  }
  do.call(rbind, rows)
}
