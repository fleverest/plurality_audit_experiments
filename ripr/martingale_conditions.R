# In this file, we load resulting weights for different n and check whether the
# sequence of fixed-sample-size Approximate-GRO (AGRO) e-values yields a
# martingale. We do this by computing the conditional expectation of the next
# e-value given the previous ones, and checking whether it is less than or equal
# to the current e-value.

box::use(
  ripr / multinomial[build_counts_tensor, mnom_logpmf],
  ripr / grids[make_2d_simplex_grid],
  ripr / torch_settings[device, dtype],
  torch[torch_tensor, torch_logsumexp, torch_stack, torch_eye]
)

# Reconstruct the grids used in optimisation (must match smoke_test.R)
q      <- c(7 / 16, 5 / 16, 4 / 16)
thetas <- make_2d_simplex_grid(10001)
ws     <- make_2d_simplex_grid(1001)

to_log_tensor <- function(prob_list) {
  do.call(rbind, prob_list) |>
    torch_tensor(device = device, dtype = dtype) |>
    (\(x) x$transpose(1, 2))() |>
    (\(x) x$log())()
}

log_q     <- torch_tensor(q, device = device, dtype = dtype)$log()
log_theta <- to_log_tensor(thetas)  # (3, T)
log_ws    <- to_log_tensor(ws)      # (3, C)

files <- list.files("results", pattern = "optim_n_.*\\.rds", full.names = TRUE)
ns    <- as.integer(regmatches(files, regexpr("(?<=optim_n_)\\d+", files, perl = TRUE)))
ord   <- order(ns)
files <- files[ord]
ns    <- ns[ord]

results <- lapply(files, readRDS)

weights <- lapply(results, \(f) {
  f$weights_max_exp
})


# Bona fide e-value for a set of count vectors x given log mixture weights log_wts_row (1, C).
# Divides Q(x)/P_w(x) by max_theta E_theta[Q(X)/P_w(X)] so the result genuinely (approximately,
# depends on the grid resolution) represents an e-value.
eval_e <- function(x, n, log_wts_row) {
  x_full    <- build_counts_tensor(n)                                      # (M_full, 3)
  log_Qx_f  <- mnom_logpmf(x_full, log_q$unsqueeze(2L), n)[, 1L]          # (M_full,)
  log_Pw_f  <- torch_logsumexp(mnom_logpmf(x_full, log_ws, n) + log_wts_row, dim = 2L)
  e_full    <- (log_Qx_f - log_Pw_f)$exp()                                # (M_full,)

  # E_theta[e(X)] = sum_x PMF(x; n, theta) * e(x), maximised over theta
  log_pmf_theta <- mnom_logpmf(x_full, log_theta, n)                      # (M_full, T)
  max_exp <- (log_pmf_theta$exp()$t()$mv(e_full))$max()$item()            # scalar

  log_Qx <- mnom_logpmf(x, log_q$unsqueeze(2L), n)[, 1L]                  # (M,)
  log_Pw <- torch_logsumexp(mnom_logpmf(x, log_ws, n) + log_wts_row, dim = 2L)
  (log_Qx - log_Pw)$exp() / max_exp
}

# Unit vectors for each of the 3 categories, used to form extensions x + e_i.
unit_vecs <- torch_eye(3L, device = device, dtype = dtype)  # (3, 3)

# theta matrix (T, 3) — used to weight the three possible next outcomes.
theta_mat <- log_theta$exp()$t()

# For each consecutive pair (n, n+1), check:
#   sum_i theta_i * e_{n+1}(x + e_i) <= e_n(x)   for all x, theta
conditional_check <- lapply(seq_len(length(ns) - 1L), function(i) {
  if (ns[[i + 1L]] != ns[[i]] + 1L) return(NULL)
  n <- ns[[i]]

  log_wts_n  <- torch_tensor(weights[[i]][which.min(results[[i]]$final_max_exp), , drop = FALSE],
                              device = device, dtype = dtype)$log()  # (1, C)
  log_wts_n1 <- torch_tensor(weights[[i + 1L]][which.min(results[[i + 1L]]$final_max_exp), , drop = FALSE],
                              device = device, dtype = dtype)$log()  # (1, C)

  x_n <- build_counts_tensor(n)           # (M_n, 3)
  e_n <- eval_e(x_n, n, log_wts_n)        # (M_n,)

  # Stack e_{n+1}(x + e_i) for i = 1, 2, 3 as columns: (M_n, 3)
  e_next_mat <- torch_stack(lapply(1:3, function(cat) {
    eval_e(x_n + unit_vecs[cat, ]$unsqueeze(1L), n + 1L, log_wts_n1)
  }), dim = 2L)

  # Conditional expectation under each theta: (M_n, T)
  cond_exp <- e_next_mat$mm(theta_mat$t())

  # Max excess over e_n(x) across all outcomes and thetas
  excess <- cond_exp - e_n$unsqueeze(2L)  # (M_n, T)
  max_excess <- excess$max()$item()
  # x_n and theta (col and row) that result in the max excess
  ncols <- excess$size(2)
  flat_idx <- excess$argmax()$item()
  max_excess_xn_idx    <- (flat_idx - 1L) %/% ncols + 1L
  max_excess_theta_idx <- (flat_idx - 1L) %% ncols + 1L

  prop_excess <- excess / e_n$unsqueeze(2L)
  prop_max_excess <- prop_excess$max()$item()
  flat_idx <- prop_excess$argmax()$item()
  prop_max_excess_xn_idx <- (flat_idx - 1L) %/% ncols + 1L
  prop_max_excess_theta_idx <- (flat_idx - 1L) %% ncols + 1L

  list(
    n = n,
    n1 = n + 1L,
    max_excess = max_excess,
    max_excess_xn_idx = max_excess_xn_idx,
    max_excess_theta_idx = max_excess_theta_idx,
    max_excess_en = e_n[max_excess_xn_idx]$item(),
    prop_max_excess = prop_max_excess,
    prop_max_excess_xn_idx = prop_max_excess_xn_idx,
    prop_max_excess_theta_idx = prop_max_excess_theta_idx,
    prop_max_excess_en = e_n[prop_max_excess_xn_idx]$item()
  )
})

conditional_check <- Filter(Negate(is.null), conditional_check)

results_df <- data.frame(
  n = sapply(conditional_check, `[[`, "n"),
  n1 = sapply(conditional_check, `[[`, "n1"),
  max_excess = sapply(conditional_check, `[[`, "max_excess"),
  max_excess_xn_idx = sapply(conditional_check, `[[`, "max_excess_xn_idx"),
  max_excess_theta_idx = sapply(conditional_check, `[[`, "max_excess_theta_idx"),
  max_excess_en = sapply(conditional_check, `[[`, "max_excess_en"),
  prop_max_excess = sapply(conditional_check, `[[`, "prop_max_excess"),
  prop_max_excess_xn_idx = sapply(conditional_check, `[[`, "prop_max_excess_xn_idx"),
  prop_max_excess_theta_idx = sapply(conditional_check, `[[`, "prop_max_excess_theta_idx"),
  prop_max_excess_en = sapply(conditional_check, `[[`, "prop_max_excess_en")
)

print(results_df)


# For each weights, print sum of left weights and right weights.
for (i in seq_along(weights)) {
  w <- weights[[i]][which.min(results[[i]]$final_max_exp), ]
  cat("n =", ns[i], "\n")
  cat("  sum of left weights:", sum(w[1:501]), "\n")
  cat("  sum of right weights:", sum(w[501:1001]), "\n")
}

left_mass <- sapply(seq_along(weights), function(i) {
  w <- weights[[i]][which.min(results[[i]]$final_max_exp), ]
  sum(w[1:501])
})
