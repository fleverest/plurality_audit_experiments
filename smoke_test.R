box::purge_cache()
box::use(
  torch[
    lr_reduce_on_plateau,
    torch_manual_seed
  ],
  ripr / grids[make_simplex_grid],
  ripr / optimiser[run_ripr, adam, fw, fw_ls, mirror, lbfgs],
  ripr / monitor[db_monitor]
)

library(tidyverse)

# SQLite connections for logging
con <- DBI::dbConnect(RSQLite::SQLite(), "log/optim_log.sqlite")
DBI::dbExecute(con, "PRAGMA journal_mode=WAL")

q <- c(7 / 16, 5 / 16, 4 / 16)
thetas <- make_simplex_grid(4001)
ws <- make_simplex_grid(2001)

n <- 5L
n_restarts <- 10L
iters <- 1000000L
track_interval <- 1000L
report_interval <- 1000L
early_stopping_patience <- 10000L


adam_grid <- expand_grid(
  learning_rate = c(0.1, 0.01),
  scheduler = c("none", "reduce_on_plateau"),
  rop_patience = c(1L, 10L),
  rop_factor = c(0.9, 0.99),
  lin_smooth = c(1e-1, 1e0, 1e1, 1e1, 1e2),
  log_smooth = c(1e-6, 1e-5, 1e-4, 1e-3, 1e-2)
)

for (i in seq_len(nrow(adam_grid))) {
  params <- adam_grid[i, ]

  param_str <- paste0(names(params), "=", params, collapse = ", ")
  cat("Running with params:", param_str, "\n")

  # Set up scheduler based on parameters
  if (params$scheduler == "reduce_on_plateau") {
    scheduler <- function(opt) lr_reduce_on_plateau(
      opt,
      patience = params$rop_patience,
      factor = params$rop_factor,
      min_lr = 1e-4
    )
  } else {
    # Only run once, repeats are caught by monitor.
    scheduler <- NULL
  }

  # Construct optimiser with scheduler and learning rate
  optim <- adam(
    lr = params$learning_rate,
    scheduler = scheduler,
    loss = TRUE
  )

  # Database monitor for logging optimization progress
  tryCatch(
    mon <- db_monitor(
      con,
      optim = "adam",
      learning_rate = params$learning_rate,
      scheduler = params$scheduler,
      lin_smooth = params$lin_smooth,
      log_smooth = params$log_smooth
    ),
    error = function(e) {
      cat("Error setting up monitor:", e$message, "\n")
      mon <<- FALSE
    }
  )
  if (isFALSE(mon)) {
    cat("Skipping optimization due to monitor setup failure.\n")
    next
  }

  set.seed(20260422)
  torch_manual_seed(sample.int(.Machine$integer.max, 1L))
  res_null <- run_ripr(
    n = n,
    thetas = thetas,
    q = q,
    ws = ws,
    n_restarts = n_restarts,
    optim = optim,
    iters = iters,
    track_interval = track_interval,
    report_interval = report_interval,
    lin_smooth = params$lin_smooth,
    log_smooth = params$log_smooth,
    monitor_fn = mon,
    early_stopping_patience = early_stopping_patience
  )
}
DBI::dbDisconnect(con)


# LBFGS 
optim <- lbfgs(lr = 1e-3)
set.seed(20260422)
torch_manual_seed(sample.int(.Machine$integer.max, 1L))
res_lbfgs <- run_ripr(
  n = n,
  thetas = thetas,
  q = q,
  ws = ws,
  n_restarts = n_restarts,
  optim = optim,
  iters = iters,
  track_interval = 10,
  report_interval = 1,
  lin_smooth = 1e0,
  log_smooth = 1e-4,
  early_stopping_patience = early_stopping_patience,
  monitor_fn = db_monitor(con, optim = "lbfgs", learning_rate = 0.1, lin_smooth = 1e0, log_smooth = 1e-4, overwrite = TRUE)
)


# Some plots
res_null$weights_max_exp |>
  as.array() |>
  t() |>
  as_tibble() |>
  mutate(w_ind = 1:n()) |>
  pivot_longer(
    -w_ind,
    names_to = "restart",
    values_to = "prob",
    names_prefix = "V"
  ) |>
  ggplot(aes(x = w_ind, y = prob, group = restart)) +
  geom_line() +
  # scale_y_log10() +
  labs(title = "Best weights", y = "max_theta(E) - 1 (log scale)")


res_null$loss_history |>
  as.array() |>
  as_tibble() |>
  mutate(iter = row_number()) |>
  pivot_longer(
    -iter,
    names_to = "restart",
    values_to = "loss",
    names_prefix = "V"
  ) |>
  ggplot(aes(x = iter, y = loss, group = restart)) +
  geom_line(alpha = 0.1) +
  scale_y_log10() +
  labs(title = "Loss history (no scheduler)")

res_null_mean_weights <- res_null$weights |>
  as.array() |>
  colMeans()
  

#




# Try see how resolution of ws and thetas affects convergence

resolution_grid <- tibble(
  ws = 2 ^ seq(1, 10) + 1
) |>
  mutate(
    thetas = 2 ^ 11 + 1
  )

for (i in seq_len(nrow(resolution_grid))) {
  params <- resolution_grid[i, ]

  param_str <- paste0(names(params), "=", params, collapse = ", ")
  cat("Running with params:", param_str, "\n")

  thetas <- make_simplex_grid(params$thetas)
  ws <- make_simplex_grid(params$ws)

  # Run optimisation with settings that seem to have performed well in previous tests
  optim <- adam(lr = 1e-2, scheduler = \(opt) lr_reduce_on_plateau(opt, patience = 5000L, factor = 0.5, min_lr = 1e-5), loss = TRUE)

  set.seed(20260422)
  torch_manual_seed(sample.int(.Machine$integer.max, 1L))
  res_null <- run_ripr(
    n = n,
    thetas = thetas,
    q = q,
    ws = ws,
    n_restarts = n_restarts,
    optim = optim,
    iters = iters,
    track_interval = track_interval,
    report_interval = report_interval,
    lin_smooth = 0,
    log_smooth = 0,
    monitor_fn = db_monitor(con, optim = "adam + rop", learning_rate = 1e-2, lin_smooth = 0, log_smooth = 0, ws = params$ws, thetas = params$thetas, overwrite = TRUE),
    early_stopping_patience = early_stopping_patience
  )
}
