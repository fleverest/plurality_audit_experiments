box::purge_cache()
box::use(
  torch[
    lr_reduce_on_plateau,
    torch_manual_seed
  ],
  ripr / grids[make_simplex_grid],
  ripr / optimiser[run_ripr, adam, fw, fw_ls, mirror],
  ripr / monitor[db_monitor]
)

library(tidyverse)

# SQLite connections for logging
con <- DBI::dbConnect(RSQLite::SQLite(), "log/optim.sqlite")
DBI::dbExecute(con, "PRAGMA journal_mode=WAL")

# Define the problem parameters
q <- c(7 / 16, 5 / 16, 4 / 16)
thetas <- make_simplex_grid(4001)
ws <- make_simplex_grid(2001)

n <- 5L
n_restarts <- 20L
iters <- 100000L
track_interval <- 1000L
report_interval <- 1000L


cat("\n--- sched = NULL ---\n")
set.seed(20260422)
torch_manual_seed(sample.int(.Machine$integer.max, 1L))
res_null <- run_ripr(
  n = n,
  thetas = thetas,
  q = q,
  ws = ws,
  n_restarts = n_restarts,
  optim = adam(lr = 0.001),
  iters = iters,
  track_interval = track_interval,
  report_interval = report_interval,
  lin_smooth = 1e2,
  log_smooth = 1e-5
)
DBI::dbDisconnect(con)

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
