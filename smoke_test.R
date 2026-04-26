box::purge_cache()
box::use(
  torch[
    optim_adam,
    lr_step,
    lr_cosine_annealing,
    lr_reduce_on_plateau,
    torch_manual_seed
  ],
  ripr / grids[make_simplex_grid],
  ripr / optimiser[run_ripr, optim_frank_wolfe, optim_mirror_descent],
  ripr / monitor[db_monitor]
)

library(tidyverse)

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
con <- DBI::dbConnect(RSQLite::SQLite(), "log/optim_log_linsmooth.sqlite")
DBI::dbExecute(con, "PRAGMA journal_mode=WAL")
res_null <- run_ripr(
  n = n,
  thetas = thetas,
  q = q,
  ws = ws,
  n_restarts = n_restarts,
  optim = function(params, eval_fn = NULL) {
    op <- optim_adam(params, lr = 0.001)
    function(loss) op$step()
  },
  iters = iters,
  track_interval = track_interval,
  report_interval = report_interval,
  lin_smooth = 1e2,
  log_smooth = 1e-5,
  monitor_fn = db_monitor(con)
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


cat("--- sched = reduce_on_plateau ---\n")
set.seed(20260422)
torch_manual_seed(sample.int(.Machine$integer.max, 1L))
res_rop <- run_ripr(
  n = n,
  thetas = thetas,
  q = q,
  ws = ws,
  n_restarts = n_restarts,
  optim = function(params, eval_fn = NULL) {
    op <- optim_adam(params, lr = 0.01)
    sched <- lr_reduce_on_plateau(op, patience = 50, factor = 0.99, min_lr = 1e-4)
    function(loss) {
      op$step()
      sched$step(loss)
      invisible(op$param_groups[[1]]$lr)
    }
  },
  iters = iters,
  track_interval = track_interval,
  report_interval = report_interval,
  smooth_lambda = 1000
)

res_rop$weights_max_exp |>
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
  scale_y_log10() +
  labs(title = "Max over expectation profile (Reduce on Plateau)", y = "max expectation - 1 (log scale)")


res_rop$loss_history |>
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
  labs(title = "Loss history (Reduce on Plateau)", y = "loss (log scale)")


# Frank-Wolfe: operates directly on the simplex (use_softmax = FALSE).
# The step size schedule is gamma_t = (step / (t + step))^alpha.
# Default step=2, alpha=1 gives the classic convergent schedule 2/(t+2).
# Larger `step` keeps gamma larger for longer (slower decay).

n_restarts_fw <- 5L  # fewer restarts for FW since it's more expensive
iters_fw <- 1000000L
track_interval <- 1000L
report_interval <- 1000L

cat("\n--- Frank-Wolfe, step=2 (classic 2/(t+2)) ---\n")
set.seed(20260422)
torch_manual_seed(sample.int(.Machine$integer.max, 1L))
res_fw2 <- run_ripr(
  n = n,
  thetas = thetas,
  q = q,
  ws = ws,
  n_restarts = n_restarts_fw,
  optim = function(params, eval_fn = NULL) {
    op <- optim_frank_wolfe(params, step = 2, alpha = 1)
    function(loss) op$step()
  },
  use_softmax = FALSE,
  iters = iters_fw,
  track_interval = track_interval,
  report_interval = report_interval,
  smooth_lambda = 0.00001,
  smooth_type = "log_l2"
)


res_fw2$weights_max_exp |>
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
  scale_y_log10() +
  labs(title = "Max over expectation profile (no scheduler)", y = "max expectation - 1 (log scale)")


res_fw2$loss_history |>
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


cat("\n--- Frank-Wolfe, step=50 (slower decay) ---\n")
set.seed(20260422)
torch_manual_seed(sample.int(.Machine$integer.max, 1L))
res_fw50 <- run_ripr(
  n = n,
  thetas = thetas,
  q = q,
  ws = ws,
  n_restarts = n_restarts_fw,
  optim = function(params, eval_fn = NULL) {
    op <- optim_frank_wolfe(params, step = 50, alpha = 1)
    function(loss) op$step()
  },
  use_softmax = FALSE,
  iters = iters_fw,
  track_interval = track_interval,
  report_interval = report_interval
)

cat("\n--- Frank-Wolfe pairwise, step=2 ---\n")
set.seed(20260422)
torch_manual_seed(sample.int(.Machine$integer.max, 1L))
res_fw_pw2 <- run_ripr(
  n = n,
  thetas = thetas,
  q = q,
  ws = ws,
  n_restarts = n_restarts_fw,
  optim = function(params, eval_fn = NULL) {
    op <- optim_frank_wolfe(params, step = 2, alpha = 1, pairwise = TRUE)
    function(loss) op$step()
  },
  use_softmax = FALSE,
  iters = iters_fw,
  track_interval = track_interval,
  report_interval = report_interval
)

cat("\n--- Frank-Wolfe pairwise, step=50 ---\n")
set.seed(20260422)
torch_manual_seed(sample.int(.Machine$integer.max, 1L))
res_fw_pw50 <- run_ripr(
  n = n,
  thetas = thetas,
  q = q,
  ws = ws,
  n_restarts = n_restarts_fw,
  optim = function(params, eval_fn = NULL) {
    op <- optim_frank_wolfe(params, step = 50, alpha = 1, pairwise = TRUE)
    function(loss) op$step()
  },
  use_softmax = FALSE,
  iters = iters_fw,
  track_interval = track_interval,
  report_interval = report_interval
)

# Line search more expensive (multiple weights compared per iteration), so fewer iterations but more tracking for comparability.
iters_fw <- 1000L
track_interval <- 100L
report_interval <- 100L

cat("\n--- Frank-Wolfe exact line search, standard ---\n")
set.seed(20260422)
torch_manual_seed(sample.int(.Machine$integer.max, 1L))
res_fw_ls <- run_ripr(
  n = n,
  thetas = thetas,
  q = q,
  ws = ws,
  n_restarts = n_restarts_fw,
  optim = function(params, eval_fn = NULL) {
    op <- optim_frank_wolfe(params, line_search = TRUE)
    function(loss) op$step(eval_fn = eval_fn)
  },
  use_softmax = FALSE,
  iters = iters_fw,
  track_interval = track_interval,
  report_interval = report_interval
)

cat("\n--- Frank-Wolfe exact line search, pairwise ---\n")
set.seed(20260422)
torch_manual_seed(sample.int(.Machine$integer.max, 1L))
res_fw_pw_ls <- run_ripr(
  n = n,
  thetas = thetas,
  q = q,
  ws = ws,
  n_restarts = n_restarts_fw,
  optim = function(params, eval_fn = NULL) {
    op <- optim_frank_wolfe(params, pairwise = TRUE, line_search = TRUE)
    function(loss) op$step(eval_fn = eval_fn)
  },
  use_softmax = FALSE,
  iters = iters_fw,
  track_interval = track_interval,
  report_interval = report_interval
)

# Filter to earliest iter where all losses are < Inf, to accomodate early stopping in some rune (e.g., I interrupted early)
max_iter <- min(
  max((as.array(res_fw2$loss_history) < Inf) |> apply(2L, which) |> max()),
  max((as.array(res_fw50$loss_history) < Inf) |> apply(2L, which) |> max()),
  max((as.array(res_fw_pw2$loss_history) < Inf) |> apply(2L, which) |> max()),
  max((as.array(res_fw_pw50$loss_history) < Inf) |> apply(2L, which) |> max()),
  max((as.array(res_fw_ls$loss_history) < Inf) |> apply(2L, which) |> max()),
  max((as.array(res_fw_pw_ls$loss_history) < Inf) |> apply(2L, which) |> max())
)

bind_rows(
  res_fw2$loss_history     |> as.array() |> as_tibble() |> mutate(iter = row_number(), config = "schedule step=2"),
  res_fw50$loss_history     |> as.array() |> as_tibble() |> mutate(iter = row_number(), config = "schedule step=50"),
  res_fw_pw2$loss_history   |> as.array() |> as_tibble() |> mutate(iter = row_number(), config = "pairwise step=2"),
  res_fw_pw50$loss_history |> as.array() |> as_tibble() |> mutate(iter = row_number(), config = "pairwise step=50"),
  res_fw_ls$loss_history   |> as.array() |> as_tibble() |> mutate(iter = row_number(), config = "line search"),
  res_fw_pw_ls$loss_history |> as.array() |> as_tibble() |> mutate(iter = row_number(), config = "pairwise line search")
) |>
  pivot_longer(
    -c(iter, config),
    names_to = "restart",
    values_to = "loss",
    names_prefix = "V"
  ) |>
  filter(iter < max_iter) |>
  ggplot(aes(x = iter, y = loss, group = interaction(restart, config), colour = config)) +
  geom_line(alpha = 0.3) +
  scale_y_log10() +
  labs(title = "Frank-Wolfe: schedule vs exact line search", colour = NULL)


cat("\n--- mirror descent, lr=0.01 ---\n")
set.seed(20260422)
torch_manual_seed(sample.int(.Machine$integer.max, 1L))
n_restarts <- 1L # just one restart to keep it quick, since mirror descent is more expensive per iteration than FW
iters <- 1000000L
res_md <- run_ripr(
  n = n,
  thetas = thetas,
  q = q,
  ws = ws,
  n_restarts = n_restarts,
  optim = function(params, eval_fn = NULL) {
    op <- optim_mirror_descent(params, lr = 0.0001)
    function(loss) op$step()
  },
  use_softmax = FALSE,
  iters = iters,
  track_interval = track_interval,
  report_interval = report_interval
)

res_md$loss_history |>
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
  geom_line(alpha = 0.3) +
  scale_y_log10() +
  labs(title = "Mirror descent (lr=0.01)")

# Weight profile for the best restart
best_r <- which.min(res_md$final_loss |> as.array())
res_md$weights[best_r, ] |>
  as.array() |>
  as_tibble() |>
  setNames("weight") |>
  mutate(component = seq_len(n())) |>
  ggplot(aes(x = component, y = weight)) +
  geom_col() +
  labs(title = sprintf("Mirror descent best-restart weights (restart %d)", best_r))


# Compare smooth_lambda values with mirror descent
iters <- 1000000L
lambdas <- c(0, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4)
res_md_smooth <- lapply(lambdas, function(lam) {
  cat(sprintf("\n--- mirror descent, smooth_lambda=%.0e ---\n", lam))
  set.seed(20260422)
  torch_manual_seed(sample.int(.Machine$integer.max, 1L))
  run_ripr(
    n = n,
    thetas = thetas,
    q = q,
    ws = ws,
    n_restarts = n_restarts,
    optim = function(params, eval_fn = NULL) {
      op <- optim_mirror_descent(params, lr = 0.00005)
      function(loss) op$step()
    },
    use_softmax = FALSE,
    iters = iters,
    track_interval = track_interval,
    report_interval = report_interval,
    smooth_lambda = lam
  )
})

# Compare weight distributions for best restart across smooth_lambda values
bind_rows(
  lapply(seq_along(lambdas), function(i) {
    best_r <- which.min(res_md_smooth[[i]]$final_loss |> as.array())
    res_md_smooth[[i]]$weights[best_r, ] |>
      as.array() |>
      tibble(weight = _, lambda = as.character(lambdas[[i]]), component = 1:2001)
  })
) |> # Now plot the weight profiles, labelled by lambda in order of smallest to largest
  ggplot(aes(x = component, y = weight)) +
  geom_line() +
  scale_y_log10() +
  facet_wrap(vars(lambda), ncol = 2) +
  labs(title = "Mirror descent best-restart weights by smooth_lambda", y = "weight (log scale)")

# Compare best-restart maximum over expectation profile across smooth_lambda values
bind_rows(
  lapply(seq_along(lambdas) |> head(-2), function(i) {
    best_r <- which.min(res_md_smooth[[i]]$final_loss |> as.array())
    tibble(
      lambda = as.character(lambdas[[i]]),
      max_expectation = res_md_smooth[[i]]$expectation_profile[best_r, ] |> as.array() |> max()
    )
  })
) |>
  ggplot(aes(x = lambda, y = max_expectation - 1)) +
  geom_point() + # Log scale and minor grid lines to help see differences among smaller lambda values
  scale_y_log10() +
  theme(panel.grid.minor = element_line(color = "grey80", size = 0.5)) +
  labs(title = "Mirror descent best-restart maximum over expectation by smooth_lambda")
