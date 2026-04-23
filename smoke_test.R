box::use(
  torch[
    optim_adam,
    lr_step,
    lr_cosine_annealing,
    lr_reduce_on_plateau,
    torch_manual_seed
  ],
  ripr / grids[make_simplex_grid],
  ripr / optimiser[run_ripr]
)

library(tidyverse)

q <- c(7 / 16, 5 / 16, 4 / 16)
thetas <- make_simplex_grid(301)
ws <- make_simplex_grid(201)

n <- 5
n_restarts <- 1
batch_size <- 1000
n_batches <- 100
track_freq <- 1L

cat("\n--- sched = NULL ---\n")
set.seed(20260422)
torch_manual_seed(sample.int(.Machine$integer.max, 1L))
res_null <- run_ripr(
  n = n,
  thetas = thetas,
  q = q,
  ws = ws,
  n_restarts = n_restarts,
  optim_fn = function(params) optim_adam(params, lr = 0.01),
  sched_fn = NULL,
  batch_size = batch_size,
  n_batches = n_batches,
  track_freq = track_freq
)

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
  #filter(iter < 500) |>
  ggplot(aes(x = iter, y = loss, group = restart)) +
  geom_point(alpha = 0.1) +
  scale_y_log10() +
  labs(title = "Loss history (no scheduler)") +
  facet_grid(rows = vars(restart))

cat("--- sched = reduce_on_plateau ---\n")
set.seed(20260422)
torch_manual_seed(sample.int(.Machine$integer.max, 1L))
res_rop <- run_ripr(
  n = n,
  thetas = thetas,
  q = q,
  ws = ws,
  n_restarts = n_restarts,
  optim_fn = function(params) optim_adam(params, lr = 0.01),
  sched_fn = function(op) {
    sched <- lr_reduce_on_plateau(
      op,
      patience = 25,
      factor = 0.99,
      min_lr = 1e-4
    )
    function(loss) sched$step(loss)
  },
  batch_size = batch_size,
  n_batches = n_batches,
  track_freq = track_freq
)

# check how often loss_history decreases, which should trigger the scheduler
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
  group_by(restart) |>
  summarise(
    increases_twice = sum(
      loss > lag(loss) & lag(loss) > lag(loss, n = 2),
      na.rm = TRUE
    ),
    total = n(),
    decrease_rate = increases_twice / total
  )

res_rop$loss_history |>
  as.array() |>
  as_tibble() |>
  mutate(iter = row_number(), lr = res_rop$lr_history) |>
  pivot_longer(
    -c(iter, lr),
    names_to = "restart",
    values_to = "loss",
    names_prefix = "V"
  ) |>
  #filter(iter < 500) |>
  ggplot(aes(x = iter, y = loss, group = restart)) +
  geom_point(alpha = 0.5) +
  geom_line(aes(y = lr), color = "red") +
  scale_y_log10() +
  labs(title = "Loss history (reduce_on_plateau scheduler)")


cat("--- sched = step ---\n")
set.seed(20260422)
torch_manual_seed(sample.int(.Machine$integer.max, 1L))
res_step <- run_ripr(
  n = n,
  thetas = thetas,
  q = q,
  ws = ws,
  n_restarts = n_restarts,
  optim_fn = function(params) optim_adam(params, lr = 0.01),
  sched_fn = function(op) {
    sched <- lr_step(
      op,
      step_size = 1,
      gamma = 0.1^(1 / (n_batches * batch_size))
    )
    function(loss) sched$step()
  },
  batch_size = batch_size,
  n_batches = n_batches,
  track_freq = track_freq
)

# Plot loss history for each restart
res_step$loss_history |>
  as.array() |>
  as_tibble() |>
  mutate(iter = row_number()) |>
  pivot_longer(
    -iter,
    names_to = "restart",
    values_to = "loss",
    names_prefix = "V"
  ) |>
  #filter(iter < 500) |>
  ggplot(aes(x = iter, y = loss - 1, group = restart)) +
  geom_point(alpha = 0.5) +
  scale_y_log10() +
  labs(title = "Loss history (step scheduler)") +
  facet_grid(rows = vars(restart))


cat("\n--- sched = cosine ---\n")
set.seed(20260422)
torch_manual_seed(sample.int(.Machine$integer.max, 1L))
res_cosine <- run_ripr(
  n = n,
  thetas = thetas,
  q = q,
  ws = ws,
  n_restarts = n_restarts,
  optim_fn = function(params) optim_adam(params, lr = 0.01),
  sched_fn = function(op) {
    sched <- lr_cosine_annealing(
      op,
      T_max = n_batches * batch_size,
      eta_min = 0.001
    )
    function(loss) sched$step()
  },
  batch_size = batch_size,
  n_batches = n_batches,
  track_freq = track_freq
)

res_cosine$loss_history |>
  as.array() |>
  as_tibble() |>
  mutate(iter = row_number()) |>
  pivot_longer(
    -iter,
    names_to = "restart",
    values_to = "loss",
    names_prefix = "V"
  ) |>
  #filter(iter < 500) |>
  ggplot(aes(x = iter, y = loss - 1, group = restart)) +
  geom_point(alpha = 0.5) +
  scale_y_log10() +
  labs(title = "Loss history (cosine annealing scheduler)") +
  facet_grid(rows = vars(restart))


cat("final losses (cosine):", sort(as.array(res_cosine$final_loss)), "\n")
cat("final losses (step):", sort(as.array(res_step$final_loss)), "\n")
cat("final losses (null):", sort(as.array(res_null$final_loss)), "\n")
cat(
  "final losses (reduce_on_plateau):",
  sort(as.array(res_rop$final_loss)),
  "\n"
)
