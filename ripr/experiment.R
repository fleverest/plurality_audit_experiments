box::use(
  torch[cuda_is_available, cuda_empty_cache, torch_manual_seed, optim_adam],
  ggplot2[ggsave],
  . / grids[make_simplex_grid],
  . / optimiser[run_ripr],
  . / plots[plot_results_weights, plot_results_loss_history]
)

# ==============================================================================
# Experiment: RIPr w* for 3-category DGP path, q = (7/16, 5/16, 4/16)
# ==============================================================================

set.seed(20251111)
torch_manual_seed(20251111)

# --- Problem parameters -------------------------------------------------------

q <- c(7 / 16, 5 / 16, 4 / 16) # numerator distribution
thetas <- make_simplex_grid(101) # DGP grid (101 points on 2D simplex path)
ws <- make_simplex_grid(51) # mixture component grid

# --- Optimiser ----------------------------------------------------------------

optim_fn <- function(params, lr = 0.01, ...) {
  optim_adam(params = params, lr = lr)
}

# --- Helper: run, serialise, and free GPU memory ------------------------------

run_and_save <- function(n, n_restarts) {
  res <- run_ripr(
    n = n,
    thetas = thetas,
    q = q,
    ws = ws,
    n_restarts = n_restarts,
    optim_fn = optim_fn,
    use_softmax = TRUE,
    lr = 1.0,
    batch_size = 1000,
    n_batches = 250,
    tol = 1e-8,
    gamma = 0.95
  )
  saveRDS(
    list(
      best_weights = as.array(res$best_weights),
      best_losses = as.array(res$best_losses),
      losses_history = as.array(res$losses_history)
    ),
    sprintf("results_n%d.rds", n)
  )
  rm(res)
  gc()
  if (cuda_is_available()) cuda_empty_cache()
}

# --- Small n: lots of restarts, no memory issues ------------------------

for (n in 1:5) {
  run_and_save(n, n_restarts = 500)
}

# --- Large n: fewer restarts (vram constraints) -------------------

for (n in c(10, 20, 50, 100)) {
  run_and_save(n, n_restarts = 100)
}

# ==============================================================================
# Plots
# ==============================================================================

files <- list.files("data/", pattern = "\\.rds$", full.names = TRUE)

for (f in files) {
  res <- readRDS(f)
  ggsave(
    filename = paste0("plots/weights_", f, ".png"),
    plot = plot_results_weights(res$best_weights, res$best_losses, top = 50),
    width = 8,
    height = 6
  )
  ggsave(
    filename = paste0("plots/loss_history_", f, ".png"),
    plot = plot_results_loss_history(log(1 - res$losses_history)),
    width = 8,
    height = 6
  )
  rm(res)
  gc()
}

# ==============================================================================
# Post-hoc analysis (exploratory)
# ==============================================================================

for (f in files) {
  res <- readRDS(f)
  cat(sprintf("\nResults for %s:\n", f))
  cat(sprintf("  Best loss: %.8f\n", min(res$best_losses)))
  cat(sprintf("  Best weights (top 10 components by index):\n"))
  print(res$best_weights[which.min(res$best_losses), ] |> order() |> tail(10))
}
