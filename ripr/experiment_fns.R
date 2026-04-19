box::use(
  torch[cuda_is_available, cuda_empty_cache, torch_manual_seed, optim_adam],
  ripr / optimiser[run_ripr]
)

run_ripr_target <- function(n, n_restarts, thetas, q, ws) {
  torch_manual_seed(sample.int(.Machine$integer.max, 1L))

  optim_fn <- function(params, lr = 0.01, ...) {
    optim_adam(params = params, lr = lr)
  }

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

  result <- list(
    best_weights = as.array(res$best_weights),
    best_losses = as.array(res$best_losses),
    losses_history = as.array(res$losses_history)
  )

  rm(res)
  gc()
  if (cuda_is_available()) {
    cuda_empty_cache()
  }

  result
}
