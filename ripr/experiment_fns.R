box::use(
  torch[cuda_is_available, cuda_device_count, cuda_empty_cache, torch_manual_seed, optim_adam],
  ripr / optimiser[run_ripr]
)

run_ripr_target <- function(n, n_restarts, thetas, q, ws) {
  torch_manual_seed(sample.int(.Machine$integer.max, 1L))

  device_info <- if (cuda_is_available()) {
    gpu_name <- tryCatch(
      system2("nvidia-smi", c("--query-gpu=name", "--format=csv,noheader"), stdout = TRUE)[1L],
      error = function(e) "unknown"
    )
    sprintf("CUDA — %s (%d device(s))", gpu_name, cuda_device_count())
  } else {
    "CPU"
  }
  message(sprintf(
    "[run_ripr_target] n=%d | n_restarts=%d | thetas=%d | ws=%d | q=(%s) | device=%s",
    n, n_restarts, length(thetas), length(ws),
    paste(round(q, 4), collapse = ", "),
    device_info
  ))

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
