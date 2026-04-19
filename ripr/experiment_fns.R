box::use(
  torch[cuda_is_available, cuda_device_count, cuda_empty_cache, torch_manual_seed, optim_adam],
  ripr / optimiser[run_ripr]
)

run_ripr_target <- function(n, n_restarts, thetas, q, ws) {
  emit_path <- sprintf("ripr/emits/experiment_n%03d.emit", n)
  emit <- function(...) {
    cat(format(Sys.time(), "[%H:%M:%S]"), ..., "\n", file = emit_path, append = TRUE)
  }

  emit(sprintf("START | n=%d | n_restarts=%d | thetas=%d | ws=%d | q=(%s)",
    n, n_restarts, length(thetas), length(ws),
    paste(round(q, 4), collapse = ", ")))

  torch_manual_seed(sample.int(.Machine$integer.max, 1L))
  emit("RNG seeded")

  device_info <- if (cuda_is_available()) {
    gpu_name <- tryCatch(
      system2("nvidia-smi", c("--query-gpu=name", "--format=csv,noheader"), stdout = TRUE)[1L],
      error = function(e) "unknown"
    )
    sprintf("CUDA — %s (%d device(s))", gpu_name, cuda_device_count())
  } else {
    "CPU"
  }
  emit("Device:", device_info)

  optim_fn <- function(params, lr = 0.01, ...) optim_adam(params = params, lr = lr)

  emit("Calling run_ripr...")
  res <- run_ripr(
    emit_fn = emit,
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
  emit("run_ripr complete — converting results to arrays...")

  result <- list(
    weights = as.array(res$weights),
    final_loss = as.array(res$final_loss),
    loss_history = as.array(res$loss_history),
    expectation_profile = as.array(res$expectation_profile)
  )
  emit("Conversion complete")

  rm(res)
  if (cuda_is_available()) cuda_empty_cache()

  emit("DONE")
  result
}
