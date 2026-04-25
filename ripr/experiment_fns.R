box::use(
  torch[
    cuda_is_available,
    cuda_device_count,
    cuda_empty_cache,
    torch_manual_seed,
    optim_adam,
    lr_cosine_annealing,
    lr_reduce_on_plateau,
    lr_step
  ],
  ripr / optimiser[run_ripr]
)

run_ripr_target <- function(
  n,
  n_restarts,
  thetas,
  q,
  ws,
  optim = "adam",
  iters = 250000L,
  track_interval = 250L,
  report_interval = 1000L
) {
  emit_path <- sprintf("ripr/emits/experiment_n%03d.emit", n)
  gpu_stats <- if (cuda_is_available()) {
    function() {
      tryCatch(
        {
          raw <- system2(
            "nvidia-smi",
            c(
              "--query-gpu=utilization.gpu,memory.used,memory.total",
              "--format=csv,noheader,nounits"
            ),
            stdout = TRUE
          )[1L]
          parts <- trimws(strsplit(raw, ",")[[1L]])
          sprintf(
            "[GPU %s%% | VRAM %s/%s MiB]",
            parts[1L],
            parts[2L],
            parts[3L]
          )
        },
        error = function(e) ""
      )
    }
  } else {
    function() ""
  }
  emit <- function(...) {
    cat(
      format(Sys.time(), "[%H:%M:%S]"),
      ...,
      gpu_stats(),
      "\n",
      file = emit_path,
      append = TRUE
    )
  }

  emit(sprintf(
    "START | n=%d | n_restarts=%d | thetas=%d | ws=%d | q=(%s)",
    n,
    n_restarts,
    length(thetas),
    length(ws),
    paste(round(q, 4), collapse = ", ")
  ))

  torch_manual_seed(sample.int(.Machine$integer.max, 1L))
  emit("RNG seeded")

  device_info <- if (cuda_is_available()) {
    gpu_name <- tryCatch(
      system2(
        "nvidia-smi",
        c("--query-gpu=name", "--format=csv,noheader"),
        stdout = TRUE
      )[1L],
      error = function(e) "unknown"
    )
    sprintf("CUDA — %s (%d device(s))", gpu_name, cuda_device_count())
  } else {
    "CPU"
  }
  emit("Device:", device_info)

  optim_closure <- if (optim == "adam") {
    function(params, eval_fn = NULL) {
      op <- optim_adam(params, lr = 0.01)
      function(loss) {
        op$step()
        invisible(op$param_groups[[1]]$lr)
      }
    }
  } else {
    stop(sprintf("Unknown optimizer: optim=%s", optim))
  }

  emit("Calling run_ripr...")
  res <- run_ripr(
    emit_fn = emit,
    n = n,
    thetas = thetas,
    q = q,
    ws = ws,
    n_restarts = n_restarts,
    optim = optim_closure,
    use_softmax = TRUE,
    iters = iters,
    track_interval = track_interval,
    report_interval = report_interval
  )
  emit("run_ripr complete — converting results to arrays...")

  result <- list(
    weights         = as.array(res$weights),
    final_loss      = as.array(res$final_loss),
    loss_history    = as.array(res$loss_history),
    tracked_history = res$tracked_history,
    expectation_profile = as.array(res$expectation_profile)
  )
  emit("Conversion complete")

  rm(res)
  if (cuda_is_available()) {
    cuda_empty_cache()
  }

  emit("DONE")
  result
}
