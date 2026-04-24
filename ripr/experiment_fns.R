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
  sched = "null",
  optim = "adam"
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

  iters          <- 250000L
  track_interval <- 1000L
  report_interval <- 10000L

  optim_closure <- if (optim == "adam" && sched == "null") {
    function(params) {
      op <- optim_adam(params, lr = 0.01)
      function(loss) {
        op$step()
        invisible(op$param_groups[[1]]$lr)
      }
    }
  } else if (optim == "adam" && sched == "step") {
    function(params) {
      op <- optim_adam(params, lr = 0.01)
      sched_obj <- lr_step(op, step_size = 1, gamma = 0.001^(1 / iters))
      function(loss) {
        op$step()
        sched_obj$step()
        invisible(op$param_groups[[1]]$lr)
      }
    }
  } else if (optim == "adam" && sched == "cosine") {
    function(params) {
      op <- optim_adam(params, lr = 0.01)
      sched_obj <- lr_cosine_annealing(op, T_max = iters, eta_min = 0.001)
      function(loss) {
        op$step()
        sched_obj$step()
        invisible(op$param_groups[[1]]$lr)
      }
    }
  } else if (optim == "adam" && sched == "plateau") {
    function(params) {
      op <- optim_adam(params, lr = 0.01)
      sched_obj <- lr_reduce_on_plateau(op, patience = 5, factor = 0.5, min_lr = 1e-4)
      function(loss) {
        op$step()
        sched_obj$step(loss)
        invisible(op$param_groups[[1]]$lr)
      }
    }
  } else {
    stop(sprintf("Unknown optimizer/scheduler combination: optim=%s sched=%s", optim, sched))
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
