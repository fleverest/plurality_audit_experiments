box::use(
  ripr / plurality_ripr[run_plurality_ripr],
  ripr / plurality_geometry[full_plurality_face_descriptors]
)

#' Run one (config, Q) combination and tag the resulting history with the
#' identifying columns (`Q_name`, `algo`, `n`) needed once all combinations
#' are stacked together downstream.
#' @export
run_one <- function(q_info, cfg) {
  args <- c(
    list(
      q = q_info$Q,
      n = q_info$n,
      face_descriptors = full_plurality_face_descriptors(q_info$K),
      checkpoint_iters = 0L:cfg$fw_iters
    ),
    cfg[setdiff(names(cfg), c("name"))]
  )
  result <- do.call(run_plurality_ripr, args)
  # Rewrite iter for em_only to reflect the number of atoms rather than steps.
  if (cfg$name == "em_only") {
    result$history$iter <- cfg$init
  }
  result$history$Q_name <- q_info$name
  result$history$algo <- cfg$name
  result$history$n <- q_info$n
  list(
    history = result$history,
    checkpoints = result$checkpoints,
    mixture = result$mixture
  )
}
