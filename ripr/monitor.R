box::use(
  DBI[dbConnect, dbDisconnect, dbExecute, dbWithTransaction, dbAppendTable],
  RSQLite[SQLite]
)

#' @export
init_db <- function(db_path) {
  con <- dbConnect(SQLite(), db_path)
  on.exit(dbDisconnect(con))
  dbExecute(con, "PRAGMA journal_mode=WAL")
  dbExecute(con, "
    CREATE TABLE IF NOT EXISTS iterations (
      iter      INTEGER PRIMARY KEY,
      timestamp REAL,
      loss      REAL
    )
  ")
  # Per-restart scalars: cur_loss, best_loss, cur_max_exp, best_max_exp, tracked.
  dbExecute(con, "
    CREATE TABLE IF NOT EXISTS loss_metrics (
      iter    INTEGER,
      restart INTEGER,
      metric  TEXT,
      value   REAL
    )
  ")
  # Per-restart, per-component arrays.
  # type: 'cur_weight' | 'best_weight' | 'best_weight_max_exp' | 'expectation'
  dbExecute(con, "
    CREATE TABLE IF NOT EXISTS profiles (
      iter      INTEGER,
      restart   INTEGER,
      type      TEXT,
      component INTEGER,
      value     REAL
    )
  ")
  invisible(db_path)
}

# Write one tracked iteration snapshot to the database.
#
# `state` is the named list passed by monitor_fn in optimize_mixture_weights:
#   $iter                 — integer iteration number
#   $cur_loss             — numeric vector (n_restarts)
#   $cur_max_exp          — numeric vector (n_restarts)
#   $best_loss            — numeric vector (n_restarts)
#   $best_max_exp         — numeric vector (n_restarts)
#   $cur_weights          — numeric matrix (n_restarts × C)
#   $best_weights         — numeric matrix (n_restarts × C)
#   $best_weights_max_exp — numeric matrix (n_restarts × C)
#   $exp_llr              — numeric matrix (n_restarts × T)
#   $tracked              — numeric scalar (NA when unavailable)
log_iteration <- function(con, state) {
  iter     <- state$iter
  R        <- length(state$cur_loss)
  restarts <- seq_len(R) - 1L  # 0-indexed

  dbWithTransaction(con, {
    dbExecute(con,
      "INSERT OR REPLACE INTO iterations (iter, timestamp, loss) VALUES (?, ?, ?)",
      list(as.integer(iter), as.numeric(Sys.time()), min(state$best_loss))
    )

    scalar_metrics <- c("cur_loss", "cur_max_exp", "best_loss", "best_max_exp")
    metrics_rows <- do.call(rbind, lapply(scalar_metrics, function(m) {
      data.frame(
        iter    = as.integer(iter),
        restart = restarts,
        metric  = m,
        value   = as.numeric(state[[m]]),
        stringsAsFactors = FALSE
      )
    }))
    if (!is.na(state$tracked)) {
      metrics_rows <- rbind(metrics_rows, data.frame(
        iter    = as.integer(iter),
        restart = -1L,
        metric  = "tracked",
        value   = as.numeric(state$tracked),
        stringsAsFactors = FALSE
      ))
    }
    dbAppendTable(con, "loss_metrics", metrics_rows)

    profile_types <- list(
      cur_weight          = state$cur_weights,
      best_weight         = state$best_weights,
      best_weight_max_exp = state$best_weights_max_exp,
      expectation         = state$exp_llr
    )
    profile_rows <- do.call(rbind, lapply(names(profile_types), function(type) {
      mat <- profile_types[[type]]  # (n_restarts, K)
      K   <- ncol(mat)
      do.call(rbind, lapply(seq_len(R), function(r) {
        data.frame(
          iter      = as.integer(iter),
          restart   = r - 1L,
          type      = type,
          component = seq_len(K) - 1L,
          value     = as.numeric(mat[r, ]),
          stringsAsFactors = FALSE
        )
      }))
    }))
    dbAppendTable(con, "profiles", profile_rows)
  })
}

#' Returns a monitor_fn closure bound to an open DBI connection.
#' The connection must have WAL mode enabled before passing it here.
#' @export
db_monitor <- function(con) {
  function(state) log_iteration(con, state)
}
