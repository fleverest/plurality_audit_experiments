box::use(
  DBI[dbConnect, dbDisconnect, dbExecute, dbGetQuery, dbWithTransaction, dbAppendTable],
  RSQLite[SQLite]
)

#' @export
init_db <- function(db_path) {
  con <- dbConnect(SQLite(), db_path)
  on.exit(dbDisconnect(con))
  dbExecute(con, "PRAGMA journal_mode=WAL")
  dbExecute(con, "
    CREATE TABLE IF NOT EXISTS experiments (
      id         INTEGER PRIMARY KEY AUTOINCREMENT,
      created_at REAL
    )
  ")
  dbExecute(con, "
    CREATE TABLE IF NOT EXISTS experiment_params (
      experiment_id INTEGER,
      key           TEXT,
      value         TEXT
    )
  ")
  dbExecute(con, "
    CREATE TABLE IF NOT EXISTS iterations (
      experiment_id INTEGER,
      iter          INTEGER,
      timestamp     REAL,
      loss          REAL,
      PRIMARY KEY (experiment_id, iter)
    )
  ")
  dbExecute(con, "
    CREATE TABLE IF NOT EXISTS loss_metrics (
      experiment_id INTEGER,
      iter          INTEGER,
      restart       INTEGER,
      metric        TEXT,
      value         REAL
    )
  ")
  dbExecute(con, "
    CREATE TABLE IF NOT EXISTS profiles (
      experiment_id INTEGER,
      iter          INTEGER,
      restart       INTEGER,
      type          TEXT,
      component     INTEGER,
      value         REAL
    )
  ")
  # Migrate pre-experiment_params DBs.
  for (tbl in c("iterations", "loss_metrics", "profiles")) {
    tryCatch(
      dbExecute(con, paste0("ALTER TABLE ", tbl, " ADD COLUMN experiment_id INTEGER")),
      error = function(e) invisible(NULL)
    )
  }
  invisible(db_path)
}

# Returns the experiment_id whose param set exactly matches `params` (a named
# list), or NULL if no match exists. "Exactly" means same keys and same values;
# an experiment with extra keys is a different experiment.
find_matching_experiment <- function(con, params) {
  all_params <- dbGetQuery(con,
    "SELECT experiment_id, key, value FROM experiment_params")

  all_exp_ids <- dbGetQuery(con, "SELECT id FROM experiments")$id

  for (exp_id in all_exp_ids) {
    ep <- all_params[all_params$experiment_id == exp_id, ]
    if (nrow(ep) != length(params)) next
    if (!setequal(ep$key, names(params))) next
    if (all(mapply(function(k, v) {
      ep$value[ep$key == k] == as.character(v)
    }, names(params), params))) return(exp_id)
  }
  NULL
}

log_iteration <- function(con, state, experiment_id) {
  iter     <- state$iter
  R        <- length(state$cur_loss)
  restarts <- seq_len(R) - 1L

  dbWithTransaction(con, {
    dbExecute(con,
      "INSERT OR REPLACE INTO iterations (experiment_id, iter, timestamp, loss)
       VALUES (?, ?, ?, ?)",
      list(experiment_id, as.integer(iter), as.numeric(Sys.time()), min(state$best_loss))
    )

    scalar_metrics <- c("cur_loss", "cur_max_exp", "best_loss", "best_max_exp")
    metrics_rows <- do.call(rbind, lapply(scalar_metrics, function(m) {
      data.frame(
        experiment_id = experiment_id,
        iter    = as.integer(iter),
        restart = restarts,
        metric  = m,
        value   = as.numeric(state[[m]]),
        stringsAsFactors = FALSE
      )
    }))
    if (!is.na(state$tracked)) {
      metrics_rows <- rbind(metrics_rows, data.frame(
        experiment_id = experiment_id,
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
      mat <- profile_types[[type]]
      K   <- ncol(mat)
      do.call(rbind, lapply(seq_len(R), function(r) {
        data.frame(
          experiment_id = experiment_id,
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

#' Returns a monitor_fn closure for use with optimize_mixture_weights / run_ripr.
#'
#' Named arguments in `...` are stored as experiment parameters. Calling with
#' the exact same set of keys and values as an existing experiment is an error
#' unless overwrite = TRUE, in which case that experiment's run data is wiped
#' and the run starts fresh under the same experiment ID.
#'
#' Parameters absent from a given call but present in other experiments in the
#' same DB are displayed as "unset" in the dashboard.
#'
#' @export
db_monitor <- function(con, ..., overwrite = FALSE) {
  params <- list(...)
  # Remove NULLs, e.g., `not_used = NULL` means this param is not set for this experiment.
  params <- params[!sapply(params, is.null)]
  if (length(params) > 0 && any(names(params) == "")) {
    stop("All arguments to db_monitor must be named.")
  }

  existing_id <- find_matching_experiment(con, params)

  if (!is.null(existing_id)) {
    if (!overwrite) {
      stop(sprintf(paste0(
        "An experiment with these parameters already exists (id = %d).\n",
        "Use overwrite = TRUE to wipe its run data and restart."
      ), existing_id))
    }
    for (tbl in c("iterations", "loss_metrics", "profiles")) {
      dbExecute(con,
        paste0("DELETE FROM ", tbl, " WHERE experiment_id = ?"),
        list(existing_id)
      )
    }
    return(function(state) log_iteration(con, state, existing_id))
  }

  dbExecute(con,
    "INSERT INTO experiments (created_at) VALUES (?)",
    list(as.numeric(Sys.time()))
  )
  experiment_id <- dbGetQuery(con, "SELECT last_insert_rowid()")[[1L]]

  if (length(params) > 0) {
    dbAppendTable(con, "experiment_params", data.frame(
      experiment_id = experiment_id,
      key           = names(params),
      value         = as.character(unlist(params)),
      stringsAsFactors = FALSE
    ))
  }

  function(state) log_iteration(con, state, experiment_id)
}
