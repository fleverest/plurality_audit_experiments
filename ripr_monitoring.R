box::use(
  ripr / monitor[init_db, db_monitor],
  ripr / dashboard[run_dashboard]
)

DB_PATH <- "log/optim_log.sqlite"

init_db(DB_PATH)

run_dashboard(DB_PATH, host = "127.0.0.1", port = 8081L)
