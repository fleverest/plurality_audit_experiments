box::use(
  shiny[
    shinyApp, runApp,
    reactivePoll, observe, isolate,
    radioButtons,
    renderText, verbatimTextOutput
  ],
  plotly[plot_ly, add_trace, layout, plotlyOutput, renderPlotly,
         plotlyProxy, plotlyProxyInvoke],
  bslib[page_sidebar, sidebar, card, card_header],
  DBI[dbConnect, dbDisconnect, dbExecute, dbGetQuery],
  RSQLite[SQLite]
)

METRICS    <- c("best_loss", "best_max_exp")
COLORS     <- c(best_loss = "#2c7bb6", best_max_exp = "#d7191c")
LABELS     <- c(best_loss = "best_loss", best_max_exp = "best_max_exp - 1")
TRANSFORMS <- list(best_loss = identity, best_max_exp = function(y) y - 1)

make_ui <- function() {
  page_sidebar(
    title   = "Optimisation Monitor",
    sidebar = sidebar(
      radioButtons("view_mode", label = "Restarts",
                   choices  = c("Min" = "min", "All" = "all"),
                   selected = "min", inline = TRUE),
      verbatimTextOutput("status")
    ),
    card(
      card_header("Loss & max expectation"),
      plotlyOutput("loss_plot", height = "500px"),
      full_screen = TRUE
    )
  )
}

# Aggregate a metric to its minimum per iteration across restarts.
min_by_iter <- function(lm, metric) {
  sub <- lm[lm$metric == metric & lm$restart >= 0L, ]
  if (nrow(sub) == 0L) return(list(x = NA_real_, y = NA_real_))
  by_iter <- tapply(sub$value, sub$iter, min)
  list(x = as.integer(names(by_iter)), y = TRANSFORMS[[metric]](as.numeric(by_iter)))
}

# Split a metric into one list entry per restart (each a list(x, y)).
by_restart <- function(lm, metric) {
  sub      <- lm[lm$metric == metric & lm$restart >= 0L, ]
  restarts <- sort(unique(sub$restart))
  lapply(restarts, function(r) {
    s <- sub[sub$restart == r, ]
    list(x = s$iter, y = TRANSFORMS[[metric]](s$value))
  })
}

make_server <- function(db_path) {
  function(input, output, session) {
    con <- dbConnect(SQLite(), db_path)
    dbExecute(con, "PRAGMA journal_mode=WAL")
    session$onSessionEnded(function() dbDisconnect(con))

    db_data <- reactivePoll(
      intervalMillis = 1000,
      session        = session,
      checkFunc      = function() {
        tryCatch(dbGetQuery(con, "SELECT MAX(iter) FROM iterations")[[1L]],
                 error = function(e) NA_integer_)
      },
      valueFunc = function() {
        tryCatch(
          dbGetQuery(con,
            "SELECT iter, restart, metric, value
             FROM loss_metrics
             WHERE metric IN ('best_loss', 'best_max_exp')
             ORDER BY iter, restart"),
          error = function(e) data.frame(
            iter = integer(0), restart = integer(0),
            metric = character(0), value = numeric(0),
            stringsAsFactors = FALSE
          )
        )
      }
    )

    # renderPlotly re-runs when view_mode changes (reactive dep),
    # but not when data changes (isolate) — the observer handles that.
    output$loss_plot <- renderPlotly({
      lm   <- isolate(db_data())
      view <- input$view_mode

      p <- plot_ly()
      if (view == "min") {
        for (m in METRICS) {
          td <- min_by_iter(lm, m)
          p  <- add_trace(p, x = td$x, y = td$y,
                          name = LABELS[[m]], type = "scatter", mode = "lines",
                          line = list(color = COLORS[[m]]))
        }
      } else {
        for (m in METRICS) {
          rts <- by_restart(lm, m)
          if (length(rts) == 0L) {
            p <- add_trace(p, x = NA_real_, y = NA_real_,
                           name = LABELS[[m]], legendgroup = m,
                           type = "scatter", mode = "lines",
                           line = list(color = COLORS[[m]]))
          } else {
            for (ri in seq_along(rts)) {
              p <- add_trace(p, x = rts[[ri]]$x, y = rts[[ri]]$y,
                             name = LABELS[[m]], legendgroup = m,
                             showlegend = (ri == 1L),
                             type = "scatter", mode = "lines",
                             line = list(color = COLORS[[m]], width = 1))
            }
          }
        }
      }

      p |> layout(
        xaxis = list(title = "Iteration"),
        yaxis = list(title = "Value", type = "log")
      )
    })

    # Observer updates trace data on each poll without redrawing.
    # Reads view_mode with isolate so toggle changes don't trigger it
    # (renderPlotly handles those by rebuilding the plot).
    observe({
      lm   <- db_data()
      view <- isolate(input$view_mode)
      if (nrow(lm) == 0L) return()

      if (view == "min") {
        xs <- lapply(METRICS, function(m) min_by_iter(lm, m)$x)
        ys <- lapply(METRICS, function(m) min_by_iter(lm, m)$y)
        plotlyProxy("loss_plot", session) |>
          plotlyProxyInvoke("restyle", list(x = xs, y = ys), as.list(0:1))

      } else {
        R <- length(unique(lm$restart[lm$restart >= 0L]))
        if (R == 0L) return()
        # Trace order mirrors renderPlotly: all restarts of metric 1, then metric 2.
        flat <- unlist(lapply(METRICS, function(m) by_restart(lm, m)), recursive = FALSE)
        plotlyProxy("loss_plot", session) |>
          plotlyProxyInvoke("restyle",
            list(x = lapply(flat, `[[`, "x"), y = lapply(flat, `[[`, "y")),
            as.list(seq_along(flat) - 1L)
          )
      }
    })

    output$status <- renderText({
      lm <- db_data()
      bl <- lm[lm$metric == "best_loss" & lm$restart >= 0L, ]
      if (nrow(bl) == 0L) return("Waiting for data...")
      latest_iter <- max(bl$iter)
      latest_loss <- min(bl$value[bl$iter == latest_iter])
      elapsed_s   <- as.numeric(Sys.time()) -
        tryCatch(dbGetQuery(con, "SELECT MIN(timestamp) FROM iterations")[[1L]],
                 error = function(e) as.numeric(Sys.time()))
      elapsed_str <- if (elapsed_s < 60) {
        sprintf("%.0fs", elapsed_s)
      } else if (elapsed_s < 3600) {
        sprintf("%dm %02ds", floor(elapsed_s / 60), floor(elapsed_s) %% 60L)
      } else {
        sprintf("%dh %02dm", floor(elapsed_s / 3600), floor((elapsed_s %% 3600) / 60))
      }
      sprintf("Iter:    %d\nLoss:    %.6f\nElapsed: %s",
              latest_iter, latest_loss, elapsed_str)
    })
  }
}

#' @export
run_dashboard <- function(db_path, host = "127.0.0.1", port = 8081L, ...) {
  runApp(shinyApp(ui = make_ui(), server = make_server(db_path)),
         host = host, port = port, ...)
}
