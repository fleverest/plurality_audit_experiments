box::use(
  shiny[
    shinyApp, runApp,
    reactivePoll, observe, isolate,
    renderText, verbatimTextOutput
  ],
  plotly[plot_ly, add_trace, layout, plotlyOutput, renderPlotly,
         plotlyProxy, plotlyProxyInvoke],
  bslib[page_sidebar, sidebar, card, card_header],
  viridisLite[viridis],
  DBI[dbConnect, dbDisconnect, dbExecute, dbGetQuery],
  RSQLite[SQLite]
)

make_ui <- function() {
  page_sidebar(
    title  = "Optimisation Monitor",
    sidebar = sidebar(verbatimTextOutput("status")),
    card(
      card_header("Loss (best across restarts)"),
      plotlyOutput("loss_plot", height = "500px"),
      full_screen = TRUE
    )
  )
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
          dbGetQuery(con, "SELECT iter, loss FROM iterations ORDER BY iter"),
          error = function(e) data.frame(iter = integer(0L), loss = numeric(0L))
        )
      }
    )

    output$loss_plot <- renderPlotly({
      iters <- isolate(db_data())
      plot_ly(
        x    = if (nrow(iters) > 0L) iters$iter else NA_real_,
        y    = if (nrow(iters) > 0L) iters$loss else NA_real_,
        type = "scatter", mode = "lines",
        line = list(color = "#2c7bb6"), name = "best loss"
      ) |>
        layout(
          xaxis       = list(title = "Iteration"),
          yaxis       = list(title = "Best loss (min across restarts)"),
          showlegend  = FALSE,
          updatemenus = list(list(
            type      = "buttons",
            direction = "left",
            x = 0, xanchor = "left",
            y = 1.08, yanchor = "top",
            buttons   = list(
              list(method = "relayout",
                   args   = list(list("yaxis.type" = "linear")),
                   label  = "Linear"),
              list(method = "relayout",
                   args   = list(list("yaxis.type" = "log")),
                   label  = "Log")
            )
          ))
        )
    })

    observe({
      iters <- db_data()
      if (nrow(iters) == 0L) return()
      plotlyProxy("loss_plot", session) |>
        plotlyProxyInvoke("restyle",
          list(x = list(iters$iter), y = list(iters$loss)),
          list(0L)
        )
    })

    output$status <- renderText({
      iters <- db_data()
      if (nrow(iters) == 0L) return("Waiting for data...")
      latest    <- iters[nrow(iters), ]
      elapsed_s <- as.numeric(Sys.time()) -
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
              latest$iter, latest$loss, elapsed_str)
    })
  }
}

#' @export
run_dashboard <- function(db_path, host = "127.0.0.1", port = 8081L, ...) {
  runApp(shinyApp(ui = make_ui(), server = make_server(db_path)),
         host = host, port = port, ...)
}
