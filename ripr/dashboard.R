box::use(
  shiny[
    shinyApp, runApp,
    reactive, reactiveVal, reactivePoll, observe, observeEvent, isolate,
    radioButtons, selectInput, updateSelectInput, actionButton,
    renderUI, uiOutput,
    renderText, verbatimTextOutput,
    tagList, tags
  ],
  plotly[plot_ly, add_trace, layout, plotlyOutput, renderPlotly,
         plotlyProxy, plotlyProxyInvoke],
  bslib[page_sidebar, sidebar, layout_columns, card, card_header],
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
      width = "30%",
      uiOutput("filters_ui"),
      tags$hr(class = "my-2"),
      tags$p("Experiments", class = "fw-semibold mb-1"),
      uiOutput("exp_list_ui")
    ),
    layout_columns(
      col_widths = c(9, 3),
      card(
        card_header("Loss & max expectation"),
        radioButtons("view_mode", label = NULL,
                     choices  = c("Min across restarts" = "min", "All restarts" = "all"),
                     selected = "min", inline = TRUE),
        plotlyOutput("loss_plot", height = "460px"),
        full_screen = TRUE
      ),
      card(
        card_header("Run info"),
        verbatimTextOutput("status")
      )
    )
  )
}

# Build labels for a set of experiments, showing only `show_keys`.
# Keys absent from a particular experiment are shown as "unset".
format_exp_labels <- function(exp_df, params_df, show_keys) {
  sapply(exp_df$id, function(id) {
    ep      <- params_df[params_df$experiment_id == id, ]
    created <- format(as.POSIXct(exp_df$created_at[exp_df$id == id],
                                 origin = "1970-01-01"), "%m-%d %H:%M")
    if (length(show_keys) == 0L) return(sprintf("[%s] (no free params)", created))
    parts <- sapply(show_keys, function(k) {
      v <- ep$value[ep$key == k]
      sprintf("%s=%s", k, if (length(v) == 0L) "unset" else v)
    })
    sprintf("[%s] %s", created, paste(parts, collapse = " / "))
  })
}

min_by_iter <- function(lm, metric) {
  sub <- lm[lm$metric == metric & lm$restart >= 0L, ]
  if (nrow(sub) == 0L) return(list(x = NA_real_, y = NA_real_))
  by_iter <- tapply(sub$value, sub$iter, min)
  list(x = as.integer(names(by_iter)), y = TRANSFORMS[[metric]](as.numeric(by_iter)))
}

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

    empty_lm <- function() data.frame(
      experiment_id = integer(0), iter = integer(0), restart = integer(0),
      metric = character(0), value = numeric(0), stringsAsFactors = FALSE
    )

    # Slow poll: only ticks when experiments are added.
    exp_poll <- reactivePoll(
      intervalMillis = 5000L,
      session        = session,
      checkFunc      = function() {
        tryCatch(dbGetQuery(con, "SELECT MAX(id) FROM experiments")[[1L]],
                 error = function(e) NA_integer_)
      },
      valueFunc = function() Sys.time()
    )

    # Fast poll: ticks on every new iteration.
    db_poll <- reactivePoll(
      intervalMillis = 1000L,
      session        = session,
      checkFunc      = function() {
        tryCatch(dbGetQuery(con, "SELECT MAX(iter) FROM iterations")[[1L]],
                 error = function(e) NA_integer_)
      },
      valueFunc = function() Sys.time()
    )

    experiments <- reactive({
      exp_poll()
      exp_df <- tryCatch(
        dbGetQuery(con, "SELECT * FROM experiments ORDER BY created_at"),
        error = function(e) data.frame(id = integer(0), created_at = numeric(0),
                                       stringsAsFactors = FALSE)
      )
      params_df <- tryCatch(
        dbGetQuery(con, "SELECT * FROM experiment_params"),
        error = function(e) data.frame(experiment_id = integer(0), key = character(0),
                                       value = character(0), stringsAsFactors = FALSE)
      )
      list(exp = exp_df, params = params_df)
    })

    # ---------------------------------------------------------------------------
    # Filter state: a list of list(id, key) with stable IDs.
    # Values are read directly from input$fv_<id> so filtered_experiments
    # automatically depends on them without any syncing observer.
    # ---------------------------------------------------------------------------
    filter_counter  <- 0L
    active_filters  <- reactiveVal(list())

    # Selecting a param from the "add filter" dropdown creates a new filter row.
    observeEvent(input$add_filter_key, {
      key <- input$add_filter_key
      if (is.null(key) || key == "") return()
      filter_counter <<- filter_counter + 1L
      fid <- filter_counter
      observeEvent(input[[paste0("frm_", fid)]], {
        active_filters(Filter(function(f) f$id != fid, active_filters()))
      }, ignoreInit = TRUE, once = TRUE)
      active_filters(c(isolate(active_filters()),
                       list(list(id = fid, key = key))))
      updateSelectInput(session, "add_filter_key", selected = "")
    }, ignoreInit = TRUE)

    output$filters_ui <- renderUI({
      d       <- experiments()
      filters <- active_filters()
      all_keys  <- sort(unique(d$params$key))
      used_keys <- sapply(filters, `[[`, "key")
      unused    <- setdiff(all_keys, used_keys)

      rows <- lapply(filters, function(f) {
        vals <- c("(all)" = "", "(set)" = "__set__",
                  sort(unique(d$params$value[d$params$key == f$key])))
        tags$div(class = "d-flex gap-1 align-items-center mb-1",
          tags$div(class = "fw-semibold small", style = "min-width:6rem", f$key),
          tags$div(class = "flex-fill",
            selectInput(paste0("fv_", f$id), NULL, choices = vals)),
          tags$div(class = "mb-2",
            actionButton(paste0("frm_", f$id), "×",
                         class = "btn-sm btn-outline-danger"))
        )
      })

      add_ctrl <- if (length(unused) > 0L)
        selectInput("add_filter_key", NULL,
                    choices  = c("(+ add filter)" = "", unused),
                    selected = "")

      tagList(tagList(rows), add_ctrl)
    })

    # ---------------------------------------------------------------------------
    # Filtered experiment list
    # ---------------------------------------------------------------------------
    filtered_experiments <- reactive({
      d       <- experiments()
      filters <- active_filters()
      ids     <- d$exp$id

      for (f in filters) {
        val <- input[[paste0("fv_", f$id)]]
        if (is.null(val) || val == "") next
        matched <- if (val == "__set__")
          unique(d$params$experiment_id[d$params$key == f$key])
        else
          d$params$experiment_id[d$params$key == f$key & d$params$value == val]
        ids <- ids[ids %in% matched]
      }
      d$exp[d$exp$id %in% ids, ]
    })

    output$exp_list_ui <- renderUI({
      d          <- experiments()
      fe         <- filtered_experiments()
      filters    <- active_filters()
      active_keys <- sapply(filters, `[[`, "key")
      all_keys    <- sort(unique(d$params$key))
      # Only show params not already fixed by an active (non-"all") filter.
      fixed_keys  <- sapply(filters, function(f) {
        val <- input[[paste0("fv_", f$id)]]
        if (!is.null(val) && val != "" && val != "__set__") f$key else NA_character_
      })
      show_keys   <- setdiff(all_keys, fixed_keys[!is.na(fixed_keys)])

      if (nrow(fe) == 0L) return(tags$p("No matching experiments.", class = "text-muted"))

      labels   <- format_exp_labels(fe, d$params, show_keys)
      current  <- isolate(input$experiment_id)
      selected <- if (!is.null(current) && current %in% as.character(fe$id)) current
                  else as.character(fe$id[nrow(fe)])
      radioButtons("experiment_id", NULL,
                   choiceValues = as.character(fe$id),
                   choiceNames  = as.list(labels),
                   selected     = selected)
    })

    # ---------------------------------------------------------------------------
    # Plot data
    # ---------------------------------------------------------------------------

    # Axis range spanning all filtered experiments — keeps the plot stable when
    # switching between experiments.  Depends on db_poll so it grows as runs progress.
    filtered_range <- reactive({
      db_poll()
      fe <- filtered_experiments()
      if (nrow(fe) == 0L) return(NULL)
      ids <- paste(fe$id, collapse = ",")
      lm <- tryCatch(
        dbGetQuery(con, sprintf(
          "SELECT iter, metric, value FROM loss_metrics
           WHERE experiment_id IN (%s)
             AND metric IN ('best_loss', 'best_max_exp')
             AND restart >= 0", ids)),
        error = function(e) NULL
      )
      if (is.null(lm) || nrow(lm) == 0L) return(NULL)
      y_vals <- unlist(lapply(METRICS, function(m) {
        v <- TRANSFORMS[[m]](lm$value[lm$metric == m])
        v[is.finite(v) & v > 0]
      }))
      if (length(y_vals) == 0L) return(NULL)
      # 10 % padding in log space
      y_log  <- log10(range(y_vals))
      pad    <- diff(y_log) * 0.1
      list(
        x = c(min(lm$iter), max(lm$iter)),
        y = c(y_log[[1L]] - pad, y_log[[2L]] + pad)
      )
    })

    db_data <- reactive({
      db_poll()
      exp_id <- suppressWarnings(as.integer(input$experiment_id))
      if (length(exp_id) == 0L || is.na(exp_id)) return(empty_lm())
      tryCatch(
        dbGetQuery(con,
          "SELECT iter, restart, metric, value FROM loss_metrics
           WHERE experiment_id = ? AND metric IN ('best_loss', 'best_max_exp')
           ORDER BY iter, restart",
          list(exp_id)),
        error = function(e) empty_lm()
      )
    })

    # Re-renders on view_mode or experiment switch; data updates go via restyle.
    output$loss_plot <- renderPlotly({
      lm   <- isolate(db_data())
      view <- input$view_mode
      input$experiment_id   # reactive dep so experiment switch triggers redraw

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
      rng <- isolate(filtered_range())
      p |> layout(
        xaxis = list(title = "Iteration",
                     range = if (!is.null(rng)) rng$x else NULL),
        yaxis = list(title = "Value", type = "log",
                     range = if (!is.null(rng)) rng$y else NULL)
      )
    })

    # Keep axis range updated between full redraws.
    observe({
      rng <- filtered_range()
      if (is.null(rng)) return()
      plotlyProxy("loss_plot", session) |>
        plotlyProxyInvoke("relayout", list(
          "xaxis.range[0]" = rng$x[[1L]], "xaxis.range[1]" = rng$x[[2L]],
          "yaxis.range[0]" = rng$y[[1L]], "yaxis.range[1]" = rng$y[[2L]]
        ))
    })

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
        flat <- unlist(lapply(METRICS, function(m) by_restart(lm, m)), recursive = FALSE)
        plotlyProxy("loss_plot", session) |>
          plotlyProxyInvoke("restyle",
            list(x = lapply(flat, `[[`, "x"), y = lapply(flat, `[[`, "y")),
            as.list(seq_along(flat) - 1L))
      }
    })

    # ---------------------------------------------------------------------------
    # Status
    # ---------------------------------------------------------------------------
    output$status <- renderText({
      exp_id <- suppressWarnings(as.integer(input$experiment_id))
      lm     <- db_data()
      bl     <- lm[lm$metric == "best_loss" & lm$restart >= 0L, ]
      if (length(exp_id) == 0L || is.na(exp_id) || nrow(bl) == 0L) return("Waiting for data...")

      d  <- experiments()
      ep <- d$params[d$params$experiment_id == exp_id, ]
      all_keys  <- sort(unique(d$params$key))
      param_str <- if (length(all_keys) == 0L) "(no params)" else {
        parts <- sapply(all_keys, function(k) {
          v <- ep$value[ep$key == k]
          sprintf("%s=%s", k, if (length(v) == 0L) "unset" else v)
        })
        paste(parts, collapse = "\n")
      }

      latest_iter <- max(bl$iter)
      latest_loss <- min(bl$value[bl$iter == latest_iter])
      elapsed_s   <- as.numeric(Sys.time()) -
        tryCatch(
          dbGetQuery(con,
            "SELECT MIN(timestamp) FROM iterations WHERE experiment_id = ?",
            list(exp_id))[[1L]],
          error = function(e) as.numeric(Sys.time())
        )
      elapsed_str <- if (elapsed_s < 60) {
        sprintf("%.0fs", elapsed_s)
      } else if (elapsed_s < 3600) {
        sprintf("%dm %02ds", floor(elapsed_s / 60), floor(elapsed_s) %% 60L)
      } else {
        sprintf("%dh %02dm", floor(elapsed_s / 3600), floor((elapsed_s %% 3600) / 60))
      }
      sprintf("%s\n\nIter:    %d\nLoss:    %.6f\nElapsed: %s",
              param_str, latest_iter, latest_loss, elapsed_str)
    })
  }
}

#' @export
run_dashboard <- function(db_path, host = "127.0.0.1", port = 8081L, ...) {
  runApp(shinyApp(ui = make_ui(), server = make_server(db_path)),
         host = host, port = port, ...)
}
