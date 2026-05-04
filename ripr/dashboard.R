box::use(
  shiny[
    shinyApp, runApp,
    reactive, reactiveVal, reactivePoll, observe, observeEvent, isolate,
    radioButtons, checkboxInput, selectInput, updateSelectInput, actionButton,
    renderUI, uiOutput,
    renderText, verbatimTextOutput,
    tagList, tags, req, conditionalPanel, HTML
  ],
  plotly[plot_ly, add_trace, layout, subplot, plotlyOutput, renderPlotly,
         event_data, event_register],
  bslib[page_sidebar, sidebar, layout_columns, card, card_header,
        navset_card_tab, nav_panel],
  DBI[dbConnect, dbDisconnect, dbExecute, dbGetQuery],
  RSQLite[SQLite]
)

METRICS    <- c("best_loss", "cur_loss", "best_max_exp", "cur_max_exp")
COLORS     <- c(best_loss = "#2c7bb6", cur_loss = "#2c7bb6",
                best_max_exp = "#d7191c", cur_max_exp = "#d7191c")
LABELS     <- c(best_loss = "best_loss", cur_loss = "cur_loss",
                best_max_exp = "best_max_exp - 1", cur_max_exp = "cur_max_exp - 1")
TRANSFORMS <- list(best_loss = identity, cur_loss = identity,
                   best_max_exp = function(y) y - 1,
                   cur_max_exp  = function(y) y - 1)

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
      navset_card_tab(
        id     = "plot_tabs",
        header = tagList(
          tags$style(HTML("
            .card > .card-body:not(.tab-content) { flex-grow: 0 !important; }
            .tab-header-cbs .mb-3 { margin-bottom: 0 }
          ")),
          conditionalPanel(
            "input.plot_tabs !== 'Weights'",
            tags$div(
              class = "tab-header-cbs d-flex gap-4 px-3 pt-1",
              checkboxInput("view_min",  "Min across restarts", value = TRUE),
              checkboxInput("show_best", "Best achieved",       value = TRUE)
            )
          )
        ),
        nav_panel("Loss",            plotlyOutput("loss_plot",    height = "420px")),
        nav_panel("Max expectation", plotlyOutput("max_exp_plot", height = "420px")),
        nav_panel("Weights",
          tags$div(
            class = "px-3 pt-2 pb-1",
            radioButtons("weights_view", NULL, inline = TRUE,
                         choices  = c("All restarts" = "all", "Mean" = "mean", "Best restart" = "best"),
                         selected = "all")
          ),
          plotlyOutput("weights_plot", height = "380px")
        ),
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
        tryCatch(dbGetQuery(con, "SELECT MAX(rowid) FROM iterations")[[1L]],
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
      all_keys    <- sort(unique(d$params$key))
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

    # Shared x range: extends as data comes in (depends on db_poll).
    filtered_x_range <- reactive({
      db_poll()
      fe  <- filtered_experiments()
      if (nrow(fe) == 0L) return(NULL)
      ids <- paste(fe$id, collapse = ",")
      res <- tryCatch(
        dbGetQuery(con, sprintf(
          "SELECT MIN(iter) AS lo, MAX(iter) AS hi FROM iterations
           WHERE experiment_id IN (%s)", ids)),
        error = function(e) NULL
      )
      if (is.null(res) || is.na(res$lo)) return(NULL)
      c(res$lo, res$hi)
    })
    filtered_x_range_val <- reactiveVal(NULL)
    observe({
      v <- filtered_x_range()
      if (!identical(v, isolate(filtered_x_range_val()))) filtered_x_range_val(v)
    })

    db_data <- reactive({
      db_poll()
      exp_id <- suppressWarnings(as.integer(input$experiment_id))
      if (length(exp_id) == 0L || is.na(exp_id)) return(empty_lm())
      tryCatch(
        dbGetQuery(con,
          "SELECT iter, restart, metric, value FROM loss_metrics
           WHERE experiment_id = ?
             AND metric IN ('best_loss', 'cur_loss', 'best_max_exp', 'cur_max_exp')
           ORDER BY iter, restart",
          list(exp_id)),
        error = function(e) empty_lm()
      )
    })

    weights_data <- reactive({
      db_poll()
      exp_id <- suppressWarnings(as.integer(input$experiment_id))
      if (length(exp_id) == 0L || is.na(exp_id)) return(NULL)
      type_filter <- if (input$show_best) "best_weight" else "cur_weight"
      tryCatch(
        dbGetQuery(con,
          "SELECT restart, component, value FROM profiles
           WHERE experiment_id = ?
             AND type = ?
             AND iter = (SELECT MAX(iter) FROM profiles WHERE experiment_id = ?)
           ORDER BY restart, component",
          list(exp_id, type_filter, exp_id)),
        error = function(e) NULL
      )
    })

    # Creates all reactive/observer/render infrastructure for one plot tab.
    # Captured from enclosing scope: session, input, output, con,
    #   filtered_experiments, filtered_x_range_val, db_data.
    make_plot_server <- function(plot_id, tab_name, metric_fn, y_metrics) {

      # Per-plot y range — fixed at filter time, no db_poll dependency.
      filtered_y_range <- reactive({
        fe  <- filtered_experiments()
        if (nrow(fe) == 0L) return(NULL)
        ids <- paste(fe$id, collapse = ",")
        lm  <- tryCatch(
          dbGetQuery(con, sprintf(
            "SELECT metric, value FROM loss_metrics
             WHERE experiment_id IN (%s)
               AND metric IN (%s)
               AND restart >= 0",
            ids,
            paste(sprintf("'%s'", y_metrics), collapse = ","))),
          error = function(e) NULL
        )
        if (is.null(lm) || nrow(lm) == 0L) return(NULL)
        y_vals <- unlist(lapply(y_metrics, function(m) {
          v <- TRANSFORMS[[m]](lm$value[lm$metric == m])
          v[is.finite(v) & v > 0]
        }))
        if (length(y_vals) == 0L) return(NULL)
        y_log <- log10(range(y_vals))
        pad   <- diff(y_log) * 0.1
        c(y_log[[1L]] - pad, y_log[[2L]] + pad)
      })
      filtered_y_range_val <- reactiveVal(NULL)
      observe({
        v <- filtered_y_range()
        if (!identical(v, isolate(filtered_y_range_val()))) filtered_y_range_val(v)
      })

      # Per-axis zoom state.
      user_zoom_x   <- reactiveVal(NULL)
      user_zoom_y   <- reactiveVal(NULL)
      last_manual_x <- NULL
      last_manual_y <- NULL
      zoom_active   <- FALSE

      observeEvent(filtered_y_range_val(), {
        user_zoom_x(NULL); user_zoom_y(NULL)
        last_manual_x <<- NULL; last_manual_y <<- NULL
        zoom_active   <<- FALSE
      }, ignoreInit = TRUE, ignoreNULL = TRUE)

      observe({
        req(isTRUE(input$plot_tabs == tab_name))
        rly <- event_data("plotly_relayout", source = plot_id)
        if (is.null(rly)) return()
        has_x      <- !is.null(rly[["xaxis.range[0]"]])
        has_y      <- !is.null(rly[["yaxis.range[0]"]])
        is_autorng <- isTRUE(rly[["xaxis.autorange"]]) || isTRUE(rly[["yaxis.autorange"]])
        if (is_autorng) {
          if (zoom_active) {
            user_zoom_x(NULL); user_zoom_y(NULL)
            zoom_active <<- FALSE
          } else if (!is.null(last_manual_x) || !is.null(last_manual_y)) {
            user_zoom_x(last_manual_x); user_zoom_y(last_manual_y)
            zoom_active <<- TRUE
          }
        } else if (has_x || has_y) {
          if (has_x) {
            new_x <- c(rly[["xaxis.range[0]"]], rly[["xaxis.range[1]"]])
            user_zoom_x(new_x); last_manual_x <<- new_x
          }
          if (has_y) {
            new_y <- c(rly[["yaxis.range[0]"]], rly[["yaxis.range[1]"]])
            user_zoom_y(new_y); last_manual_y <<- new_y
          }
          zoom_active <<- TRUE
        }
      })

      # Redraws on new data, tab switch, metric change, or view change.
      # Shiny uses Plotly.react() for output updates, so this is an efficient
      # diff rather than a full destroy+create. Zoom state is preserved via
      # reactiveVals read with isolate().
      output[[plot_id]] <- renderPlotly({
        input$plot_tabs
        lm   <- db_data()
        m    <- metric_fn()
        view <- if (input$view_min) "min" else "all"

        rng_x <- if (!is.null(isolate(user_zoom_x()))) isolate(user_zoom_x())
                 else isolate(filtered_x_range_val())
        rng_y <- if (!is.null(isolate(user_zoom_y()))) isolate(user_zoom_y())
                 else isolate(filtered_y_range_val())

        p <- plot_ly(source = plot_id)
        if (view == "min") {
          td <- min_by_iter(lm, m)
          p  <- add_trace(p, x = td$x, y = td$y,
                          name = LABELS[[m]], type = "scatter", mode = "lines",
                          line = list(color = COLORS[[m]]))
        } else {
          rts <- by_restart(lm, m)
          if (length(rts) == 0L) {
            p <- add_trace(p, x = NA_real_, y = NA_real_,
                           name = LABELS[[m]], type = "scatter", mode = "lines",
                           line = list(color = COLORS[[m]]))
          } else {
            for (ri in seq_along(rts)) {
              p <- add_trace(p, x = rts[[ri]]$x, y = rts[[ri]]$y,
                             name = LABELS[[m]], showlegend = (ri == 1L),
                             type = "scatter", mode = "lines",
                             line = list(color = COLORS[[m]], width = 1))
            }
          }
        }
        p |>
          layout(
            xaxis = list(title = "Iteration", range = rng_x),
            yaxis = list(title = "Value", type = "log", range = rng_y)
          ) |>
          event_register("plotly_relayout")
      })
    }

    make_plot_server(
      plot_id   = "loss_plot",
      tab_name  = "Loss",
      metric_fn = reactive(if (input$show_best) "best_loss" else "cur_loss"),
      y_metrics = c("best_loss", "cur_loss")
    )
    make_plot_server(
      plot_id   = "max_exp_plot",
      tab_name  = "Max expectation",
      metric_fn = reactive(if (input$show_best) "best_max_exp" else "cur_max_exp"),
      y_metrics = c("best_max_exp", "cur_max_exp")
    )

    output$weights_plot <- renderPlotly({
      input$plot_tabs
      wd   <- weights_data()
      view <- input$weights_view
      if (is.null(wd) || nrow(wd) == 0L) {
        return(plot_ly(x = numeric(0), y = numeric(0), type = "bar"))
      }

      restarts <- sort(unique(wd$restart))
      bar_color <- "#2c7bb6"

      no_yaxis <- list(visible = FALSE)

      if (view == "mean") {
        mean_w <- tapply(wd$value, wd$component, mean)
        plot_ly() |>
          add_trace(x = as.integer(names(mean_w)), y = as.numeric(mean_w),
                    type = "bar", marker = list(color = bar_color),
                    showlegend = FALSE) |>
          layout(xaxis = list(title = "Component"), yaxis = no_yaxis)

      } else if (view == "best") {
        lm     <- isolate(db_data())
        metric <- if (isolate(input$show_best)) "best_loss" else "cur_loss"
        sub    <- lm[lm$metric == metric & lm$restart >= 0L, ]
        best_r <- if (nrow(sub) > 0L) {
          li <- max(sub$iter)
          sl <- sub[sub$iter == li, ]
          sl$restart[which.min(sl$value)]
        } else restarts[[1L]]
        wd_r <- wd[wd$restart == best_r, ]
        plot_ly() |>
          add_trace(x = wd_r$component, y = wd_r$value,
                    type = "bar", marker = list(color = bar_color),
                    showlegend = FALSE) |>
          layout(xaxis = list(title = "Component"), yaxis = no_yaxis)

      } else {
        plots <- lapply(restarts, function(r) {
          wd_r <- wd[wd$restart == r, ]
          plot_ly() |>
            add_trace(x = wd_r$component, y = wd_r$value,
                      type = "bar", marker = list(color = bar_color),
                      showlegend = FALSE) |>
            layout(
              yaxis = no_yaxis,
              annotations = list(list(
                x = 0, y = 1, xref = "paper", yref = "paper",
                text = paste0("<b>R", r, "</b>"), showarrow = FALSE,
                xanchor = "left", yanchor = "bottom", font = list(size = 10)
              ))
            )
        })
        subplot(plots, nrows = length(restarts), shareX = TRUE, margin = 0.06) |>
          layout(xaxis = list(title = "Component"))
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
