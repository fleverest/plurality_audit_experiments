# In this file we generate sequences of categorical observations and save them to disk.
# Later we use the sequences to compare the performance of different test martingales.
# At first, we generate under theta = q = c(7/16, 5/16, 4/16) to compare best-case
# performance for each martingale.
library(tidyverse)

set.seed(20260511)
q <- c(7 / 16, 5 / 16, 4 / 16)

seqs <- replicate(
  10000,
  sample(1:3, size = 200, replace = TRUE, prob = q)
)


source("grand_prix/martingales.R")

# Compare the martingales on each sequence.
ui <- UITest(q, 0.05)
im <- InfimumMartingale(q, 0.05)
dripr <- DirectRIPrSequenceTest(q, 0.05)

results3 <- apply(seqs, 2, function(seq) {
  reset(ui)
  reset(im)
  reset(dripr)
  suppressMessages({
    update(ui, seq)
    update(im, seq)
    update(dripr, seq)
  })
  cbind(
    ui = value(ui, Inf),
    im = value(im, Inf),
    dripr_corrected = value(dripr, Inf),
    dripr_uncorrected = dripr@state$history_Y
  )
})

simulation_data3 <- sapply(seq_len(10000), \(seq_idx) {
    seq_result <- results[, seq_idx]
    # Split vector into 4 equal-length segments (corresponding to the 3 martingales + uncorrected)
    seq_result_split <- split(seq_result, ceiling(seq_along(seq_result) / (length(seq_result) / 4)))
    names(seq_result_split) <- c("Universal Inference", "Infimum Martingale", "Corrected DirectRIPr", "Uncorrected DirectRIPr")
    # Plot each as a line plot, coloured by the segment name and labelled with the segment name
    tibble(
      step = rep(1:length(seq_result_split[[1]]), times = 4),
      value = unlist(seq_result_split),
      segment = rep(names(seq_result_split), each = length(seq_result_split[[1]])),
      seq_idx = seq_idx
    )
  },
  simplify = FALSE
) |>
  bind_rows()


simulation_data3 |>
  summarise(
    value = mean(value),
    .by = c("step", "segment")
  ) |>
  ggplot(aes(x = step, y = value, color = segment)) +
  geom_line() +
  theme_minimal() +
  labs(
    title = "Average Martingale Values Over Time",
    x = "Step",
    y = "Average Martingale Value",
    color = "Martingale Type"
  )




# K = 4 candidates

set.seed(20260512)
q <- c(7 / 16, 5 / 16, 3 / 16, 1 / 16)

seqs <- replicate(
  10000,
  sample(1:4, size = 100, replace = TRUE, prob = q)
)

ui <- UITest(q, 0.05)
im <- InfimumMartingale(q, 0.05)
dripr <- DirectRIPrSequenceTest(q, 0.05, results_dir = "results/k4/")


results4 <- apply(seqs, 2, function(seq) {
  reset(ui)
  reset(im)
  reset(dripr)
  suppressMessages({
    update(ui, seq)
    update(im, seq)
    update(dripr, seq)
  })
  cbind(
    ui = value(ui, Inf),
    im = value(im, Inf),
    dripr_corrected = value(dripr, Inf),
    dripr_uncorrected = dripr@state$history_Y
  )
})

simulation_data4 <- sapply(seq_len(10000), \(seq_idx) {
    seq_result <- results4[, seq_idx]
    # Split vector into 4 equal-length segments (corresponding to the 3 martingales + uncorrected)
    seq_result_split <- split(seq_result, ceiling(seq_along(seq_result) / (length(seq_result) / 4)))
    names(seq_result_split) <- c("Universal Inference", "Infimum Martingale", "Corrected DirectRIPr", "Uncorrected DirectRIPr")
    # Plot each as a line plot, coloured by the segment name and labelled with the segment name
    tibble(
      step = rep(1:length(seq_result_split[[1]]), times = 4),
      value = unlist(seq_result_split),
      segment = rep(names(seq_result_split), each = length(seq_result_split[[1]])),
      seq_idx = seq_idx
    )
  },
  simplify = FALSE
) |>
  bind_rows()

batch2 <- BatchRIPr(q, 0.05, batch_size = 2, results_dir = "results/k4/")
batch5 <- BatchRIPr(q, 0.05, batch_size = 5, results_dir = "results/k4/")
batch10 <- BatchRIPr(q, 0.05, batch_size = 10, results_dir = "results/k4/")
batch20 <- BatchRIPr(q, 0.05, batch_size = 20, results_dir = "results/k4/")

results4_batch <- apply(seqs, 2, function(seq) {
  reset(batch2)
  reset(batch5)
  reset(batch10)
  reset(batch20)
  suppressMessages({
    update(batch2, seq)
    update(batch5, seq)
    update(batch10, seq)
    update(batch20, seq)
  })
  cbind(
    batch2 = value(batch2, Inf),
    batch5 = value(batch5, Inf),
    batch10 = value(batch10, Inf),
    batch20 = value(batch20, Inf)
  )
})

simulation_data4_batch <- sapply(seq_len(10000), \(seq_idx) {
    seq_result <- results4_batch[, seq_idx]
    # Split vector into 4 equal-length segments (corresponding to the 3 martingales + uncorrected)
    seq_result_split <- split(seq_result, ceiling(seq_along(seq_result) / (length(seq_result) / 4)))
    names(seq_result_split) <- c("BatchRIPr (batch size = 2)", "BatchRIPr (batch size = 5)", "BatchRIPr (batch size = 10)", "BatchRIPr (batch size = 20)")
    # Plot each as a line plot, coloured by the segment name and labelled with the segment name
    tibble(
      step = rep(1:length(seq_result_split[[1]]), times = 4),
      value = unlist(seq_result_split),
      segment = rep(names(seq_result_split), each = length(seq_result_split[[1]])),
      seq_idx = seq_idx
    )
  },
  simplify = FALSE
) |>
  bind_rows()

simulation_data4_batch |>
  ggplot(aes(x = step, y = value, color = segment)) +
  geom_line() +
  theme_minimal() +
  labs(
    title = "Average Martingale Values Over Time (K=4)",
    x = "Step",
    y = "Average Martingale Value",
    color = "Martingale Type"
  )

bind_rows(simulation_data4, simulation_data4_batch) |>
  summarise(
    value = mean(value),
    .by = c("step", "segment")
  ) |>
  ggplot(aes(x = step, y = value, color = segment)) +
  geom_line() +
  theme_minimal() +
  labs(
    title = "Average Martingale Values Over Time",
    x = "Step",
    y = "Average Martingale Value",
    color = "Martingale Type"
  )
