local({
  torch_home <- file.path(getwd(), ".torch")
  dir.create(torch_home, showWarnings = FALSE, recursive = TRUE)
  Sys.setenv(TORCH_HOME = torch_home)
  # Prevent OpenBLAS and LibTorch from racing over OpenMP threads, which
  # causes segfaults in BLAS calls (e.g. solve()) after torch is loaded.
  Sys.setenv(OMP_NUM_THREADS = "1", OPENBLAS_NUM_THREADS = "1")
  options(box.path = getwd())
})
source("renv/activate.R")
