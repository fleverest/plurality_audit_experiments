library(targets)
library(crew)
library(crew.cluster)
library(nanonext)


# Second IP address is reachable from other compute nodes on M3
is_m3 <- grepl("m3", Sys.info()[["nodename"]])

cpu_controller <- if (!is_m3) {
  crew_controller_local(name = "cpu", workers = 4L)
} else {
  crew_controller_slurm(
    name = "cpu",
    host = ip_addr()[2L],
    workers = 95L, # My QOS allows 100 jobs, but leave some headroom for the controller, plus an interactive job or two.
    seconds_idle = 120L,
    options_cluster = crew_options_slurm(
      partition = "comp", # your CPU partition
      cpus_per_task = 1L,
      memory_gigabytes_per_cpu = 10, # MaxTRESPU is 1TB total
      time_minutes = 300L, # 5 hours; MaxWall is 7 days
      log_output = "logs/crew_cpu_%A_%a.out",
      log_error = "logs/crew_cpu_%A_%a.err",
      script_lines = "module load r"
    )
  )
}

tar_option_set(
  seed = 20260516L,
  controller = cpu_controller,
  resources = tar_resources(
    crew = tar_resources_crew(controller = "cpu")
  )
)


source("experiments/compare_algorithms.R")
source("experiments/grand_prix.R")

c(
  algo_comparison_plan()
  # grand_prix_plan()
)
