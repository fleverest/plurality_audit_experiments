# Reproducing the environment on M3

Here are some of the steps I had to follow to get up and running on M3.

1. Load R and required modules for M3:

```bash
module load r/4.4.1 cuda/12.6
```

2. Install R package dependencies from source:

```
R -e "renv::restore(rebuild = TRUE); torch::install_torch()"
```
