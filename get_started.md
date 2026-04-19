# Reproducing the environment on M3

Here are some of the steps I had to follow to get up and running on M3.

## Installing R 4.5.3 and dependencies

```bash
module load r cuda/12.6
R -e "renv::restore(); torch::install_torch()"
```
