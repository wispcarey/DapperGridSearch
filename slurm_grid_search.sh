#!/bin/bash

# Submit this script with: sbatch <this-filename>

#SBATCH --time=2-00:00:00   # walltime (2 days)
#SBATCH --ntasks=1          # number of tasks (1 task)
#SBATCH --cpus-per-task=64  # number of CPU cores for the task
#SBATCH --nodes=1           # number of nodes (1 node)
#SBATCH -J "grid-search-job"  # job name
#SBATCH --mail-user=bhchen@caltech.edu   # email address

# Notify at the beginning, end of job and on failure.
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

## Optional: Uncomment the following to specify the partition
## /SBATCH -p general

## Optional: Uncomment the following lines for custom output/error files
## /SBATCH -o slurm.%j.out # STDOUT
## /SBATCH -e slurm.%j.err # STDERR

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
python grid_search.py
