#!/bin/bash

#SBATCH --time=120:00:00
#SBATCH --mem=5G
#SBATCH --partition=cpu_long

module purge
module load anaconda3/2021.05/gcc-9.2.0

# Activate anaconda environment
source activate cedric_bs

# Launch command
python main.py --var1 "classe1" --var2 "HLA-A" --input "data/exemple.csv" --run-name "exemple"