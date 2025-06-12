#!/bin/bash

#SBATCH --account=jhjin1
#SBATCH --job-name=evaluation
#SBATCH --mail-user=lhalice@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=standard
#SBATCH --mem=50GB
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --output=/home/lhalice/EIS_fit_ECM_with_ML/generate_data.log


python evaluation.py