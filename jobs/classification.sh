#!/bin/bash

#SBATCH --account=jhjin1
#SBATCH --job-name=Classification_gpu
#SBATCH --mail-user=lhalice@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=16GB
#SBATCH --time=4:00:00
#SBATCH --output=/home/lhalice/EIS_fit_ECM_with_ML/classification.log


python Classification_ECM.py 