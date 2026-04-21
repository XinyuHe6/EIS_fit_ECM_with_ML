#!/bin/bash

#SBATCH --account=jhjin1

#SBATCH --job-name=Classification_gpu

#SBATCH --mail-user=xyhe@umich.edu

#SBATCH --mail-type=BEGIN,END,FAIL

#SBATCH --nodes=1

#SBATCH --partition=gpu

#SBATCH --gpus=1

#SBATCH --mem-per-gpu=16GB

#SBATCH --time=4:00:00

#SBATCH --output=/home/xyhe/EIS_fit_ECM_with_ML/classification.log



module purge

module load cuda/11.2.2
module load cudnn/11.2-v8.1.1

module load python3.9-anaconda/2021.11

source "$(conda info --base)/etc/profile.d/conda.sh"

# conda activate tf27



conda run -n tf27 python -c "import sys; print('PY', sys.executable); import pandas as pd; print('pandas ok')"

conda run -n tf27 python -c "import tensorflow as tf; print('TF', tf.__version__); print(tf.config.list_physical_devices('GPU'))"



conda run -n tf27 python Classification_ECM.py --neglectable-fit-trials 1 "$@"
