#!/bin/bash
#SBATCH --job-name=model_train
#SBATCH -t 8:00:00
#SBATCH --mem-per-gpu=64G
#SBATCH --output=./home/hice1/madewolu9/scratch/madewolu9/SOLID/SOLID/outputs/logs/slurm_%j.out
#SBATCH -n 12
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -C "A100-40GB|A100-80GB|H100|V100-16GB|V100-32GB|RTX6000|A40|L40S"

echo "launching model training..."

module load cuda/11.7

HYDRA_FULL_ERROR=1

source /home/hice1/madewolu9/scratch/madewolu9/SOLID/SOLID/.venv/bin/activate

cd /home/hice1/madewolu9/scratch/madewolu9/SOLID/SOLID

python -m scripts.baseline.train_rgb_baseline
