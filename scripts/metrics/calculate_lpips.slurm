#!/bin/bash

#SBATCH --job-name=calculate_lpips
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --time=30
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --nodes=1
#SBATCH --output=./logs/metrics_logs/lpips_%j_%a.out

export ORIGINAL_DIR="./dataset/dreambooth-main/dataset/dog6"
export GENERATED_DIR="./generated/dog6"
export LOG_DIR="./logs/metrics_logs"
export HF_HOME="/net/tscratch/people/plgfilipeckijan/.cache/huggingface"

python scripts/metrics/calculate_lpips.py \
  --original_dir=$ORIGINAL_DIR \
  --generated_dir=$GENERATED_DIR \
  --log_dir=$LOG_DIR
