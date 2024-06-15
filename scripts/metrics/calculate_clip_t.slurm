#!/bin/bash

#SBATCH --job-name=calculate_clip_t
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --time=30
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --nodes=1
#SBATCH --output=./logs/metrics_logs/clip_t_%j_%a.out

export MODEL_NAME="openai/clip-vit-base-patch32"
export GENERATED_DIR="./generated/dog6_no_ppl"
export LOG_DIR="./logs/metrics_logs"
export HF_HOME="/net/tscratch/people/plgfilipeckijan/.cache/huggingface"

python scripts/metrics/calculate_clip_t.py \
    --generated_dir=$GENERATED_DIR \
    --model_name=$MODEL_NAME \
    --log_dir=$LOG_DIR
