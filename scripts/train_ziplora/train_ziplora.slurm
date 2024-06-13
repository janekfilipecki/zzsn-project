#!/bin/bash

#SBATCH --job-name=ziplora_train
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --time=360
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --nodes=1
#SBATCH --output=./logs/dreambooth_lora_sdxl/train_ziplora_sdxl_%j_%a.out

export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export LOG_DIR="./logs/tensorboard_logs"
export HF_HOME="/net/tscratch/people/plgas2000/.cache/huggingface"

# for subject
export LORA_PATH="./models/dreambooth_lora_sdxl_monster_toy_subject_r32_2024-06-12_02-14-22"
export INSTANCE_DIR="./dataset/dreambooth-main/dataset/dog6"
export PROMPT="a sbu toy"

# for style
export LORA_PATH2="./models/dreambooth_lora_sdxl_style2_r256_2024-06-12_02-10-52"
export INSTANCE_DIR2="./dataset/styles/style3-sketch"
export PROMPT2="a bicycle in the crt style"

# general
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
export OUTPUT_DIR="./models/ziplora_style2_r256_monster_toy_r32_$timestamp"
export VALID_PROMPT="a sbu toy in the crt style"


accelerate launch scripts/train_ziplora/train_dreambooth_ziplora_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --output_dir=$OUTPUT_DIR \
  --lora_name_or_path=$LORA_PATH \
  --instance_prompt="${PROMPT}" \
  --instance_data_dir=$INSTANCE_DIR \
  --lora_name_or_path_2=$LORA_PATH2 \
  --instance_prompt_2="${PROMPT2}" \
  --instance_data_dir_2=$INSTANCE_DIR2 \
  --resolution=1024 \
  --train_batch_size=1 \
  --learning_rate=5e-5 \
  --similarity_lambda=0.01 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=100 \
  --validation_prompt="${VALID_PROMPT}" \
  --validation_epochs=100 \
  --seed="100" \
  --mixed_precision="fp16" \
  --report_to="wandb" \
  --gradient_checkpointing \
  --use_8bit_adam \
