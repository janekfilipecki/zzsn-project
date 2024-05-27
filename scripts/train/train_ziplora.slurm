#!/bin/bash

#SBATCH --job-name=train_dreambooth_lora
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --time=360
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --nodes=1
#SBATCH --output=./logs/dreambooth_lora_sdxl/train_dreambooth_lora_sdxl_%j_%a.out

export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"

# for subject
export LORA_PATH="mkshing/lora-sdxl-dog"
export INSTANCE_DIR="dog"
export PROMPT="a sbu dog"

# for style
export LORA_PATH2="mkshing/lora-sdxl-waterpainting"
export INSTANCE_DIR2="waterpainting"
export PROMPT2="a cat of in szn style"

# general 
export OUTPUT_DIR="ziplora-sdxl-dog-waterpainting"
export VALID_PROMPT="a sbu dog in szn style"


accelerate launch train_dreambooth_ziplora_sdxl.py \
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
  --validation_epochs=10 \
  --seed="0" \
  --mixed_precision="fp16" \
  --report_to="wandb" \
  --gradient_checkpointing \
  --use_8bit_adam \
  --enable_xformers_memory_efficient_attention \
