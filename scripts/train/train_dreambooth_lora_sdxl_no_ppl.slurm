#!/bin/bash

#SBATCH --job-name=train_dreambooth_lora_sdxl_dog6
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --time=480
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --nodes=1
#SBATCH --output=./logs/dreambooth_lora_sdxl/train_dreambooth_lora_sdxl_no_ppl_dog6_%j_%a.out

export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export INSTANCE_DIR="./dataset/dreambooth-main/dataset/dog6"
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
export OUTPUT_DIR="./models/dreambooth_lora_sdxl_no_ppl_dog6_$timestamp"
export HF_HOME="/net/tscratch/people/plgfilipeckijan/.cache/huggingface"
export LOG_DIR="./logs/tensorboard_logs"
export VAE_PATH="madebyollin/sdxl-vae-fp16-fix"

accelerate launch scripts/train/train_dreambooth_lora_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --pretrained_vae_model_name_or_path=$VAE_PATH \
  --instance_prompt="a photo of sks dog" \
  --resolution=1024 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=200 \
  --max_train_steps=1000 \
  --logging_dir=$LOG_DIR \
  --report_to="tensorboard" \
  --mixed_precision="fp16"
