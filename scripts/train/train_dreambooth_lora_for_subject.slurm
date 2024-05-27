#!/bin/bash

#SBATCH --job-name=train_loras_for_subject_lower_ppl
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --time=360
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --nodes=1
#SBATCH --output=./logs/dreambooth_lora_sdxl/train_lora_for_subject_lower_ppl_%j_%a.out

export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"

timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
export OUTPUT_DIR="./models/dreambooth_lora_sdxl_dog3_subject_$timestamp"
export INSTANCE_DIR="./dataset/dreambooth-main/dataset/dog3"
export CLASS_DIR="./dataset/dreambooth_generated/dog"
export PROMPT="a photo of sks dog"
export VALID_PROMPT="a photo of sks dog in a bucket"
export HF_HOME="/net/tscratch/people/plgas2000/.cache/huggingface"
export LOG_DIR="./logs/tensorboard_logs"
export VAE_PATH="madebyollin/sdxl-vae-fp16-fix"

accelerate launch scripts/train/train_dreambooth_lora_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --class_data_dir=$CLASS_DIR \
  --class_prompt="a photo of a dog" \
  --instance_prompt="${PROMPT}" \
  --resolution=1024 \
  --with_prior_preservation --prior_loss_weight=0.1 \
  --train_batch_size=1 \
  --learning_rate=5e-5 \
  --report_to="tensorboard" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=800 \
  --validation_prompt="${VALID_PROMPT}" \
  --validation_epochs=50 \
  --mixed_precision="fp16" \
  --enable_xformers_memory_efficient_attention \
  --logging_dir=$LOG_DIR \
  --gradient_checkpointing \
  --use_8bit_adam \