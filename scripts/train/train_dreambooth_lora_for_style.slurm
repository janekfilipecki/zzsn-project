#!/bin/bash

#SBATCH --job-name=train_loras_for_style_lower_ppl
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --time=360
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --nodes=1
#SBATCH --output=./logs/dreambooth_lora_sdxl/train_lora_for_style_lower_ppl_%j_%a.out

export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"

timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
export OUTPUT_DIR="./models/dreambooth_lora_sdxl_style1_cartoon_$timestamp"
export INSTANCE_DIR="./dataset/styles/style1-cartoon"
export PROMPT="a cat of in crt style"
export VALID_PROMPT="a man in crt style"
export HF_HOME="/net/tscratch/people/plgas2000/.cache/huggingface"
export LOG_DIR="./logs/tensorboard_logs"
export VAE_PATH="madebyollin/sdxl-vae-fp16-fix"

accelerate launch scripts/train/train_dreambooth_lora_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --class_data_dir=None \
  --class_prompt=None \
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