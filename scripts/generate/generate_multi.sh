#! /bin/bash
sbatch scripts/generate/infere.slurm scripts/generate/generate_sdxl.py \
    --weights_path="models/dreambooth_lora_sdxl_can_2024-06-15_18-29-12/pytorch_lora_weights.safetensors" \
    --output_dir="generated/can_3000" \
    --prompt_input="prompts/can_prompts.txt" \
    --num_generations=1 \
    --num_inference_steps=30