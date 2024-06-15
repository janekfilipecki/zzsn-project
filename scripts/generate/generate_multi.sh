#! /bin/bash
sbatch scripts/generate/infere.slurm scripts/generate/generate_sdxl.py \
    --weights_path="models/dreambooth_lora_sdxl_dog6_2024-06-13_10-31-11/pytorch_lora_weights.safetensors" \
    --output_dir="generated/dog6_maybe_fixed" \
    --prompt_input="prompts/dog6_prompts.txt" \
    --num_generations=1 \
    --num_inference_steps=30