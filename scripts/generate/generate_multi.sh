#! /bin/bash
sbatch scripts/generate/infere.slurm scripts/generate/generate_sdxl.py \
    --weights_path="models/dreambooth_lora_sdxl_berry_bowl_2024-05-27_18-44-03/pytorch_lora_weights.safetensors" \
    --output_dir="generated/berry_bowl" \
    --prompt_input="prompts/bowl_prompts.txt" \
    --num_generations=1 \
    --num_inference_steps=100