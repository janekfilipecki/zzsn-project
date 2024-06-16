#! /bin/bash
sbatch scripts/generate/infere.slurm scripts/generate/generate_sdxl.py \
    --weights_path="models/dreambooth_lora_sdxl_no_ppl_can_2024-06-15_18-20-38/pytorch_lora_weights.safetensors" \
    --output_dir="generated/can_no_ppl_no_target" \
    --prompt_input="a photo of a can" \
    --num_generations=5 \
    --num_inference_steps=50