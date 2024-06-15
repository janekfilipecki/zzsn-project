#! /bin/bash
sbatch scripts/generate/infere.slurm scripts/generate/generate_sdxl.py \
    --weights_path="models/dreambooth_lora_sdxl_no_ppl_dog6_2024-06-13_07-39-19/pytorch_lora_weights.safetensors" \
    --output_dir="generated/dog6" \
    --prompt_input="A photo of sks dog in a bucket" \
    --num_generations=5 \
    --num_inference_steps=50