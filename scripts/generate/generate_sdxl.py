from huggingface_hub import snapshot_download
from diffusers import DiffusionPipeline, StableDiffusionXLImg2ImgPipeline
import torch

# Set the paths
pretrained_model_name_or_path = "stabilityai/stable-diffusion-xl-base-1.0"
lora_weights_path = "../../models/dreambooth_lora_sdxl_dog6_2024-05-21_15-46-31/pytorch_lora_weights.safetensors"
output_dir = "../../generated"

# Load the base pipeline and apply the LoRA weights
pipe = DiffusionPipeline.from_pretrained(pretrained_model_name_or_path, torch_dtype=torch.float16)
pipe = pipe.to("cuda")
pipe.load_lora_weights(lora_weights_path)

# Define the prompt and the random generator
prompt = "sks dog in a bucket"

# Perform inference with the base pipeline
image = pipe(prompt=prompt, num_inference_steps=25).images[0]
image.save("generated/test_sdxl_another.png")
