import torch
from diffusers import StableDiffusionXLPipeline
from ziplora_utils import insert_ziplora_to_unet

pretrained_model_name_or_path = "stabilityai/stable-diffusion-xl-base-1.0"
ziplora_name_or_path = "./models/ziplora_style2_r256_monster_toy_r32_2024-06-12_03-31-53/pytorch_lora_weights.safetensors"

# Define the prompt and the random generator
prompt = "a sbu toy in the crt style"

pipeline = StableDiffusionXLPipeline.from_pretrained(pretrained_model_name_or_path)
pipeline.unet = insert_ziplora_to_unet(pipeline.unet, ziplora_name_or_path)
pipeline.to(device="cuda", dtype=torch.float16)
image = pipeline(prompt=prompt, num_inference_steps=25).images[0]
image.save("generated/test_ziplora_style3_r256_monster_toy_r32_v1.png")