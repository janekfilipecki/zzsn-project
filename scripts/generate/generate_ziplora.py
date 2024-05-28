import torch
from diffusers import StableDiffusionXLPipeline
from ziplora_utils import insert_ziplora_to_unet

pretrained_model_name_or_path = "stabilityai/stable-diffusion-xl-base-1.0"
ziplora_name_or_path = "./models/ziplora_style1_dog3_2024-05-28_03-29-59/pytorch_lora_weights.safetensors"

# Define the prompt and the random generator
prompt = "a photo of a cat in crt style"

pipeline = StableDiffusionXLPipeline.from_pretrained(pretrained_model_name_or_path)
pipeline.unet = insert_ziplora_to_unet(pipeline.unet, ziplora_name_or_path)
pipeline.to(device="cuda", dtype=torch.float16)
image = pipeline(prompt=prompt, num_inference_steps=25).images[0]
image.save("generated/test_ziplora_style1_cat.png")