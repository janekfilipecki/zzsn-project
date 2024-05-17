from huggingface_hub import snapshot_download
from diffusers import DiffusionPipeline, StableDiffusionXLImg2ImgPipeline
import torch

# Set the paths
pretrained_model_name_or_path = "stabilityai/stable-diffusion-xl-base-1.0"
lora_weights_path = "./models/dreambooth_lora_sdxl_dog6/pytorch_lora_weights.safetensors"
output_dir = "./output"

# Load the base pipeline and apply the LoRA weights
pipe = DiffusionPipeline.from_pretrained(pretrained_model_name_or_path, torch_dtype=torch.float16)
pipe = pipe.to("cuda")
pipe.load_lora_weights(lora_weights_path)

# Load the refiner
refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
)
refiner.to("cuda")

# Define the prompt and the random generator
prompt = "a sks dog"
generator = torch.Generator("cuda").manual_seed(0)

# Perform inference with the base pipeline and refine the output
image = pipe(prompt=prompt, output_type="latent", generator=generator).images[0]
refined_image = refiner(prompt=prompt, image=image[None, :], generator=generator).images[0]
refined_image.save("test_sdxl_dog_simple.png")
