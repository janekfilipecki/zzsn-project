from diffusers import DiffusionPipeline, StableDiffusionXLImg2ImgPipeline
import torch, argparse
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--weights_path", required=True)
parser.add_argument("--output_dir", required=True)
parser.add_argument("--prompt", required=True)
parser.add_argument("--num_inference_steps", type=int, required=True)

args = parser.parse_args()

# Set the paths
pretrained_model_name_or_path = "stabilityai/stable-diffusion-xl-base-1.0"
lora_weights_path = args.weights_path
output_dir = args.output_dir
prompt = args.prompt
num_inference_steps = args.num_inference_steps


# Load the base pipeline and apply the LoRA weights
pipe = DiffusionPipeline.from_pretrained(pretrained_model_name_or_path, torch_dtype=torch.float16)
pipe = pipe.to("cuda")
pipe.load_lora_weights(lora_weights_path)

# Perform inference with the base pipeline
image = pipe(prompt=prompt, num_inference_steps=num_inference_steps).images[0]
image.save(output_dir + "/" + prompt.replace(" ", "_") + "_" + str(datetime.now()) + ".png")
