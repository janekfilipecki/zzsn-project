import os
import argparse
import torch
from PIL import Image
import torchvision.transforms as transforms 
from torchmetrics.multimodal import CLIPScore
import logging

# Set up logging
def setup_logging(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, 'clip_t_metric.log')
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

def extract_description(image_name):
   description = " ".join(image_name.split("_")[1:-3])
   print(description)
   return description


def calculate_clip(generated_dir, model_name, log_dir):
    setup_logging(log_dir)
    metric = CLIPScore(model_name_or_path=model_name)

    transform = transforms.ToTensor() 

    # Get image paths
    image_paths = [os.path.join(generated_dir, img) for img in os.listdir(generated_dir) if img.endswith(('jpg', 'png'))]

    # Extract prompts from image file names
    images_with_prompts = [(img, extract_description(os.path.basename(img))) for img in image_paths]

    # Calculate CLIP scores
    scores = []
    for img_path, prompt in images_with_prompts:
        image = Image.open(img_path).convert("RGB")
        score = metric(transform(image), prompt)
        scores.append(score.item())

    avg_clip_score = sum(scores) / len(scores)
    logging.info(f'generated directory: {generated_dir}')
    print(f'generated directory: {generated_dir}')
    logging.info(f'CLIP_T Metric: {avg_clip_score}')
    print(f'Average CLIP_T Score: {avg_clip_score}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate the CLIP metric for generated images and their prompts.")
    parser.add_argument('--generated_dir', type=str, required=True, help='Path to the directory containing generated images.')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the pre-trained model from HuggingFace.')
    parser.add_argument('--log_dir', type=str, required=True, help='Directory to save the log file.')

    args = parser.parse_args()

    calculate_clip(
        generated_dir=args.generated_dir,
        model_name=args.model_name,
        log_dir=args.log_dir
    )
