
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path
import json
import argparse
import numpy as np
from tqdm import tqdm
import torch
from pathlib import Path
from PIL import Image
from llava.mm_utils import  process_images
import math
def main(args):
    """
    json format:
    [
    {
        "path": "video_needle_haystack/data/query_images/image1.jpg",
        "prompt": "Find the frame with the image of Selenium tablets. What is the color of the bottle cap?\nAnswer the question using a single word or phrase."
    }
    ]
    """
    model_name = get_model_name_from_path(args.model)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model, None, model_name, load_8bit=False,device_map="cuda:0")
    model.config.image_aspect_ratio = "pad"
    model.config.mm_patch_merge_type="flat"
    # load json
    if args.add_newline_token:
        newline_token_embeddong = model.model.image_newline
    with open(args.image_json, "r") as f:
        data = json.load(f)
    for instance in data:
        prompt = instance["prompt"]
        image_path = instance["path"]
        # load image
        image = Image.open(image_path).convert("RGB")
        image = process_images([image], image_processor, model.config).half()
        image_features = model.encode_images(image)
        if args.pooling_size != 0:
            B, _, F = image_features.shape
            image_features_spatial = image_features.view(B, int(math.sqrt(_)), int(math.sqrt(_)), F).permute(0, 3, 1, 2) # B, F, 24, 24
            image_features_spatial_pool = torch.nn.functional.avg_pool2d(image_features_spatial, args.pooling_size, args.pooling_size) # B, F, 12, 12
            image_features = image_features_spatial_pool.flatten(2).transpose(1, 2).contiguous() # B, 144, F
            if args.add_newline_token:
                image_features = torch.cat([image_features, newline_token_embeddong.unsqueeze(0).expand(image_features.shape[0], 1, -1)], dim=1)
        image_features = image_features.squeeze(0)
        print(image_features.shape)
        torch.save(image_features, f"{args.output_dir}/{Path(image_path).stem}.pt")
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="output/LLaVA-NeXT-Video-7B-Vicuna")
    parser.add_argument("--image_json", type=str, default="vision_niah/data/query_images/prompts.json")
    parser.add_argument("--output_dir", type=str, default="video_needle_haystack/data/needle_vicuna_embeddings")
    parser.add_argument("--pooling_size", type=int, default=0)
    parser.add_argument("--add_newline_token", action="store_true")
    args = parser.parse_args()
    main(args)
