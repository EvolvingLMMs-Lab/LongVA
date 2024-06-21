
from longva.model.builder import load_pretrained_model
from longva.mm_utils import tokenizer_image_token, get_model_name_from_path
import json
import argparse
import numpy as np
from tqdm import tqdm
import torch
from pathlib import Path
from PIL import Image
from datasets import load_dataset
from longva.mm_utils import  process_images
import math
def main(args):
    model_name = "llava_qwen"
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model, None, model_name, load_8bit=False,device_map="cuda:0")
    model.config.image_aspect_ratio = "pad"
    model.config.mm_patch_merge_type="flat"
    dataset = load_dataset(args.needle_dataset)["test"]
    for index, instance in enumerate(dataset):
        image = instance["image"].convert("RGB")
        image = process_images([image], image_processor, model.config).half()
        image_features = model.encode_images(image)
        if args.pooling_size != 0:
            B, _, F = image_features.shape
            image_features_spatial = image_features.view(B, int(math.sqrt(_)), int(math.sqrt(_)), F).permute(0, 3, 1, 2) # B, F, 24, 24
            image_features_spatial_pool = torch.nn.functional.avg_pool2d(image_features_spatial, args.pooling_size, args.pooling_size) # B, F, 12, 12
            image_features = image_features_spatial_pool.flatten(2).transpose(1, 2).contiguous() # B, 144, F
        image_features = image_features.squeeze(0)
        torch.save(image_features, f"{args.output_dir}/{index}.pt")
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="output/LLaVA-NeXT-Video-7B-Vicuna")
    parser.add_argument("--needle_dataset", type=str, default="LongVa/longva_needles2")
    parser.add_argument("--output_dir", type=str, default="video_needle_haystack/data/needle_vicuna_embeddings")
    parser.add_argument("--pooling_size", type=int, default=0)
    args = parser.parse_args()
    main(args)
