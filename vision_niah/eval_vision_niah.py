import argparse
import gc
import sys
import torch
from transformers import AutoTokenizer
from transformers import LlamaForCausalLM
from tqdm import tqdm
from accelerate import Accelerator
import glob
import numpy as np
from tqdm import tqdm
import gc
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import pandas as pd
from pathlib import Path
import random
from easy_context import (
    prepare_seq_parallel_inputs,
    apply_seq_parallel_monkey_patch,
)
apply_seq_parallel_monkey_patch("zigzag_ring_attn", "llama")


SEED = 24242424
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

prompts = {
    "mistral": {
        "preprompt": "<s>[INST]",
        "postprompt": "Find the frame with the image of Selenium tablets. What is the color of the bottle cap?\nAnswer the question using a single word or phrase. [/INST]"
    },
    "vicuna": {
        "preprompt": "<s>A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER:",
        "postprompt": "\nFind the frame with the image of Selenium tablets. What is the color of the bottle cap?\nAnswer the question using a single word or phrase. ASSISTANT:"
    },
    #     "vicuna": {
    #     "preprompt": "<s>A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER:",
    #     "postprompt": "\nThe Chinese name of this movie is 孤注一掷. What is its English translation?\nAnswer the question using a single word or phrase. ASSISTANT: No"
    # }
    "llama3_color_easy": {
        "preprompt": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n",
        "postprompt": "\nFind the frame with the image of Selenium tablets. What is the color of the bottle cap?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nThe color of the bottle cap is",
        "answer" : " green"
    },
    "llama3_color": {
        "preprompt": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n",
        "postprompt": "\nFind the frame with the image of Selenium tablets. What is the color of the bottle cap?\nAnswer the question using a single word or phrase.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        "answer" : "Green"
    },
    "llama3_color_medecine_bottle": {
        "preprompt": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n",
        "postprompt": "\nInside the video, there is a frame of the selenium medicine bottle. Find that frame. What is the color of the bottle cap?\nAnswer the question using a single word or phrase.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        "answer" : "Green"
    },
    "llama3_ocr": {
        "preprompt": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n",
        "postprompt": "\nFind the frame with the image of Selenium tablets. How many mg does each tablet contain?\nAnswer the question using a single word or phrase.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        "answer": "200"
    },
    "llama3_tablets_count": {
        "preprompt": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n",
        "postprompt": "\nFind the frame with the image of Selenium tablets. How many pills does it contain?\nAnswer the question using a single word or phrase.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        "answer": "100"
    }
}
# \nAnswer the question using a single word or phrase.
# The color of the bottle cap is
# answer = "Yellow"

# answer = "more bet"
def eval_forward(accelerator, model, input_embeds, answer_embeds, pad_id, answer_ids, tokenizer):
    # first append answer_embeds to input_embeds
    prompt_length = input_embeds.shape[1]
    labels_length = answer_embeds.shape[1]
    input_embeds = torch.cat([input_embeds, answer_embeds], dim=1)
    # second pad input_embeds to the multiple of accelerator.num_processes
    pad_tensor = torch.tensor(
        [pad_id]
        * (
            (accelerator.num_processes * 2)
            - input_embeds.shape[1] % (accelerator.num_processes * 2)
        )
    ).unsqueeze(0).unsqueeze(-1).expand(-1, -1, input_embeds.shape[-1]).to(accelerator.device)
    input_embeds = torch.cat([input_embeds, pad_tensor], dim=1)
    position_ids = (
        torch.arange(input_embeds.shape[1]).unsqueeze(0).expand(input_embeds.shape[0], -1)
    ).to(accelerator.device)
    prepared = prepare_seq_parallel_inputs(
        "zigzag_ring_attn",
        input_embeds,
        position_ids,
        None,
        accelerator.process_index,
        accelerator.num_processes,
        accelerator.device,
    )
    local_input_embeds = prepared["local_input_ids"]
    local_position_ids = prepared["local_position_ids"]
    with torch.inference_mode():
        logits = model(
            inputs_embeds=local_input_embeds,
            position_ids=local_position_ids,
            use_cache=False,
        ).logits
        pred = logits.argmax(dim=-1)

    # gather all logits using accelerator.gather
    def undo_extract_local(gathered_value, world_size, dim=1):
        value_chunks = gathered_value.chunk(2 * world_size, dim=dim)
        reordered_chunks = [None] * (2 * world_size)
        for i in range(world_size):
            reordered_chunks[i] = value_chunks[i * 2]
            reordered_chunks[2 * world_size - i - 1] = value_chunks[i * 2 + 1]
        return torch.cat(reordered_chunks, dim=dim)

    correct = False

    gathered_logits = accelerator.gather(pred.squeeze(0)).unsqueeze(0)
    # undo extract local on the gathered logits
    pred = undo_extract_local(gathered_logits, accelerator.num_processes)
    pred = pred[:, prompt_length - 1 : prompt_length + labels_length - 1]
    # check if the logits are correct, extract argmax id
    # compare the predicted_ids with the labels
    correct = (pred == answer_ids.to(accelerator.device)).all()
    if  accelerator.is_main_process:
        print(
            "Predicted: ",
            tokenizer.decode(pred.squeeze().tolist()),
            "Answer: ",
            tokenizer.decode(answer_ids.squeeze().tolist()),
        )
    return int(correct)


def load_haystack(args, accelerator):
    haystack_embeddings = torch.load(f"{args.haystack_dir}/video_embeddings.pt").to(torch.bfloat16)
    # for file_path in tqdm(sorted(Path(args.haystack_dir).glob("*.pt"))[:args.max_frame_num], desc="Loading Haystack Embeddings...", disable=not accelerator.is_main_process):
    #     embeddings = torch.load(file_path, map_location="cpu").to(torch.bfloat16).unsqueeze(0)
    #     haystack_embeddings = embeddings if haystack_embeddings is None else torch.cat(
    #         [haystack_embeddings, embeddings], dim=0
    #     )
    return haystack_embeddings

def load_text_embeddings(str, tokenizer, model, accelerator, replace_double_newline=False): 
    token_ids = tokenizer.encode(str, return_tensors="pt")[:, 1:]
    def replace_double_newline_func(token_ids):
        # subsitute token id 271 to two 198]
        # for example:
        # from: tensor([[128000, 128006,   9125, 128007,    271,   2675,    527,    264,  11190, 4221,    323,  11376,  18328,     13]])
        # to: tensor([[128000, 128006,   9125, 128007,    198,    198,    2675,    527,    264,  11190, 4221,    323,  11376,  18328,     13]])
        # length will increase by number of 271
        double_newline_loc = (token_ids == 271).nonzero()[:, 1]
        double_newline_loc += torch.arange(len(double_newline_loc))
        if len(double_newline_loc) > 0:
            for loc in double_newline_loc:
                token_ids = torch.cat([token_ids[:, :loc], torch.tensor([[198, 198]]), token_ids[:, loc+1:]], dim=1)
        return token_ids
    if replace_double_newline:
        token_ids = replace_double_newline_func(token_ids)
    token_ids = token_ids.to(accelerator.device)
    with torch.inference_mode():
        embeddings = model.model.embed_tokens(token_ids)
    return embeddings.to(torch.bfloat16)

def main(args):
    model = args.model
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        model_max_length=sys.maxsize,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    accelerator = Accelerator(
        mixed_precision="bf16",
    )
    kwargs = {"rope_theta": args.rope_theta} if args.rope_theta is not None else {}
    model = LlamaForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        _attn_implementation="flash_attention_2",
        device_map=accelerator.device,
        **kwargs,
    )
    tokenizer.pad_token = tokenizer.eos_token
    # remember to remove <s>
    accelerator.print("Preparing Haystack...")
    haystack_embeddings = load_haystack(args, accelerator)
    assert len(haystack_embeddings) >= args.max_frame_num, "Haystack embeddings are not enough. Max frame {} is not found. Currently only {} frames.".format(args.max_frame_num, len(haystack_embeddings))
    haystack_embeddings = haystack_embeddings[:args.max_frame_num].to(accelerator.device)
    prompt = prompts[args.prompt_template]
    answer = prompt["answer"]
    preprompt_embeddings = load_text_embeddings(prompt["preprompt"], tokenizer, model, accelerator, args.replace_double_newline)
    postprompt_embeddings = load_text_embeddings(prompt["postprompt"], tokenizer, model, accelerator, args.replace_double_newline)
    answer_embedding_list = [load_text_embeddings(answer, tokenizer, model, accelerator)]
    answer_id_list = [tokenizer.encode(answer, return_tensors="pt")[:,1:]]
    query_embedding_list = [torch.load(args.needle_path, map_location="cpu").to(torch.bfloat16).to(accelerator.device)]
    accelerator.print("Starting Evaluation...")
    model = accelerator.prepare(model)
    model.gradient_checkpointing_enable()
    all_accuries = []
    for num_frames in tqdm(
        range(
            args.min_frame_num, args.max_frame_num + 1, args.frame_interval
        )
    ):
        for depth in np.arange(0, 1 + args.depth_interval, args.depth_interval):
            accuracies = []
            for query_embedding, answer_embedding, answer_id in zip(query_embedding_list, answer_embedding_list, answer_id_list):
                query_frame_idx = int(depth * num_frames)
                input_frames = torch.cat([haystack_embeddings[:query_frame_idx],query_embedding.unsqueeze(0), haystack_embeddings[query_frame_idx:num_frames]], dim=0).view(-1, haystack_embeddings.shape[-1]).unsqueeze(0)
                input_emebds = torch.cat([preprompt_embeddings, input_frames, postprompt_embeddings], dim=1)
                correct = eval_forward(
                    accelerator, model, input_emebds, answer_embedding, tokenizer.pad_token_id, answer_id, tokenizer
                )
                gc.collect()
                torch.cuda.empty_cache()
                if accelerator.is_main_process:
                    accuracies.append(correct)
            if accelerator.is_main_process:
                result = {
                    "Num. Frame": num_frames,
                    "Document Depth": round(depth * 100, -1),
                    "Score": sum(accuracies) / len(accuracies),
                }
                accelerator.print(result)
                all_accuries.append(result)
    if accelerator.is_main_process:
        df = pd.DataFrame(all_accuries)
        cmap = LinearSegmentedColormap.from_list(
            "custom_cmap", ["#F0496E", "#EBB839", "#0CD79F"]
        )

        pivot_table = pd.pivot_table(
            df,
            values="Score",
            index=["Document Depth", "Num. Frame"],
            aggfunc="mean",
        ).reset_index()  # This will aggregate
        pivot_table = pivot_table.pivot(
            index="Document Depth", columns="Num. Frame", values="Score"
        )
        # Create the heatmap with better aesthetics
        plt.figure(figsize=(17.5, 8))  # Can adjust these dimensions as needed
        sns.heatmap(
            pivot_table,
            # annot=True,
            fmt="g",
            cmap=cmap,
            cbar_kws={"label": "Score"},
        )

        # More aesthetics
        plt.xlabel("Token Limit")  # X-axis label
        plt.ylabel("Depth Percent")  # Y-axis label
        plt.xticks(rotation=45)  # Rotates the x-axis labels to prevent overlap
        plt.yticks(rotation=0)  # Ensures the y-axis labels are horizontal
        plt.tight_layout()  # Fits everything neatly into the figure area
        # save
        model_name = args.model.split("/")[-1]
        plt.savefig(f"data/{args.prompt_template}/heatmap_{model_name}.png".format(model_name))


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--model", type=str, default="output/LLaVA-NeXT-Video-7B-32K")
    args.add_argument("--max_frame_num", type=int, default=300)
    args.add_argument("--min_frame_num", type=int, default=20)
    args.add_argument("--frame_interval", type=int, default=20)
    args.add_argument("--depth_interval", type=float, default=0.1)
    args.add_argument("--num_samples", type=int, default=1)
    args.add_argument("--rope_theta", type=float, default=None)
    args.add_argument("--haystack_dir", type=str, default="video_needle_haystack/data/haystack_embeddings")
    args.add_argument("--needle_path", type=str, default="")
    args.add_argument("--prompt_template", type=str)
    args.add_argument("--replace_double_newline", action="store_true")
    
    main(args.parse_args())
