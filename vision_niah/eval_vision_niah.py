import argparse
import gc
import sys
import torch
from transformers import AutoTokenizer
from transformers import LlamaForCausalLM
from easy_context import Qwen2ForCausalLM_RingAttn
from tqdm import tqdm
from accelerate import Accelerator
import glob
import numpy as np
from tqdm import tqdm
import gc
import matplotlib.pyplot as plt
import os
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import pandas as pd
from pathlib import Path
import random
import json
from datasets import load_dataset
from easy_context import (
    prepare_seq_parallel_inputs,
    apply_seq_parallel_monkey_patch,
)
apply_seq_parallel_monkey_patch("zigzag_ring_attn", "llama")


SEED = 24242424
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

prompt_templates = {
    "mistral": {
        "preprompt": "<s>[INST]",
        "postprompt": " [/INST]"
    },
    "vicuna": {
        "preprompt": "<s>A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER:",
        "postprompt": "ASSISTANT:"
    },
    "llama3": {
        "preprompt": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n",
        "postprompt": "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    },
    "qwen2": {
        "preprompt": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n",
        "postprompt": "<|im_end|>\n<|im_start|>assistant\n",
    }, 
    "yi": {
        "preprompt": "<|im_start|>system\nAnswer the questions.<|im_end|>\n<|im_start|>user\n",
        "postprompt": "<|im_end|>\n<|im_start|>assistant\n",
    },
}
# \nAnswer the question using a single word or phrase.
# The color of the bottle cap is
# answer = "Yellow"


def safe_tokenize(tokenizer, text):
    tokenized = tokenizer.encode(text, return_tensors="pt")
    if tokenizer.bos_token != None and len(tokenized) > 0 and tokenized[0, 0] == tokenizer.bos_token_id:
        tokenized = tokenized[:, 1:]
    return tokenized

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
    accelerator.print(input_embeds.shape)
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
        # print id as well
        print(
            "Predicted: ",
            pred.squeeze().tolist(),
            "Answer: ",
            answer_ids.squeeze().tolist(),
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
    token_ids = safe_tokenize(tokenizer, str)
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

def inference(args):
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
    if "qwen2" in args.model.lower() or "longva" in args.model.lower():
        model = Qwen2ForCausalLM_RingAttn.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            _attn_implementation="flash_attention_2",
            device_map=accelerator.device,
            **kwargs,
        )
    else:
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
    prompt = prompt_templates[args.prompt_template]
    preprompt_embeddings = load_text_embeddings(prompt["preprompt"], tokenizer, model, accelerator, args.replace_double_newline)
    postprompt_embeddings = load_text_embeddings(prompt["postprompt"], tokenizer, model, accelerator, args.replace_double_newline)
    
    needle_dataset = load_dataset(args.needle_dataset)["test"]
    answer_embedding_list = []
    answer_id_list = []
    needle_embedding_list = []
    question_embeding_list = []
    for index, instance in enumerate(needle_dataset):
        answer = instance["answer"]
        question = instance["question"]
        needle_embedding_list.append(torch.load(args.needle_embedding_dir + f"/{index}.pt", map_location="cpu").to(torch.bfloat16).to(accelerator.device))
        answer_embedding_list.append(load_text_embeddings(answer, tokenizer, model, accelerator))
        answer_id_list.append(safe_tokenize(tokenizer, answer))
        question_embeding_list.append(load_text_embeddings(question, tokenizer, model, accelerator))
        
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
            for question_embedding, needle_embedding, answer_embedding, answer_id in zip(question_embeding_list, needle_embedding_list, answer_embedding_list, answer_id_list):
                query_frame_idx = int(depth * num_frames)
                input_frames = torch.cat([haystack_embeddings[:query_frame_idx],needle_embedding.unsqueeze(0), haystack_embeddings[query_frame_idx:num_frames]], dim=0).view(-1, haystack_embeddings.shape[-1]).unsqueeze(0)
                input_emebds = torch.cat([preprompt_embeddings, input_frames,question_embedding, postprompt_embeddings], dim=1)
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
                    "Frame Depth": round(depth * 100, -1),
                    "Score": sum(accuracies) / len(accuracies),
                }
                accelerator.print(result)
                all_accuries.append(result)
    if accelerator.is_main_process:
        model_name = args.model.split("/")[-1]
        os.makedirs(f"{args.output_path}/{model_name}", exist_ok=True)
        # save all_accuries as json
        with open(f"{args.output_path}/{model_name}/all_accuracies.json", "w") as f:
            json.dump(all_accuries, f, indent=4)
    return all_accuries, accelerator


def plot(args,  all_accuries):
    df = pd.DataFrame(all_accuries)
    cmap = LinearSegmentedColormap.from_list(
        "custom_cmap", ["#F0496E", "#EBB839", "#9ad5b3"]
    )

    pivot_table = pd.pivot_table(
        df,
        values="Score",
        index=["Frame Depth", "Num. Frame"],
        aggfunc="mean",
    ).reset_index()  # This will aggregate
    pivot_table = pivot_table.pivot(
        index="Frame Depth", columns="Num. Frame", values="Score"
    )
    # Create the heatmap with better aesthetics
    plt.figure(figsize=(17.5, 8))  # Can adjust these dimensions as needed
    ax = sns.heatmap(
        pivot_table,
        # annot=True,
        fmt="g",
        vmin=0,
        vmax=1,
        linecolor='white',
        linewidths=1.5, 
        cmap=cmap,
        cbar_kws={"label": "Score"},
    )
    
    # Set the color bar label font size
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.label.set_size(14)
    cbar.ax.tick_params(labelsize=14)

    
    # Define the formatter function
    def thousands_formatter(x, pos):
        if x >= 1000:
            return f'{x/1000:.1f}K'
        return f'{x}'

    context_lengths = pivot_table.columns
    formatted_context_lengths = [thousands_formatter(x, None) for x in context_lengths]

    # More aesthetics
    plt.xlabel("Num. of Frames", fontsize=14)  # X-axis label
    plt.ylabel("Depth Percent", fontsize=14)  # Y-axis label
    plt.xticks(ticks=[i + 0.5 for i in range(len(context_lengths))], labels=formatted_context_lengths, rotation=45, fontsize=14)
    # plt.xticks(rotation=45, fontsize=14)  # Rotates the x-axis labels to prevent overlap
    plt.yticks(rotation=0, fontsize=14)  # Ensures the y-axis labels are horizontal
    plt.tight_layout()  # Fits everything neatly into the figure area
    # save
    model_name = args.model.split("/")[-1]

    plt.savefig(f"{args.output_path}/{model_name}/heatmap.png")
    # calculate average accuracy
    average_accuracy = df["Score"].mean()
    print(f"Average Accuracy: {average_accuracy}")
    # save as txt
    with open(f"{args.output_path}/{model_name}/avg_accuracy.txt", "w") as f:
        f.write(f"Average Accuracy: {average_accuracy}\n")
        
def main(args):
    if args.plot_only:
        # load all_accuracies from json
        model_name = args.model.split("/")[-1]
        with open(f"{args.output_path}/{model_name}/all_accuracies.json", "r") as f:
            all_accuracies = json.load(f)
        plot(args, all_accuracies)
    else:
        all_accuracies, accelerator = inference(args)
        if accelerator.is_main_process:
            plot(args, all_accuracies)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--model", type=str, default="output/LLaVA-NeXT-Video-7B-32K")
    args.add_argument("--max_frame_num", type=int, default=300)
    args.add_argument("--needle_dataset", type=str, default="lmms-lab/v_niah_needles")
    args.add_argument("--min_frame_num", type=int, default=20)
    args.add_argument("--frame_interval", type=int, default=20)
    args.add_argument("--output_path", type=str, default="vision_niah/niah_output")
    args.add_argument("--depth_interval", type=float, default=0.1)
    args.add_argument("--num_samples", type=int, default=1)
    args.add_argument("--rope_theta", type=float, default=None)
    args.add_argument("--haystack_dir", type=str, default="video_needle_haystack/data/haystack_embeddings")
    args.add_argument("--needle_embedding_dir", type=str, default="vision_niah/data/needle_embeddings")
    args.add_argument("--prompt_template", type=str)
    args.add_argument("--replace_double_newline", action="store_true")
    args.add_argument("--plot_only", action="store_true")
    
    main(args.parse_args())
