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
import json
from tqdm import tqdm
import gc
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter
from easy_context import Qwen2ForCausalLM_RingAttn
import os
import seaborn as sns
import pandas as pd
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


NEEDLE_FORMAT = "\nThe special magic Singapore number is: {}.\n"
DISTRACTOR_LIST = ["\nThe special magic New York number is: {}.\n", \
    "\nThe special magic London number is: {}.\n",
    "\nThe special magic Paris number is: {}.\n",
    "\nThe special magic Tokyo number is: {}.\n",
    "\nThe special magic Beijing number is: {}.\n",
    "\nThe special magic Berlin number is: {}.\n",]
PREFIX = "This is a very long story book: <book>"
QUESTION_STR = "</book>.\n Based on the content of the book, Question: What is the special magic Singapore number? Answer: The special magic Singapore number is: "


def safe_tokenize(tokenizer, text):
    tokenized = tokenizer.encode(text)
    if tokenizer.bos_token != None and len(tokenized) > 0 and tokenized[0] == tokenizer.bos_token_id:
        tokenized = tokenized[1:]
    return tokenized
def eval_forward(accelerator, model, input_ids, pad_id, answer_ids, tokenizer, distractor_number_list):
    # first append labels to input_ids
    prompt_length = input_ids.shape[1]
    labels_length = answer_ids.shape[1]
    input_ids = torch.cat([input_ids, answer_ids], dim=1)
    # second pad input_ids to the multiple of accelerator.num_processes
    pad_tensor = torch.tensor(
        [pad_id]
        * (
            (accelerator.num_processes * 2)
            - input_ids.shape[1] % (accelerator.num_processes * 2)
        )
    ).unsqueeze(0)
    input_ids = torch.cat([input_ids, pad_tensor], dim=1)
    position_ids = (
        torch.arange(input_ids.shape[1]).unsqueeze(0).expand(input_ids.shape[0], -1)
    )
    prepared = prepare_seq_parallel_inputs(
        "zigzag_ring_attn",
        input_ids,
        position_ids,
        None,
        accelerator.process_index,
        accelerator.num_processes,
        accelerator.device,
    )
    local_input_ids = prepared["local_input_ids"]
    local_position_ids = prepared["local_position_ids"]
    with torch.inference_mode():
        logits = model(
            local_input_ids,
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
    if not correct and accelerator.is_main_process:
        print(
            "Predicted: ",
            tokenizer.decode(pred.squeeze().tolist()),
            "Answer: ",
            tokenizer.decode(answer_ids.squeeze().tolist()),
            "Distactor: ",
            distractor_number_list,
        )
    return int(correct)


def load_haystack(args, accelerator, tokenizer):
    context = ""
    # do not count <s>
    while len(safe_tokenize(tokenizer,context)) - 1 < args.max_context_length:
        accelerator.print(f"Current Context Length: {len(safe_tokenize(tokenizer,context))-1}")
        accelerator.print(glob.glob(f"{args.haystack_dir}/*.txt"))
        for file in glob.glob(f"{args.haystack_dir}/*.txt"):
            with open(file, "r") as f:
                accelerator.print(f"Reading {file}")
                context += f.read()
        if len(safe_tokenize(tokenizer,context)) - 1 > args.max_context_length:
            break
    tokenized_haystack = safe_tokenize(tokenizer,context)
    return tokenized_haystack


def construct_prompt(
    tokenized_haystack,
    tokenized_prefix,
    tokenized_postfix,
    tokenized_needle,
    context_length,
    tokenized_distractor_list,
    depth,
):
    period_tokens = [29889, 869]
    prompt = tokenized_haystack[:context_length]
    # insert distractors
    for distractor in tokenized_distractor_list:
        start_index = np.random.randint(0, len(prompt))
        for i in range(start_index, len(prompt)):
            if prompt[i] in period_tokens:
                start_index = i + 1
                break
        prompt = prompt[:start_index] + distractor + prompt[start_index:]
    # insert the needle into depth of the haystack
    if depth == 0:
        start_index = 0
    else:
        start_index = int(len(prompt) * depth)
        # find the closest period token
        for i in range(start_index, len(prompt)):
            if prompt[i] in period_tokens:
                start_index = i + 1
                break
    prompt = prompt[:start_index] + tokenized_needle + prompt[start_index:]
    
    prompt = tokenized_prefix + prompt + tokenized_postfix
    # from transformers import AutoTokenizer
    # tk = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    # tk.decode(prompt)
    return prompt


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
    if "qwen2" in args.model.lower():
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
    model = accelerator.prepare(model)
    model.gradient_checkpointing_enable()
    tokenizer.pad_token = tokenizer.eos_token
    # remember to remove <s>
    accelerator.print("Preparing Haystack...")
    tokenized_haystack = load_haystack(args, accelerator, tokenizer)
    tokenized_prefix = tokenizer.encode(PREFIX)

    accelerator.print("Starting Evaluation...")
    random_number_list = [
        int(np.random.randint(10**args.rnd_number_digits))
        for i in range(args.num_samples)
    ]
    print(random_number_list)
    distractor_number_list = [
        int(np.random.randint(10**args.rnd_number_digits))
        for i in range(args.num_distractor)
    ]
    distractor_str_list = [
        DISTRACTOR_LIST[i % len(DISTRACTOR_LIST)].format(distractor_number_list[i])
        for i in range(args.num_distractor)
    ]
    tokenized_distractor_list = [
        safe_tokenize(tokenizer,distractor_str) for distractor_str in distractor_str_list
    ]
    accelerator.print(distractor_str_list)
    all_accuries = []
    for context_length in tqdm(
        range(
            args.min_context_length, args.max_context_length + 1, args.context_interval
        )
    ):
        for depth in np.arange(0, 1 + args.depth_interval, args.depth_interval):
            accuracies = []
            for random_number in random_number_list:
                needle_str = NEEDLE_FORMAT.format(random_number)
                question_str = QUESTION_STR
                asnwer_str = str(random_number)

                tokenized_needle = safe_tokenize(tokenizer,needle_str)
                tokenized_postfix = safe_tokenize(tokenizer,question_str)
                tokenizer_answer = safe_tokenize(tokenizer,asnwer_str)

                prompt = construct_prompt(
                    tokenized_haystack,
                    tokenized_prefix,
                    tokenized_postfix,
                    tokenized_needle,
                    context_length,
                    tokenized_distractor_list,
                    depth,
                )
                input_ids = torch.tensor([prompt])
                answer_ids = torch.tensor([tokenizer_answer])
                correct = eval_forward(
                    accelerator, model, input_ids, tokenizer.pad_token_id, answer_ids, tokenizer, distractor_number_list
                )
                gc.collect()
                torch.cuda.empty_cache()
                if accelerator.is_main_process:
                    accuracies.append(correct)
            if accelerator.is_main_process:
                result = {
                    "Context Length": context_length,
                    "Document Depth": round(depth * 100, -1),
                    "Score": sum(accuracies) / len(accuracies),
                }
                accelerator.print(result)
                all_accuries.append(result)
    if accelerator.is_main_process:
        model_name = args.model.split("/")[-1]
        os.makedirs(f"text_extend/niah_output/distractor_{args.num_distractor}/{model_name}", exist_ok=True)
        # save all_accuries as json
        with open(f"text_extend/niah_output/distractor_{args.num_distractor}/{model_name}/all_accuracies.json", "w") as f:
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
        index=["Document Depth", "Context Length"],
        aggfunc="mean",
    ).reset_index()  # This will aggregate
    pivot_table = pivot_table.pivot(
        index="Document Depth", columns="Context Length", values="Score"
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
            return f'{int(x/1000)}K'
        return f'{x}'

    context_lengths = pivot_table.columns
    formatted_context_lengths = [thousands_formatter(x, None) for x in context_lengths]

    # More aesthetics
    plt.xlabel("Token Limit", fontsize=14)  # X-axis label
    plt.ylabel("Depth Percent", fontsize=14)  # Y-axis label
    plt.xticks(ticks=[i + 0.5 for i in range(len(context_lengths))], labels=formatted_context_lengths, rotation=45, fontsize=14)
    # plt.xticks(rotation=45, fontsize=14)  # Rotates the x-axis labels to prevent overlap
    plt.yticks(rotation=0, fontsize=14)  # Ensures the y-axis labels are horizontal
    plt.tight_layout()  # Fits everything neatly into the figure area
    # save
    model_name = args.model.split("/")[-1]
    # mkdir if not exist text_extend/niah_output/distractor_{args.num_distractor}/
    os.makedirs(f"text_extend/niah_output/distractor_{args.num_distractor}/{model_name}", exist_ok=True)
    plt.savefig(f"text_extend/niah_output/distractor_{args.num_distractor}/{model_name}/heatmap.png")
    # calculate average accuracy
    average_accuracy = df["Score"].mean()
    print(f"Average Accuracy: {average_accuracy}")
    # save as txt
    with open(f"text_extend/niah_output/distractor_{args.num_distractor}/{model_name}/avg_accuracy.txt", "w") as f:
        f.write(f"Average Accuracy: {average_accuracy}\n")
        
def main(args):
    if args.plot_only:
        # load all_accuracies from json
        model_name = args.model.split("/")[-1]
        with open(f"text_extend/niah_output/distractor_{args.num_distractor}/{model_name}/all_accuracies.json", "r") as f:
            all_accuracies = json.load(f)
        plot(args, all_accuracies)
    else:
        all_accuracies, accelerator = inference(args)
        if accelerator.is_main_process:
            plot(args, all_accuracies)

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--model", type=str, default="PY007/Llama2-7B-64K")
    args.add_argument("--max_context_length", type=int, default=100000)
    args.add_argument("--min_context_length", type=int, default=1000)
    args.add_argument("--context_interval", type=int, default=1000)
    args.add_argument("--depth_interval", type=float, default=0.1)
    args.add_argument("--num_samples", type=int, default=10)
    args.add_argument("--rope_theta", type=float, default=None)
    args.add_argument("--rnd_number_digits", type=int, default=7)
    args.add_argument("--haystack_dir", type=str, default=None)
    args.add_argument("--num_distractor", type=int, default=0)
    args.add_argument("--plot_only", action="store_true")
    main(args.parse_args())