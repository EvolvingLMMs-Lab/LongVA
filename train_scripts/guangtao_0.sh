accelerate launch \
--config_file  accelerate_configs/single_node.yaml \
train.py \
--batch-size 1 \
--gradient-accumulate-every 4 \
--seed 2024 \
--output-dir output/Meta-Llama-3-8B-Instruct-extend \
--wandb LMExtend \
--max-train-steps 2000  \
--learning-rate 1e-5  \
--dataset  PY007/slimpajama_LLAMA3_tokenized_upsample_4096_chunk_256K \
--model meta-llama/Meta-Llama-3-8B-Instruct   \
--seq-length 256000 \
--rope-theta 1000000000 \
--parallel_mode zigzag_ring_attn \
--checkpointing-steps 200
