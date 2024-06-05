accelerate launch \
--config_file  accelerate_configs/single_node.yaml \
train.py \
--batch-size 1 \
--gradient-accumulate-every 4  \
--seed 2024 \
--output-dir  ./output/Yi-1.5-9B-Chat-16K-extend \
--wandb LMExtend \
--max-train-steps 2000  \
--learning-rate 1e-5  \
--dataset PY007/slimpajama_Yi1.5_tokenized_upsample_4096_chunk_256K \
--model 01-ai/Yi-1.5-9B-Chat-16K  \
--seq-length 256000 \
--rope-theta 1000000000 \
--parallel_mode zigzag_ring_attn \
--checkpointing-steps 200