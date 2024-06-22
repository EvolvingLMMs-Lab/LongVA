accelerate launch \
--config_file  easy_context/accelerate_configs/single_node.yaml \
text_extend/text_extend_train.py \
--batch-size 1 \
--gradient-accumulate-every 4 \
--seed 2024 \
--output-dir text_extend/training_output/Qwen2-7B-Instrcuct-224K\
--wandb LMExtend \
--max-train-steps 1000  \
--learning-rate 1e-5  \
--dataset  PY007/slimpajama_Qwen2_tokenized_upsample_4096_chunk_256K \
--model Qwen/Qwen2-7B-Instruct   \
--seq-length 224000 \
--rope-theta 1000000000 \
--parallel_mode zigzag_ring_attn \
--checkpointing-steps 200

rm text_extend/training_output/Qwen2-7B-Instruct-extend/model.safetensors