accelerate launch \
--config_file  accelerate_configs/single_node.yaml \
text_extend/text_extend_train.py \
--batch-size 1 \
--gradient-accumulate-every 4  \
--seed 2024 \
--output-dir  text_extend/training_output/vicuna-7b-v1.5-extend \
--wandb LMExtend \
--max-train-steps 1000  \
--learning-rate 1e-5  \
--dataset PY007/slimpajama_llama2_tokenized_upsample_4096_chunk_256K \
--model lmsys/vicuna-7b-v1.5  \
--seq-length 256000 \
--rope-theta 1000000000 \
--parallel_mode zigzag_ring_attn \
--checkpointing-steps 200

rm text_extend/training_output/vicuna-7b-v1.5-extend/model.safetensors