accelerate launch \
--config_file  accelerate_configs/single_node.yaml \
train.py \
--batch-size 1 \
--gradient-accumulate-every 4 \
--seed 2024 \
--output-dir output/Qwen2-7B-Instruct-extend \
--wandb LMExtend \
--max-train-steps 2000  \
--learning-rate 1e-5  \
--dataset  PY007/slimpajama_Qwen2_tokenized_upsample_4096_chunk_256K \
--model /home/yhzhang/peiyuan/LongContextTransfer/output/Qwen2-7B-Instruct   \
--seq-length 64000 \
--rope-theta 1000000000 \
--parallel_mode zigzag_ring_attn \
--checkpointing-steps 5 \
--resume-from-checkpoint output/Qwen2-7B-Instruct-extend/states/step_5

 | 17/2000 [11:10<20:52:43, 37.90s/it, loss=1.9, ppl=6.69]