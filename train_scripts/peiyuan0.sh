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
--seq-length 8000 \
--rope-theta 1000000000 \
--parallel_mode zigzag_ring_attn \
--checkpointing-steps 5 \
--resume-from-checkpoint output/Qwen2-7B-Instruct-extend/states/step_5

scratch:
tensor([[   24,    21,   382,    38,  7858,   263,    11, 29650, 89499,    13]])
tensor([[  532,    32,    61, 35702, 11665,   716,    90,    73,  3417,     7]])
tensor([[ 7149,  3308,   624,   785, 44564,  9214,   476,   264,  4709,   315]])
tensor([[ 311, 3271, 4637,  311,   30,  358, 8679,  419, 3146,  374]])
  0%|▌                                                                                                                                                                                         | 6/2000 [02:37<18:43:24, 33.80s/it, loss=2.91, ppl=18.4]
  tensor([[ 6241,  5893, 10294,   785, 91797, 34205,  6351,   374,   825,   315]])
tensor([[10981,  5486,    11,   323,   279,  3059,   525,  6839,   304,  4520]])
tensor([[17775, 16407,    11,   323,  4045,    82, 50664, 12375, 16824, 56783]])
tensor([[  645,    13, 10328,    11,  7025,   358,   633, 14036,   315,   678]])
  0%|▋                                                                                                                                                                                         | 7/2000 [02:53<15:26:33, 27.89s/it, loss=2.28, ppl=9.77]
  tensor([[   265,   9962,     71, 128209,  18593,  26524,   1694,     11,    220,
          13136]])
tensor([[   13,   362, 52335,   311, 88221, 42743,   198,   785,  8031, 28682]])
tensor([[58486, 29299,  8291, 19687,  1612,    80, 39137, 45583, 40049,  7287]])
tensor([[  311,  1414,   279, 31774,   315,   279,  7375,  1865,  2500,   389]])
  0%|▋                                                                                                                                                                                         | 8/2000 [03:09<13:16:21, 23.99s/it, loss=2.39, ppl=10.9]