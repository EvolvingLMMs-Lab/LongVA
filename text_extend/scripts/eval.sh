accelerate launch --num_processes 8 --config_file  accelerate_configs/deepspeed_inference.yaml  --main_process_port 6000 eval_needle.py \
    --model PY007/EasyContext-1M-Llama-2-7B  \
    --max_context_length 1000000 \
    --min_context_length 50000 \
    --context_interval   50000 \
    --depth_interval 0.1 \
    --num_samples 2 \
    --rnd_number_digits 7 \
    --haystack_dir text_extend/PaulGrahamEssays \
    