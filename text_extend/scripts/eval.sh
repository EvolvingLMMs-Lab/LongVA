accelerate launch --num_processes 8 --config_file  accelerate_configs/deepspeed_inference.yaml  --main_process_port 6000 text_extend/eval_text_niah.py \
    --model text_extend/training_output  \
    --max_context_length 400000 \
    --min_context_length 400000 \
    --context_interval   400000 \
    --depth_interval 0.1 \
    --num_samples 2 \
    --rnd_number_digits 7 \
    --haystack_dir text_extend/PaulGrahamEssays \
    --num_distractor 5