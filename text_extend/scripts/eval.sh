for model in Qwen2-7B-Instruct-extend-step_1000 Meta-Llama-3-8B-Instruct-extend-step1000 Qwen2-7B-Instruct
do
for num_distractor in 0 5
do
accelerate launch --num_processes 8 --config_file  accelerate_configs/deepspeed_inference.yaml  --main_process_port 6000 text_extend/eval_text_niah.py \
    --model text_extend/training_output/$model  \
    --max_context_length 512000 \
    --min_context_length 32000 \
    --context_interval   32000 \
    --depth_interval 0.2 \
    --num_samples 3 \
    --rnd_number_digits 7 \
    --haystack_dir text_extend/PaulGrahamEssays \
    --num_distractor $num_distractor
done
done









# to plot the heatmap only
accelerate launch --num_processes 1 --config_file  accelerate_configs/deepspeed_inference.yaml  --main_process_port 6000 text_extend/eval_text_niah.py \
     --model text_extend/training_output/Qwen2-7B-Instruct-extend-step_1000  \
     --num_distractor 5 \
     --plot_only