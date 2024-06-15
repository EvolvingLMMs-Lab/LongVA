for MODEL_NAME in LLaVA-NeXT-Qwen2-7B-extend-avgpool2x2-anyres7x7
do
mkdir vision_niah/data/haystack_embeddings/$MODEL_NAME
mkdir vision_niah/data/needle_embeddings/$MODEL_NAME
python vision_niah/produce_haystack_embedding.py --model vision_niah/model_weights/$MODEL_NAME --output_dir vision_niah/data/haystack_embeddings/$MODEL_NAME --sampled_frames_num 3000 --pooling_size 2 
python vision_niah/produce_needle_embedding.py --model vision_niah/model_weights/$MODEL_NAME --output_dir vision_niah/data/needle_embeddings/$MODEL_NAME --pooling_size 2 --needle_dataset LongVa/v_niah_needles

accelerate launch --num_processes 8 --config_file  accelerate_configs/deepspeed_inference.yaml  --main_process_port 6000 vision_niah/eval_vision_niah.py \
    --model  vision_niah/model_weights/$MODEL_NAME \
    --needle_embedding_dir vision_niah/data/needle_embeddings/$MODEL_NAME \
    --haystack_dir vision_niah/data/haystack_embeddings/$MODEL_NAME \
    --needle_dataset LongVa/v_niah_needles \
    --prompt_template qwen2 \
    --max_frame_num 3000 \
    --min_frame_num  200\
    --frame_interval 200 \
    --depth_interval 0.2 
done


