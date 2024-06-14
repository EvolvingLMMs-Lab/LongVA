for MODEL_NAME in LLaVA-NeXT-Qwen2-7B-extend-avgpool2x2-anyres7x7
do
mkdir vision_niah/data/haystack_embeddings/$MODEL_NAME
mkdir vision_niah/data/needle_embeddings/$MODEL_NAME
python vision_niah/produce_haystack_embedding.py --model vision_niah/model_weights/$MODEL_NAME --output_dir vision_niah/data/haystack_embeddings/$MODEL_NAME --sampled_frames_num 3000 --pooling_size 2 
python vision_niah/produce_needle_embedding.py --model vision_niah/model_weights/$MODEL_NAME --output_dir vision_niah/data/needle_embeddings/$MODEL_NAME --pooling_size 2 

accelerate launch --num_processes 8 --config_file  accelerate_configs/deepspeed_inference.yaml  --main_process_port 6000 vision_niah/eval_vision_niah.py \
    --model  vision_niah/model_weights/$MODEL_NAME \
    --needle_path vision_niah/data/needle_embeddings/$MODEL_NAME/image1.pt \
    --haystack_dir vision_niah/data/haystack_embeddings/$MODEL_NAME \
    --prompt_template llama3_color_yi \
    --max_frame_num 3000 \
    --min_frame_num 300 \
    --frame_interval 300 \
    --depth_interval 0.2 
done





MODEL_NAME=LLaVA-Yi-1.5-9B-extend-CLIP-anyres7x7-avgpool2x2

mkdir vision_niah/data/haystack_embeddings/$MODEL_NAME
mkdir vision_niah/data/needle_embeddings/$MODEL_NAME
python vision_niah/produce_haystack_embedding.py --model vision_niah/model_weights/$MODEL_NAME --output_dir vision_niah/data/haystack_embeddings/$MODEL_NAME --sampled_frames_num 3000 --pooling_size 2 
python vision_niah/produce_needle_embedding.py --model vision_niah/model_weights/$MODEL_NAME --output_dir vision_niah/data/needle_embeddings/$MODEL_NAME --pooling_size 2 

accelerate launch --num_processes 8 --config_file  accelerate_configs/deepspeed_inference.yaml  --main_process_port 6000 vision_niah/eval_vision_niah.py \
    --model  vision_niah/model_weights/$MODEL_NAME \
    --needle_path vision_niah/data/needle_embeddings/$MODEL_NAME/image1.pt \
    --haystack_dir vision_niah/data/haystack_embeddings/$MODEL_NAME \
    --prompt_template yi_color \
    --max_frame_num 3000 \
    --min_frame_num 2400 \
    --frame_interval 600 \
    --depth_interval 0.2 


