# LongVA & V-NIAH
<p align="center">
    <img src="vision_niah/niah_output/LongVA-7B/heatmap.png" width="800">
</p>

<p align="center">
    🌐 <a href="XXX" target="_blank">Blog</a> | 📃 <a href="XXX" target="_blank">Paper</a> | 🤗 <a href="https://huggingface.co/collections/lmms-lab/longva-667538e09329dbc7ea498057" target="_blank">Hugging Face</a> | 🎥 <a href="XXX" target="_blank">Demo</a>
</p>

Long context capability can **zero-shot transfer** from language to vision.

LongVA can process **2000** frames or over **200K** visual tokens. It achieves state-of-the-art performance on Video-MME among 7B models.


## Installation 
This codebase is tested on CUDA 11.8 and A100-SXM.
```bash
conda create -n longva python=3.10 -y && conda activate longva
pip install torch==2.1.2 torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -e "longva/.[train]"
pip install packaging &&  pip install ninja && pip install flash-attn --no-build-isolation --no-cache-dir
pip install -r requirements.txt
```


## Local Demo

```bash
# For CLI inference, please refer to this video and image demo
python local_demo/longva_backend.py

# For multimodal chat demo with gradio on your localhost, please refer to this multimodal chat demo
python local_demo/multimodal_chat.py
```

## V-NIAH Evaluation
You need to first download a video longer than 1 hour (we sample frames at 1 fps) as the haystack video and put it at vision_niah/data/long_video.mp4. We do not provide the video because we use an actual movie in our evaluation. We can not provide it due to copyright reasons.
The can view all needle questions at [lmms-lab/v_niah_needles](https://huggingface.co/datasets/lmms-lab/v_niah_needles).
```bash
# Download the model weights
huggingface-cli download lmms-lab/LongVA-7B --local-dir vision_niah/model_weights/LongVA-7B
sh vision_niah/eval.sh
```
Results will be saved to vision_niah/niah_output.
## LMMs-Eval Evaluation
We provide both our video and image evaluation pipeline using [`lmms-eval`](https://github.com/EvolvingLMMs-Lab/lmms-eval). After installing `lmms-eval` you can install longva using
```bash
cd longva
pip install -e ".[train]"
```
Then you can use the following script to evaluate on both image and video tasks

```bash
## Image eval example
accelerate launch --num_processes 8 --main_process_port 12345 -m lmms_eval \
    --model longva \
    --model_args pretrained=lmms-lab/LongVA-7B,conv_template=qwen_1_5,model_name=llava_qwen \
    --tasks mme \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix mme_longva \
    --output_path ./logs/

## Video eval example
accelerate launch --num_processes 8 --main_process_port 12345 -m lmms_eval \
    --model longva \
    --model_args pretrained=lmms-lab/LongVA-7B,conv_template=qwen_1_5,video_decode_backend=decord,max_frames_num=32,model_name=llava_qwen \
    --tasks videomme \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix videomme_longva \
    --output_path ./logs/ 
```

## Long Text Training
```bash
sh text_extend/extend_qwen2.sh
```
It takes around 2 days to train the model on 8 A100 GPUs.
You can also download our long-context-pretrained model from huggingface:
```bash
huggingface-cli download lmms-lab/Qwen2-7B-Instrcuct-224K --local-dir text_extend/training_output/Qwen2-7B-Instrcuct-224K
```
You can evaluate the text-niah performance with this command:
```bash
sh text_extend/eval.sh
```
The results will be saved to text_extend/niah_output.

## Vision Text Alginment
Coming soon...
## Citation

If you find this work useful, please consider citing our paper:
```
@misc{zhang2024longva,
    title={LongVA: Long Context Transfer from Language to Vision},
    url={https://lmms-lab.github.io/posts/longva/},
    author={Zhang, Peiyuan and Zhang, Kaichen and Li, Bo and Zeng, Guangtao and Yang, Jingkang and Zhang, Yuanhan and Li, Chunyuan and Liu, Ziwei},
    month={June},
    year={2024}
}
```

## Acknowledgement
- LLaVA: the codebase we built upon. 
- LMMs-Eval: the codebase we used for evaluation. 

