from huggingface_hub import snapshot_download

snapshot_download(repo_id='LongVa/LLaVA-NeXT-Qwen2-7B-extend-avgpool2x2-anyres7x7',
                  local_dir='vision_niah/model_weights/LLaVA-NeXT-Qwen2-7B-extend-avgpool2x2-anyres7x7',
                  repo_type='model',
                  local_dir_use_symlinks=False,
                  resume_download=True)