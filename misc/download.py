from huggingface_hub import snapshot_download

snapshot_download(repo_id='LongVa/LLaVA-LLaMA-extend-anyres7x7-avgpool2x2',
                  local_dir='vision_niah/model_weights/LLaVA-LLaMA-extend-anyres7x7-avgpool2x2',
                  repo_type='model',
                  local_dir_use_symlinks=False,
                  resume_download=True)