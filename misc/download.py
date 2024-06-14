from huggingface_hub import snapshot_download

snapshot_download(repo_id='LongVa/LLaMA8B-LLaVA-NeXT-double-newline',
                  local_dir='vision_niah/model_weights/LLaMA8B-LLaVA-NeXT-double-newline',
                  repo_type='model',
                  local_dir_use_symlinks=False,
                  resume_download=True)