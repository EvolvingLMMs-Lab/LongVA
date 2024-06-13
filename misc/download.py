from huggingface_hub import snapshot_download

snapshot_download(repo_id='LongVa/Qwen2-7B-Instruct-extend-step_1000',
                  local_dir='output/Qwen2-7B-Instruct-extend-step_1000',
                  repo_type='model',
                  local_dir_use_symlinks=False,
                  resume_download=True)