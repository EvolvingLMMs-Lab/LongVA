from huggingface_hub import snapshot_download

snapshot_download(repo_id='LongVa/vicuna-7b-v1.5-extend-step1000',
                  local_dir='text_extend/training_output/vicuna-7b-v1.5-extend-step1000',
                  repo_type='model',
                  local_dir_use_symlinks=False,
                  resume_download=True)