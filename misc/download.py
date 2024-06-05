from huggingface_hub import snapshot_download

snapshot_download(repo_id='01-ai/Yi-1.5-9B-Chat-16K',
                  local_dir='output/Yi-1.5-9B-Chat-16K',
                  repo_type='model',
                  local_dir_use_symlinks=False,
                  resume_download=True)