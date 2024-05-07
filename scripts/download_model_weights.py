def hf():
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id='Qwen/Qwen-14B-Chat',
                      repo_type='model',
                      local_dir='./model_dir',
                      resume_download=True)


if __name__ == "__main__":
    hf()
