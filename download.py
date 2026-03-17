import os
from huggingface_hub import snapshot_download
from config.settings import Config

def download_model():
    config = Config()
    os.makedirs(config.model_path, exist_ok=True)

    print("Downloading...")
    snapshot_download(
        repo_id=config.base_model,
        local_dir=config.model_path,
        local_dir_use_symlinks=False
    )
    print(f"Model saved to f{config.model_path}")

if __name__ == "__main__":
    download_model()
