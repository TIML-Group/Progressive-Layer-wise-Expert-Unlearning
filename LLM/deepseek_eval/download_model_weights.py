import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import argparse

from huggingface_hub import snapshot_download


DEFAULT_MODEL_ID = "deepseek-ai/DeepSeek-V2-Lite-Chat"
DEFAULT_MODEL_ROOT = "/root/autodl-tmp/SEUF_MCU/LLM/model"


def get_local_model_dir(model_name_or_path: str, model_root: str) -> str:
    if os.path.isdir(model_name_or_path):
        return model_name_or_path

    model_dir_name = model_name_or_path.rstrip("/").split("/")[-1]
    return os.path.join(model_root, model_dir_name)


def download_model(model_name_or_path: str, model_root: str) -> str:
    os.makedirs(model_root, exist_ok=True)

    local_model_dir = get_local_model_dir(model_name_or_path, model_root)
    config_path = os.path.join(local_model_dir, "config.json")

    if os.path.isfile(config_path):
        print(f"Model already exists at: {local_model_dir}")
        return local_model_dir

    if os.path.isdir(model_name_or_path):
        raise FileNotFoundError(
            f"Local model directory is missing config.json: {model_name_or_path}"
        )

    print(f"Downloading model '{model_name_or_path}' to {local_model_dir}")
    snapshot_download(
        repo_id=model_name_or_path,
        local_dir=local_model_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print(f"Download completed: {local_model_dir}")
    return local_model_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download model weights to the local model directory.")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-root", type=str, default=DEFAULT_MODEL_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    download_model(args.model_name, args.model_root)


if __name__ == "__main__":
    main()
