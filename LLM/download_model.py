"""Download model weights into the local model root (default $PLEU_LLM_MODELS or LLM/models).

    python download_model.py --model deepseek
    python download_model.py --model qwen
"""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

from pleu_llm.modeling import DEFAULT_MODEL_ROOT, MODEL_IDS


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="deepseek", help="deepseek | qwen | HF repo id")
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    args = parser.parse_args()
    repo_id = MODEL_IDS.get(args.model, args.model)
    target = args.model_root / repo_id.split("/")[-1]
    snapshot_download(repo_id=repo_id, local_dir=str(target))
    print(f"Downloaded {repo_id} to {target}")


if __name__ == "__main__":
    main()
