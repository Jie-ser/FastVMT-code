"""
Download Wan2.1 models for FastVMT.

Supports downloading:
- Wan2.1-T2V-1.3B: Smaller model, lower VRAM requirement (~24GB)
- Wan2.1-T2V-14B: Larger model, better quality (~80GB VRAM)

Usage:
    python examples/download_model.py --model 1.3b
    python examples/download_model.py --model 14b
    python examples/download_model.py --model all
"""

import os
import argparse
from modelscope import snapshot_download


# Model configurations
MODELS = {
    "1.3b": {
        "repo_id": "Wan-AI/Wan2.1-T2V-1.3B",
        "local_dir": "models/Wan2.1-T2V-1.3B",
    },
    "14b": {
        "repo_id": "Wan-AI/Wan2.1-T2V-14B",
        "local_dir": "models/Wan2.1-T2V-14B",
    },
}


def download_model(model_name: str, base_dir: str = None):
    """Download a specific model."""
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}. Choose from: {list(MODELS.keys())}")

    config = MODELS[model_name]
    local_dir = config["local_dir"]

    # Use base_dir if provided
    if base_dir:
        local_dir = os.path.join(base_dir, os.path.basename(local_dir))

    print(f"Downloading {config['repo_id']} to {local_dir}...")
    snapshot_download(config["repo_id"], local_dir=local_dir)
    print(f"Successfully downloaded {model_name} model to {local_dir}")


def main():
    parser = argparse.ArgumentParser(description="Download Wan2.1 models for FastVMT")
    parser.add_argument(
        "--model",
        type=str,
        default="14b",
        choices=["1.3b", "14b", "all"],
        help="Model to download: '1.3b', '14b', or 'all' (default: 14b)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Base directory for model downloads (default: models/)"
    )
    args = parser.parse_args()

    if args.model == "all":
        for model_name in MODELS:
            download_model(model_name, args.output_dir)
    else:
        download_model(args.model, args.output_dir)


if __name__ == "__main__":
    main()
