#!/usr/bin/env python3
"""
Model download and management pipeline for Qwen 3.8 27B MLX and DFlash models on Apple Silicon.
Supports downloading from HuggingFace with progress tracking, validation, and local caching.
"""

import argparse
import sys
import time
from typing import Dict, Optional, TypedDict

from huggingface_hub import snapshot_download, HfApi


class ModelInfo(TypedDict):
    repo_id: str
    description: str
    type: str
    size_gb: str


MODELS: Dict[str, ModelInfo] = {
    "qwen3.8-27b-4bit": {
        "repo_id": "mlx-community/Qwen3.8-27B-4bit",
        "description": "Qwen 3.8 27B (4-bit MLX quantized) - Vision-Language & Reasoning, 262k context",
        "type": "target-mlx",
        "size_gb": "~16.2 GB"
    },
    "qwen3.8-27b-8bit": {
        "repo_id": "mlx-community/Qwen3.8-27B-8bit",
        "description": "Qwen 3.8 27B (8-bit MLX quantized) - Higher precision",
        "type": "target-mlx",
        "size_gb": "~29.0 GB"
    },
    "qwen3.8-27b-nvfp4": {
        "repo_id": "mlx-community/Qwen3.8-27B-nvfp4",
        "description": "Qwen 3.8 27B (NVFP4) - max-throughput arm; NOT bit-comparable to the 4-bit target",
        "type": "target-mlx",
        "size_gb": "~16 GB (approx)"
    },
    "qwen3.8-27b-mtp-4bit": {
        "repo_id": "mlx-community/Qwen3.8-27B-MTP-4bit",
        "description": "Official Qwen 3.8 MTP drafter, 4-bit (block size 3) - use with mlx-vlm --draft-model",
        "type": "draft-mtp",
        "size_gb": "~0.24 GB"
    },
    "qwen3.8-27b-mtp-8bit": {
        "repo_id": "mlx-community/Qwen3.8-27B-MTP-8bit",
        "description": "Official Qwen 3.8 MTP drafter, 8-bit - higher-precision drafter A/B",
        "type": "draft-mtp",
        "size_gb": "~0.4 GB (approx)"
    },
    "qwen3.8-27b-mtp-nvfp4": {
        "repo_id": "mlx-community/Qwen3.8-27B-MTP-nvfp4",
        "description": "Official Qwen 3.8 MTP drafter, NVFP4 - pairs with the nvfp4 target",
        "type": "draft-mtp",
        "size_gb": "~0.3 GB (approx)"
    },
    "qwen3.6-27b-4bit": {
        "repo_id": "mlx-community/Qwen3.6-27B-4bit",
        "description": "Qwen 3.6 27B (4-bit MLX quantized) - DFlash Target Model",
        "type": "target-dflash",
        "size_gb": "~16.2 GB"
    },
    "qwen3.6-27b-dflash": {
        "repo_id": "z-lab/Qwen3.6-27B-DFlash",
        "description": "z-lab Qwen 3.6 27B DFlash Drafter (Block-diffusion, 16 tokens/pass)",
        "type": "draft-dflash",
        "size_gb": "~1.5 GB"
    },
    "qwen3-8b-dflash": {
        "repo_id": "z-lab/Qwen3-8B-DFlash-b16",
        "description": "z-lab Qwen 3 8B DFlash Drafter",
        "type": "draft-dflash",
        "size_gb": "~1.2 GB"
    }
}

def list_models() -> None:
    print("\n" + "="*80)
    print(f"{'Key':<22} | {'Estimated Size':<15} | {'Repository ID':<35}")
    print("-" * 80)
    for key, info in MODELS.items():
        print(f"{key:<22} | {info['size_gb']:<15} | {info['repo_id']:<35}")
        print(f"  └─ {info['description']}")
    print("="*80 + "\n")

def check_model_exists(repo_id: str) -> bool:
    api = HfApi()
    try:
        api.model_info(repo_id)
        return True
    except Exception as e:
        print(f"Error checking {repo_id}: {e}", file=sys.stderr)
        return False

def download_model(key: str, repo_id: str, local_dir: Optional[str] = None) -> str:
    print(f"\n[+] Starting download for '{key}' ({repo_id})...")
    start = time.time()
    try:
        path = snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            resume_download=True,
            max_workers=4
        )
        elapsed = time.time() - start
        print(f"[✓] Successfully downloaded {repo_id} in {elapsed:.1f}s")
        print(f"    Local path: {path}")
        return path
    except Exception as e:
        print(f"[✗] Failed to download {repo_id}: {e}", file=sys.stderr)
        raise

def main() -> None:
    parser = argparse.ArgumentParser(description="Download and manage Qwen 3.8 MLX and DFlash models.")
    parser.add_argument("--list", action="store_true", help="List available model keys and information.")
    parser.add_argument("--model", type=str,
                        choices=list(MODELS.keys()) + ["dflash-27b-pair", "bench-arms", "max-throughput",
                                                       "all-recommended", "all"],
                        default="all-recommended", help="Model key or bundle to download.")
    parser.add_argument("--verify-only", action="store_true", help="Verify repo availability without downloading.")
    parser.add_argument("--local-dir", type=str, default=None, help="Custom local target directory.")
    args = parser.parse_args()

    if args.list:
        list_models()
        return

    targets = []
    if args.model == "all-recommended":
        # 3.8 target + its own MTP drafter + the already-cached 3.6 DFlash drafter
        # (the 3.6 *target* is deliberately not included: the cross-apply arm runs
        #  the 3.6 drafter against the 3.8 target).
        targets = ["qwen3.8-27b-4bit", "qwen3.8-27b-mtp-4bit", "qwen3.6-27b-dflash"]
    elif args.model == "bench-arms":
        targets = ["qwen3.8-27b-4bit", "qwen3.8-27b-mtp-4bit", "qwen3.6-27b-dflash",
                   "qwen3.8-27b-nvfp4", "qwen3.8-27b-mtp-nvfp4"]
    elif args.model == "max-throughput":
        targets = ["qwen3.8-27b-nvfp4", "qwen3.8-27b-mtp-nvfp4"]
    elif args.model == "dflash-27b-pair":
        targets = ["qwen3.6-27b-4bit", "qwen3.6-27b-dflash"]
    elif args.model == "all":
        targets = list(MODELS.keys())
    else:
        targets = [args.model]

    if args.verify_only:
        print(f"Verifying {len(targets)} repositories on Hugging Face...")
        all_ok = True
        for key in targets:
            repo_id = MODELS[key]["repo_id"]
            ok = check_model_exists(repo_id)
            status = "FOUND" if ok else "NOT FOUND"
            print(f" - [{status}] {key} -> {repo_id}")
            if not ok:
                all_ok = False
        if all_ok:
            print("[✓] All target repositories verified.")
        else:
            print("[✗] Some repositories could not be verified.")
            sys.exit(1)
        return

    print(f"Selected models to download: {targets}")
    for key in targets:
        repo_id = MODELS[key]["repo_id"]
        download_model(key, repo_id, local_dir=args.local_dir)

if __name__ == "__main__":
    main()
