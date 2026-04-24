#!/usr/bin/env python3
"""Upload outputs/ to a HuggingFace Dataset repository.

Usage:
    python scripts/upload_to_hf.py
    python scripts/upload_to_hf.py --repo-id akcit-rl/info-gainme
    python scripts/upload_to_hf.py --outputs-dir outputs/ --num-workers 16
    python scripts/upload_to_hf.py --dry-run

Requirements:
    pip install huggingface_hub
    HF_TOKEN must be set in .env or as an environment variable.
"""

import argparse
import os
import sys
from pathlib import Path

# Allow running from repo root or scripts/
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


IGNORE_DIRS = {".cache"}


def count_files(directory: Path) -> tuple[int, float]:
    """Return (file_count, total_size_gb), skipping .cache/ and transient lock files."""
    files = [
        f for f in directory.rglob("*")
        if f.is_file() and not any(p.name in IGNORE_DIRS for p in f.parents)
    ]
    total = 0
    for f in files:
        try:
            total += f.stat().st_size
        except FileNotFoundError:
            pass
    return len(files), total / 1e9


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Upload outputs/ to a HuggingFace Dataset repository",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="akcit-rl/info-gainme",
        help="HuggingFace repository ID",
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("outputs"),
        help="Local outputs directory to upload",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace write token (defaults to HF_TOKEN env var)",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        default=False,
        help="Make the repository public (default: private)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Parallel upload workers",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be uploaded without uploading",
    )

    args = parser.parse_args()

    # --- resolve token ---
    token = args.token or os.getenv("HF_TOKEN")
    if not token:
        print(
            "Error: HuggingFace token not found.\n"
            "  Set HF_TOKEN in .env, export it as an env var, or pass --token <value>."
        )
        return 1

    # --- validate outputs dir ---
    outputs_dir = args.outputs_dir.resolve()
    if not outputs_dir.exists():
        print(f"Error: outputs directory not found: {outputs_dir}")
        return 1

    private = not args.public
    repo_id = args.repo_id

    # --- count files ---
    print(f"Scanning {outputs_dir} ...")
    file_count, size_gb = count_files(outputs_dir)
    print(f"  {file_count:,} files  |  {size_gb:.2f} GB total")

    if args.dry_run:
        print(f"\n[Dry run] Would upload to: https://huggingface.co/datasets/{repo_id}")
        print(f"[Dry run] private={private}, workers={args.num_workers}")
        return 0

    # --- import huggingface_hub ---
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("Error: huggingface_hub not installed. Run: pip install huggingface_hub")
        return 1

    api = HfApi(token=token)

    # --- create repo (idempotent) ---
    print(f"\nEnsuring dataset repo exists: {repo_id} (private={private}) ...")
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=private,
            exist_ok=True,
        )
        print(f"Repo ready: https://huggingface.co/datasets/{repo_id}")
    except Exception as exc:
        print(f"Error creating/accessing repo: {exc}")
        return 1

    # --- upload ---
    print(f"\nUploading {outputs_dir}/ → {repo_id} ...")
    print("Upload is resumable — safe to interrupt and re-run.\n")
    try:
        api.upload_large_folder(
            folder_path=str(outputs_dir),
            repo_id=repo_id,
            repo_type="dataset",
            num_workers=args.num_workers,
            ignore_patterns=[".cache/**"],
        )
    except Exception as exc:
        print(f"\nUpload failed: {exc}")
        return 1

    print(f"\nDone! Dataset: https://huggingface.co/datasets/{repo_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
