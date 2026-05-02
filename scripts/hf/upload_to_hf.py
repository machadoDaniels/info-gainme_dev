#!/usr/bin/env python3
"""Upload outputs/ to a HuggingFace Dataset repository.

Usage:
    python scripts/hf/upload_to_hf.py
    python scripts/hf/upload_to_hf.py --repo-id akcit-rl/info-gainme
    python scripts/hf/upload_to_hf.py --outputs-dir outputs/ --num-workers 16
    python scripts/hf/upload_to_hf.py --dry-run

Requirements:
    pip install huggingface_hub
    HF_TOKEN must be set in .env or as an environment variable.
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Suppress internal 429/retry warnings — huggingface_hub handles backoff automatically
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

from huggingface_hub import HfApi

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
        "--num-workers",
        type=int,
        default=2,
        help="Parallel upload workers (keep low to avoid HF rate limits)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be uploaded without uploading",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Number of retry attempts on failure",
    )
    parser.add_argument(
        "--retry-sleep",
        type=int,
        default=300,
        help="Seconds to sleep between retries",
    )
    parser.add_argument(
        "--report-every",
        type=int,
        default=600,
        help="Seconds between progress reports (0 = disable)",
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

    repo_id = args.repo_id

    # --- count files ---
    print(f"Scanning {outputs_dir} ...")
    file_count, size_gb = count_files(outputs_dir)
    print(f"  {file_count:,} files  |  {size_gb:.2f} GB total")

    if args.dry_run:
        print(f"\n[Dry run] Would upload to: https://huggingface.co/datasets/{repo_id}")
        print(f"[Dry run] workers={args.num_workers}")
        return 0

    api = HfApi(token=token)

    # --- upload with retry ---
    print(f"\nUploading {outputs_dir}/ → {repo_id} ...")
    print("Upload is resumable — safe to interrupt and re-run.\n")

    report_every = args.report_every if args.report_every > 0 else 999999

    for attempt in range(1, args.max_retries + 1):
        try:
            api.upload_large_folder(
                folder_path=str(outputs_dir),
                repo_id=repo_id,
                repo_type="dataset",
                num_workers=args.num_workers,
                # conversations/** is bundled into conversations.zip per experiment
                # (see scripts/hf/zip_experiments.py). Uploading the loose tree
                # would explode the file count and trigger HF's 128 commits/hour
                # rate limit.
                ignore_patterns=[".cache/**", "**/conversations/**"],
                print_report_every=report_every,
            )
            break
        except Exception as exc:
            if attempt < args.max_retries:
                print(f"\nUpload error (attempt {attempt}/{args.max_retries}): {exc}")
                print(f"Retrying in {args.retry_sleep}s...")
                time.sleep(args.retry_sleep)
            else:
                print(f"\nUpload failed after {args.max_retries} attempts: {exc}")
                return 1

    print(f"\nDone! Dataset: https://huggingface.co/datasets/{repo_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
