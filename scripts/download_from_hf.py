#!/usr/bin/env python3
"""Download outputs/ from a HuggingFace Dataset repository.

Usage:
    python scripts/download_from_hf.py
    python scripts/download_from_hf.py --repo-id akcit-rl/info-gainme
    python scripts/download_from_hf.py --outputs-dir outputs/ --num-workers 16
    python scripts/download_from_hf.py --dry-run

Requirements:
    pip install huggingface_hub
    HF_TOKEN must be set in .env or as an environment variable.
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download outputs/ from a HuggingFace Dataset repository",
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
        help="Local directory to download into",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace token (defaults to HF_TOKEN env var)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Parallel download workers",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be downloaded without downloading",
    )

    args = parser.parse_args()

    token = args.token or os.getenv("HF_TOKEN")
    if not token:
        print(
            "Error: HuggingFace token not found.\n"
            "  Set HF_TOKEN in .env, export it as an env var, or pass --token <value>."
        )
        return 1

    outputs_dir = args.outputs_dir.resolve()
    repo_id = args.repo_id

    if args.dry_run:
        print(f"[Dry run] Would download from: https://huggingface.co/datasets/{repo_id}")
        print(f"[Dry run] Destination: {outputs_dir}")
        print(f"[Dry run] workers={args.num_workers}")
        return 0

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Error: huggingface_hub not installed. Run: pip install huggingface_hub")
        return 1

    outputs_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {repo_id} → {outputs_dir} ...")
    print("Download is resumable — safe to interrupt and re-run.\n")

    max_attempts = 200
    for attempt in range(1, max_attempts + 1):
        try:
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=str(outputs_dir),
                token=token,
                max_workers=args.num_workers,
            )
            break
        except Exception as exc:
            if attempt == max_attempts:
                print(f"\nDownload failed after {attempt} attempts: {exc}")
                return 1
            delay = min(300, 2 ** attempt + (30 if "429" in str(exc) else 0))
            print(f"\nAttempt {attempt} failed ({exc}). Retrying in {delay}s...")
            time.sleep(delay)

    print(f"\nDone! Local copy at: {outputs_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
