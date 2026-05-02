#!/usr/bin/env python3
"""Zip the conversations/ folder of each experiment so the HuggingFace upload
hits commit-rate limits on the order of "experiments" instead of "files".

Why: HF caps repository commits at ~128/hour. With ~467k tiny files in
outputs/, upload_large_folder packs ~20 files/commit → ~10k commits → days
of wall time. Wrapping each experiment's conversations/ in a single zip
collapses that to ~231 commits (one per experiment) and keeps files
streamable from HF.

Layout produced (conversations/ stays intact locally; only the zip gets uploaded):
    outputs/models/<triple>/<exp>/
        conversations.zip       ← new, contains everything under conversations/
        conversations/          ← untouched, ignored by upload (ignore_patterns)
        runs.csv                ← stays loose
        summary.json            ← stays loose
        variance.json           ← stays loose
        question_evaluations_summary.json   ← if present, stays loose
        oracle_judge_eval.json / pruner_judge_eval.json   ← if present, stays loose

Idempotent: skips experiments where conversations.zip exists and is newer
than every file in conversations/. Pass --force to rebuild.

Usage:
    python scripts/hf/zip_experiments.py                      # all experiments
    python scripts/hf/zip_experiments.py --dry-run            # show what would happen
    python scripts/hf/zip_experiments.py --outputs-dir outputs/
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def find_experiments(outputs_dir: Path) -> list[Path]:
    """Return every <triple>/<exp> dir that has a conversations/ subdir."""
    return sorted({p.parent for p in outputs_dir.glob("models/*/*/conversations")})


def needs_rebuild(zip_path: Path, conv_dir: Path) -> bool:
    """True iff conversations.zip is missing or older than any file under conv_dir."""
    if not zip_path.exists():
        return True
    zip_mtime = zip_path.stat().st_mtime
    for p in conv_dir.rglob("*"):
        if p.is_file() and p.stat().st_mtime > zip_mtime:
            return True
    return False


def zip_experiment(exp_dir: Path, force: bool = False) -> tuple[Path, str, int, int]:
    """Create conversations.zip for one experiment. Returns (path, status, n_files, bytes)."""
    conv_dir = exp_dir / "conversations"
    if not conv_dir.is_dir():
        return exp_dir, "skipped (no conversations/)", 0, 0
    zip_path = exp_dir / "conversations.zip"

    if not force and not needs_rebuild(zip_path, conv_dir):
        return exp_dir, "up-to-date", 0, zip_path.stat().st_size

    tmp_path = zip_path.with_suffix(".zip.tmp")
    n_files = 0
    with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for p in sorted(conv_dir.rglob("*")):
            if p.is_file():
                arcname = p.relative_to(exp_dir)
                zf.write(p, arcname=str(arcname))
                n_files += 1
    tmp_path.replace(zip_path)
    return exp_dir, "rebuilt" if zip_path.exists() else "created", n_files, zip_path.stat().st_size


def _human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-dir", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel zip workers (CPU bound on deflate). Default 4.")
    parser.add_argument("--force", action="store_true",
                        help="Rebuild zips even when up-to-date.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would happen, don't write zips.")
    args = parser.parse_args()

    outputs_dir = args.outputs_dir.resolve()
    if not outputs_dir.exists():
        print(f"ERROR: outputs dir not found: {outputs_dir}", file=sys.stderr)
        return 1

    experiments = find_experiments(outputs_dir)
    print(f"Found {len(experiments)} experiments under {outputs_dir}")

    if args.dry_run:
        n_to_build = 0
        for exp in experiments:
            zp = exp / "conversations.zip"
            cd = exp / "conversations"
            if args.force or needs_rebuild(zp, cd):
                n_files = sum(1 for p in cd.rglob("*") if p.is_file())
                n_to_build += 1
                print(f"  [build] {exp.relative_to(outputs_dir)}  ({n_files} files)")
        print(f"\nWould build {n_to_build} / {len(experiments)} zips.")
        return 0

    total_built = 0
    total_files = 0
    total_bytes = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(zip_experiment, e, args.force): e for e in experiments}
        for i, fut in enumerate(as_completed(futures), 1):
            exp_dir, status, n_files, n_bytes = fut.result()
            rel = exp_dir.relative_to(outputs_dir)
            if status in ("rebuilt", "created"):
                total_built += 1
                total_files += n_files
                total_bytes += n_bytes
                print(f"  [{i}/{len(experiments)}] {rel}  →  {n_files} files, {_human(n_bytes)}")
            elif status == "up-to-date":
                pass  # quiet
            else:
                print(f"  [{i}/{len(experiments)}] {rel}  {status}")

    print(f"\nBuilt {total_built} zips ({total_files} files, {_human(total_bytes)} total).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
