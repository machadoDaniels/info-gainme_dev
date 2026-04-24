#!/usr/bin/env python3
"""Delete question_evaluation.json files that contain connection errors.

This script finds all question_evaluation.json files and deletes those
that have connection errors in their question evaluations.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List


def has_connection_errors(evaluation_path: Path) -> bool:
    """Check if a question_evaluation.json file has connection errors.
    
    Args:
        evaluation_path: Path to question_evaluation.json file.
        
    Returns:
        True if file has connection errors, False otherwise.
    """
    try:
        with evaluation_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        
        turns_evaluation = data.get("turns_evaluation", [])
        
        for turn_eval in turns_evaluation:
            if "error" not in turn_eval:
                questions_eval = turn_eval.get("questions_evaluation", [])
                for q_eval in questions_eval:
                    error_msg = q_eval.get("error", "")
                    if error_msg and ("Connection error" in error_msg or "Connection" in error_msg.lower()):
                        return True
        
        return False
    except (json.JSONDecodeError, Exception) as e:
        print(f"Warning: Error reading {evaluation_path}: {e}", file=sys.stderr)
        return False


def find_evaluation_files(base_dir: Path) -> List[Path]:
    """Find all question_evaluation.json files.
    
    Args:
        base_dir: Base directory to search (e.g., outputs).
        
    Returns:
        List of paths to question_evaluation.json files.
    """
    pattern = "**/question_evaluation.json"
    return list(base_dir.glob(pattern))


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Delete question_evaluation.json files with connection errors"
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("outputs"),
        help="Base directory to search (default: ./outputs)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting"
    )
    
    args = parser.parse_args()
    
    print("🔍 Searching for question_evaluation.json files with connection errors")
    print(f"📁 Base directory: {args.base_dir}")
    print(f"🔍 Dry run: {args.dry_run}")
    
    if not args.base_dir.exists():
        print(f"Error: Base directory not found: {args.base_dir}", file=sys.stderr)
        return 1
    
    # Find all evaluation files
    evaluation_files = find_evaluation_files(args.base_dir)
    print(f"📊 Found {len(evaluation_files)} question_evaluation.json files")
    
    # Check for connection errors
    files_with_errors = []
    for eval_file in evaluation_files:
        if has_connection_errors(eval_file):
            files_with_errors.append(eval_file)
    
    print(f"⚠️  Found {len(files_with_errors)} files with connection errors")
    
    if not files_with_errors:
        print("✅ No files with connection errors found")
        return 0
    
    # Show files that would be deleted
    print("\n--- Files with connection errors ---")
    for i, eval_file in enumerate(files_with_errors, 1):
        print(f"[{i}/{len(files_with_errors)}] {eval_file}")
    
    if args.dry_run:
        print("\n🔍 Dry run: No files were deleted")
        return 0
    
    # Confirm deletion
    print(f"\n⚠️  About to delete {len(files_with_errors)} files. This cannot be undone!")
    response = input("Continue? (yes/no): ")
    if response.lower() not in ("yes", "y"):
        print("❌ Cancelled by user")
        return 0
    
    # Delete files
    deleted_count = 0
    error_count = 0
    for eval_file in files_with_errors:
        try:
            eval_file.unlink()
            deleted_count += 1
        except Exception as e:
            error_count += 1
            print(f"Error: Failed to delete {eval_file}: {e}", file=sys.stderr)
    
    print("\n--- Deletion Summary ---")
    print(f"✅ Deleted: {deleted_count}")
    print(f"❌ Errors: {error_count}")
    print(f"📊 Total processed: {len(files_with_errors)}")
    
    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    exit(main())

