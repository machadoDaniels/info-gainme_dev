#!/Users/daniel2/Documents/AKCIT-RL/clary_quest/.conda/bin/python
"""Batch synthesize reasoning traces for all conversations.

This script processes all seeker.json files in conversation directories
and creates seeker_traces.json files for each one.
"""

import argparse
from pathlib import Path
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from os import getenv

from src.analysis.reasoning_synthesis import create_seeker_traces_file
from src.agents.llm_adapter import LLMConfig
from src.utils import ClaryLogger

logger = ClaryLogger.get_logger(__name__)


def find_seeker_json_files(conversations_dir: Path) -> List[Path]:
    """Find all seeker.json files in conversation directories.
    
    Args:
        conversations_dir: Path to the conversations directory.
        
    Returns:
        List of paths to seeker.json files.
        
    Raises:
        FileNotFoundError: If conversations directory doesn't exist.
    """
    if not conversations_dir.exists():
        raise FileNotFoundError(f"Conversations directory not found: {conversations_dir}")
    
    seeker_files = []
    
    # Look for game_* directories
    for game_dir in sorted(conversations_dir.glob("game_*")):
        if game_dir.is_dir():
            seeker_file = game_dir / "seeker.json"
            if seeker_file.exists():
                seeker_files.append(seeker_file)
    
    return seeker_files


def process_single_conversation(
    seeker_file: Path, 
    llm_config: LLMConfig,
    force: bool = False
) -> dict:
    """Process a single conversation file.
    
    Args:
        seeker_file: Path to seeker.json file.
        llm_config: LLM configuration for synthesis.
        force: Whether to overwrite existing seeker_traces.json files.
        
    Returns:
        Dictionary with processing result.
    """
    output_file = seeker_file.parent / "seeker_traces.json"
    
    # Skip if output already exists and not forcing
    if output_file.exists() and not force:
        return {
            "file": str(seeker_file),
            "status": "skipped",
            "reason": "Output file already exists"
        }
    
    try:
        start_time = time.time()
        create_seeker_traces_file(seeker_file, output_file, llm_config)
        duration = time.time() - start_time
        
        # Check if file was created successfully
        if output_file.exists():
            size = output_file.stat().st_size
            return {
                "file": str(seeker_file),
                "status": "success",
                "duration": duration,
                "output_size": size
            }
        else:
            return {
                "file": str(seeker_file),
                "status": "error",
                "reason": "Output file was not created"
            }
            
    except Exception as e:
        return {
            "file": str(seeker_file),
            "status": "error",
            "reason": str(e)
        }


def main():
    """Main entry point for batch processing."""
    parser = argparse.ArgumentParser(
        description="Batch synthesize reasoning traces for all conversations"
    )
    parser.add_argument(
        "conversations_dir",
        type=Path,
        help="Path to conversations directory containing game_* subdirectories"
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="LLM model to use for synthesis (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "--base-url",
        help="Custom base URL for LLM (e.g., for local vLLM server)"
    )
    parser.add_argument(
        "--api-key",
        help="API key for LLM (defaults to OPENAI_API_KEY env var)",
        default=getenv("OPENAI_API_KEY")
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing seeker_traces.json files"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Maximum number of parallel workers (default: 1)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for LLM generation (default: 0.0)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without actually processing"
    )
    
    args = parser.parse_args()
    
    # Import here to ensure env vars are available
    from dotenv import load_dotenv
    load_dotenv()
    
    # Configure logging
    ClaryLogger.configure()
    
    logger.info("🧠 Batch Reasoning Trace Synthesis")
    logger.info("=" * 50)
    
    # Find all seeker.json files
    try:
        seeker_files = find_seeker_json_files(args.conversations_dir)
    except FileNotFoundError as e:
        logger.error("Error: %s", e)
        return 1
    
    if not seeker_files:
        logger.error("No seeker.json files found in %s", args.conversations_dir)
        return 1
    
    logger.info("📁 Conversations directory: %s", args.conversations_dir)
    logger.info("📊 Found %d seeker.json files", len(seeker_files))
    logger.info("🤖 LLM Model: %s", args.model)
    logger.info("🌡️  Temperature: %s", args.temperature)
    logger.info("🔄 Force overwrite: %s", args.force)
    logger.info("👥 Max workers: %s", args.max_workers)
    
    if args.dry_run:
        logger.info("🔍 DRY RUN - Files that would be processed:")
        missing_count = 0
        for seeker_file in seeker_files:
            output_file = seeker_file.parent / "seeker_traces.json"
            exists = output_file.exists()
            if not exists:
                missing_count += 1
            
            status = "✅ exists" if exists else "❌ missing"
            game_dir = seeker_file.parent.name
            logger.info("   %s/seeker.json -> seeker_traces.json [%s]", game_dir, status)
        
        logger.info("📊 Summary: %d/%d files need processing", missing_count, len(seeker_files))
        return 0
    
    # Create LLM config
    llm_config = LLMConfig(
        model=args.model,
        api_key=args.api_key or getenv("OPENAI_API_KEY"),
        base_url=args.base_url,
        temperature=args.temperature
    )
    
    logger.info("🚀 Starting batch processing...")
    start_time = time.time()
    
    # Process files
    results = []
    
    if args.max_workers == 1:
        # Sequential processing
        for i, seeker_file in enumerate(seeker_files, 1):
            logger.info("📝 Processing %d/%d: %s", i, len(seeker_files), seeker_file.parent.name)
            result = process_single_conversation(seeker_file, llm_config, args.force)
            results.append(result)
            
            if result["status"] == "success":
                logger.info("   ✅ Success (%.1fs, %,d bytes)", result['duration'], result['output_size'])
            elif result["status"] == "skipped":
                logger.info("   ⏭️  Skipped: %s", result['reason'])
            else:
                logger.error("   ❌ Error: %s", result['reason'])
    else:
        # Parallel processing with ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(process_single_conversation, seeker_file, llm_config, args.force): seeker_file
                for seeker_file in seeker_files
            }
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_file):
                seeker_file = future_to_file[future]
                completed += 1
                
                try:
                    result = future.result()
                    results.append(result)
                    
                    if result["status"] == "success":
                        logger.info("✅ [%d/%d] %s (%.1fs)", completed, len(seeker_files), seeker_file.parent.name, result['duration'])
                    elif result["status"] == "skipped":
                        logger.info("⏭️  [%d/%d] %s (skipped)", completed, len(seeker_files), seeker_file.parent.name)
                    else:
                        logger.error("❌ [%d/%d] %s - %s", completed, len(seeker_files), seeker_file.parent.name, result['reason'])
                        
                except Exception as e:
                    logger.error("❌ [%d/%d] %s - Exception: %s", completed, len(seeker_files), seeker_file.parent.name, e)
                    results.append({
                        "file": str(seeker_file),
                        "status": "error",
                        "reason": f"Exception: {e}"
                    })
    
    # Summary
    total_time = time.time() - start_time
    successful = sum(1 for r in results if r["status"] == "success")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    errors = sum(1 for r in results if r["status"] == "error")
    
    logger.info("📊 Batch Processing Complete!")
    logger.info("=" * 50)
    logger.info("⏱️  Total time: %.1fs", total_time)
    logger.info("📁 Total files: %d", len(seeker_files))
    logger.info("✅ Successful: %d", successful)
    logger.info("⏭️  Skipped: %d", skipped)
    logger.info("❌ Errors: %d", errors)
    
    if errors > 0:
        logger.error("❌ Files with errors:")
        for result in results:
            if result["status"] == "error":
                logger.error("   - %s: %s", result['file'], result['reason'])
    
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    exit(main())
