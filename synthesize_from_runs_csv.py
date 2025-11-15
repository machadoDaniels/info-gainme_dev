#!/usr/bin/env python3
"""Synthesize reasoning traces from seeker.json files listed in runs.csv.

This script reads a runs.csv file, extracts unique conversation_path values,
and processes all corresponding seeker.json files to create seeker_traces.json files.
"""

import argparse
import csv
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from os import getenv
from typing import List, Set
from dotenv import load_dotenv

from src.analysis.reasoning_synthesis import create_seeker_traces_file
from src.agents.llm_adapter import LLMConfig
from src.utils import ClaryLogger

logger = ClaryLogger.get_logger(__name__)


def extract_seeker_paths_from_csv(runs_csv: Path, outputs_base: Path) -> List[Path]:
    """Extract unique seeker.json paths from runs.csv.
    
    Args:
        runs_csv: Path to runs.csv file.
        outputs_base: Base directory for outputs (usually 'outputs').
        
    Returns:
        List of paths to seeker.json files.
    """
    if not runs_csv.exists():
        raise FileNotFoundError(f"runs.csv not found: {runs_csv}")
    
    seeker_paths: Set[Path] = set()
    
    with open(runs_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            conversation_path = row.get('conversation_path', '').strip()
            if not conversation_path:
                continue
            
            # Construir caminho completo: outputs_base / conversation_path / seeker.json
            seeker_path = outputs_base / conversation_path / "seeker.json"
            
            if seeker_path.exists():
                seeker_paths.add(seeker_path)
            else:
                logger.warning("seeker.json not found: %s", seeker_path)
    
    return sorted(seeker_paths)


def process_single_seeker(
    seeker_path: Path,
    llm_config: LLMConfig,
    force: bool = False
) -> dict:
    """Process a single seeker.json file.
    
    Args:
        seeker_path: Path to seeker.json file.
        llm_config: LLM configuration for synthesis.
        force: Whether to overwrite existing seeker_traces.json.
        
    Returns:
        Dictionary with processing result.
    """
    output_path = seeker_path.parent / "seeker_traces.json"
    
    # Check if file exists and is valid (not empty/incomplete)
    if output_path.exists() and not force:
        try:
            import json
            with open(output_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Check if file is valid (has turns and history)
                if data.get("total_turns", 0) > 0 and len(data.get("history", [])) > 0:
                    return {
                        "status": "skipped",
                        "seeker_path": str(seeker_path),
                        "output_path": str(output_path),
                        "reason": "already exists and valid"
                    }
                # File exists but is empty/incomplete - will reprocess
        except (json.JSONDecodeError, Exception):
            # File exists but is corrupted - will reprocess
            pass
    
    try:
        create_seeker_traces_file(seeker_path, output_path, llm_config)
        return {
            "status": "success",
            "seeker_path": str(seeker_path),
            "output_path": str(output_path),
            "size_bytes": output_path.stat().st_size
        }
    except Exception as e:
        return {
            "status": "error",
            "seeker_path": str(seeker_path),
            "output_path": str(output_path),
            "reason": str(e)
        }


def main():
    """Main entry point."""
    load_dotenv()
    ClaryLogger.configure()
    
    parser = argparse.ArgumentParser(
        description="Synthesize reasoning traces from seeker.json files in runs.csv"
    )
    parser.add_argument(
        "runs_csv",
        type=Path,
        help="Path to runs.csv file"
    )
    parser.add_argument(
        "--outputs-base",
        type=Path,
        default=Path("outputs"),
        help="Base directory for outputs (default: outputs)"
    )
    parser.add_argument(
        "--model",
        default="gpt-5.1-2025-11-13",
        help="LLM model to use for synthesis (default: gpt-5.1-2025-11-13)"
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
        default=None,
        help="Temperature for LLM generation (default: 0.0)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without actually processing"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.runs_csv.exists():
        logger.error("runs.csv not found: %s", args.runs_csv)
        return 1
    
    if not args.outputs_base.exists():
        logger.error("Outputs base directory not found: %s", args.outputs_base)
        return 1
    
    logger.info("🧠 Reasoning Trace Synthesis from runs.csv")
    logger.info("=" * 60)
    logger.info("📁 runs.csv: %s", args.runs_csv)
    logger.info("📁 Outputs base: %s", args.outputs_base)
    logger.info("🤖 LLM Model: %s", args.model)
    logger.info("🌡️  Temperature: %s", args.temperature)
    logger.info("⚙️  Max workers: %s", args.max_workers)
    logger.info("🔄 Force: %s", args.force)
    
    # Extract seeker.json paths
    try:
        seeker_paths = extract_seeker_paths_from_csv(args.runs_csv, args.outputs_base)
    except Exception as e:
        logger.error("Error extracting paths: %s", e)
        return 1
    
    if not seeker_paths:
        logger.error("No seeker.json files found in runs.csv")
        return 1
    
    logger.info("📊 Found %d unique seeker.json files", len(seeker_paths))
    
    # Dry run mode
    if args.dry_run:
        logger.info("\n🔍 Dry run - files that would be processed:")
        for i, path in enumerate(seeker_paths, 1):
            output_path = path.parent / "seeker_traces.json"
            exists = "✓" if output_path.exists() else "✗"
            logger.info("  %3d. %s [%s] %s", i, exists, path.parent.name, path.name)
        return 0
    
    # Create LLM config
    llm_config = LLMConfig(
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        temperature=args.temperature
    )
    
    # Process files
    logger.info("\n🔄 Processing files...")
    
    results = {
        "success": 0,
        "skipped": 0,
        "error": 0
    }
    
    if args.max_workers == 1:
        # Sequential processing
        for i, seeker_path in enumerate(seeker_paths, 1):
            logger.info("[%d/%d] Processing: %s", i, len(seeker_paths), seeker_path.parent.name)
            result = process_single_seeker(seeker_path, llm_config, args.force)
            results[result["status"]] += 1
            
            if result["status"] == "success":
                logger.info("  ✅ Created: %s (%d bytes)", result["output_path"], result.get("size_bytes", 0))
            elif result["status"] == "skipped":
                logger.info("  ⏭️  Skipped: %s", result["seeker_path"])
            else:
                logger.error("  ❌ Error: %s - %s", result["seeker_path"], result.get("reason", "Unknown"))
    else:
        # Parallel processing
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
                executor.submit(process_single_seeker, seeker_path, llm_config, args.force): seeker_path
                for seeker_path in seeker_paths
            }
            
            completed = 0
            for future in as_completed(futures):
                completed += 1
                seeker_path = futures[future]
                result = future.result()
                results[result["status"]] += 1
                
                logger.info("[%d/%d] %s: %s", 
                          completed, len(seeker_paths),
                          result["status"].upper(),
                          seeker_path.parent.name)
                
                if result["status"] == "error":
                    logger.error("  ❌ Error: %s", result.get("reason", "Unknown"))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 Summary:")
    logger.info("  ✅ Success: %d", results["success"])
    logger.info("  ⏭️  Skipped: %d", results["skipped"])
    logger.info("  ❌ Errors: %d", results["error"])
    logger.info("  📁 Total: %d", len(seeker_paths))
    logger.info("=" * 60)
    
    return 0 if results["error"] == 0 else 1


if __name__ == "__main__":
    exit(main())

