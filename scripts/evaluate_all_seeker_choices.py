#!/usr/bin/env python3
"""Evaluate Seeker's question choices for all conversations in a runs.csv.

This script reads a runs.csv file, extracts unique conversation directories,
and evaluates the Seeker's question choices for each conversation.
"""

import argparse
import csv
import json
import sys
import math
import statistics
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from os import getenv
from typing import List, Set, Dict, Any, Optional
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.analysis.question_evaluator import evaluate_seeker_choices
from src.agents.llm_config import LLMConfig
from src.utils import ClaryLogger

load_dotenv()

logger = ClaryLogger.get_logger(__name__)


def find_conversation_dirs_from_runs_csv(
    runs_csv_path: Path, 
    outputs_base_dir: Path
) -> List[Path]:
    """
    Reads a runs.csv file and returns a list of unique conversation directory paths.
    
    Args:
        runs_csv_path: Path to runs.csv file.
        outputs_base_dir: Base directory where conversation paths are relative to.
        
    Returns:
        List of conversation directory paths.
    """
    conversation_dirs = set()
    
    if not runs_csv_path.exists():
        raise FileNotFoundError(f"runs.csv not found: {runs_csv_path}")
    
    with runs_csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            conversation_path_str = row.get("conversation_path")
            if conversation_path_str:
                # conversation_path is relative to outputs_base_dir
                full_conversation_path = outputs_base_dir / conversation_path_str
                if full_conversation_path.exists():
                    conversation_dirs.add(full_conversation_path)
                else:
                    logger.warning("Conversation directory not found: %s", full_conversation_path)
    
    return sorted(list(conversation_dirs))


def load_evaluation_data(conversation_dir: Path) -> Optional[Dict[str, Any]]:
    """Load evaluation data from question_evaluation.json if it exists.
    
    Args:
        conversation_dir: Path to conversation directory.
        
    Returns:
        Evaluation data dictionary or None if file doesn't exist or is invalid.
    """
    output_path = conversation_dir / "question_evaluation.json"
    if not output_path.exists():
        return None
    
    try:
        with output_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            # Check if file is valid
            if (data.get("turns_evaluation") and 
                len(data.get("turns_evaluation", [])) > 0 and
                data.get("summary")):
                # Calculate missing fields for old files
                summary = data.get("summary", {})
                turns_evaluation = data.get("turns_evaluation", [])
                
                # Calculate avg_questions_considered_per_turn if missing
                if "avg_questions_considered_per_turn" not in summary:
                    valid_turns_considered = [t.get("total_considered") for t in turns_evaluation 
                                             if "error" not in t and t.get("total_considered") is not None]
                    summary["avg_questions_considered_per_turn"] = (
                        sum(valid_turns_considered) / len(valid_turns_considered) 
                        if valid_turns_considered else 0.0
                    )
                
                # Calculate total_connection_errors if missing
                if "total_connection_errors" not in summary:
                    connection_errors = 0
                    for turn_eval in turns_evaluation:
                        if "error" not in turn_eval:
                            questions_eval = turn_eval.get("questions_evaluation", [])
                            for q_eval in questions_eval:
                                error_msg = q_eval.get("error", "")
                                if error_msg and ("Connection error" in error_msg or "Connection" in error_msg.lower()):
                                    connection_errors += 1
                    summary["total_connection_errors"] = connection_errors
                
                return data
    except (json.JSONDecodeError, Exception) as e:
        logger.debug("Failed to load evaluation data from %s: %s", output_path, e)
    
    return None


def process_single_conversation(
    conversation_dir: Path,
    graph_csv_path: Path,
    oracle_config: LLMConfig,
    pruner_config: LLMConfig,
    force: bool = False
) -> Dict[str, Any]:
    """Process a single conversation directory.
    
    Args:
        conversation_dir: Path to conversation directory.
        graph_csv_path: Path to CSV file for loading knowledge graph.
        oracle_config: LLM configuration for Oracle simulation.
        pruner_config: LLM configuration for Pruner simulation.
        force: Whether to overwrite existing question_evaluation.json.
        
    Returns:
        Dictionary with processing result, including evaluation data if available.
    """
    output_path = conversation_dir / "question_evaluation.json"
    
    # Check if file exists and is valid (not empty/incomplete)
    if output_path.exists() and not force:
        evaluation_data = load_evaluation_data(conversation_dir)
        if evaluation_data:
            return {
                "status": "skipped",
                "conversation_dir": str(conversation_dir),
                "output_path": str(output_path),
                "reason": "already exists and valid",
                "evaluation_data": evaluation_data
            }
    
    # Check if required files exist
    required_files = ["seeker_traces.json", "turns.jsonl", "metadata.json"]
    missing_files = [f for f in required_files if not (conversation_dir / f).exists()]
    if missing_files:
        return {
            "status": "error",
            "conversation_dir": str(conversation_dir),
            "output_path": str(output_path),
            "reason": f"Missing required files: {', '.join(missing_files)}"
        }
    
    try:
        results = evaluate_seeker_choices(
            conversation_dir,
            graph_csv_path,
            oracle_config,
            pruner_config
        )
        
        # Save results
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        return {
            "status": "success",
            "conversation_dir": str(conversation_dir),
            "output_path": str(output_path),
            "size_bytes": output_path.stat().st_size,
            "turns_evaluated": results["summary"]["total_turns_evaluated"],
            "optimal_choices": results["summary"]["optimal_choices"],
            "optimal_choice_rate": results["summary"]["optimal_choice_rate"],
            "evaluation_data": results
        }
    except Exception as e:
        logger.error("Error processing %s: %s", conversation_dir, e, exc_info=True)
        return {
            "status": "error",
            "conversation_dir": str(conversation_dir),
            "output_path": str(output_path),
            "reason": str(e)
        }


def calculate_aggregate_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate aggregate statistics from all evaluation results.
    
    Args:
        results: List of processing results, each containing evaluation_data if available.
        
    Returns:
        Dictionary with aggregate statistics.
    """
    # Filter results with evaluation data
    evaluations = []
    for r in results:
        if r.get("status") in ("success", "skipped") and r.get("evaluation_data"):
            evaluations.append(r["evaluation_data"])
    
    if not evaluations:
        return {
            "total_conversations": 0,
            "total_evaluated": 0,
            "total_skipped": 0,
            "total_errors": 0
        }
    
    # Aggregate statistics
    total_turns_evaluated = sum(e["summary"]["total_turns_evaluated"] for e in evaluations)
    total_optimal_choices = sum(e["summary"]["optimal_choices"] for e in evaluations)
    
    # Calculate averages
    optimal_choice_rates = [e["summary"]["optimal_choice_rate"] for e in evaluations]
    avg_optimal_choice_rate = sum(optimal_choice_rates) / len(optimal_choice_rates) if optimal_choice_rates else 0.0
    
    # Info gain statistics
    # Filter out None and negative values (negative values indicate evaluation errors)
    avg_chosen_ig_list = [e["summary"]["avg_chosen_info_gain"] for e in evaluations 
                         if e["summary"].get("avg_chosen_info_gain") is not None 
                         and e["summary"]["avg_chosen_info_gain"] >= 0.0]
    avg_optimal_ig_list = [e["summary"]["avg_optimal_info_gain"] for e in evaluations 
                          if e["summary"].get("avg_optimal_info_gain") is not None 
                          and e["summary"]["avg_optimal_info_gain"] >= 0.0]
    
    avg_chosen_info_gain = sum(avg_chosen_ig_list) / len(avg_chosen_ig_list) if avg_chosen_ig_list else 0.0
    avg_optimal_info_gain = sum(avg_optimal_ig_list) / len(avg_optimal_ig_list) if avg_optimal_ig_list else 0.0
    
    # Average questions considered per turn
    avg_questions_considered_list = []
    for e in evaluations:
        avg_questions = e["summary"].get("avg_questions_considered_per_turn")
        if avg_questions is None:
            avg_questions = 0.0
        avg_questions_considered_list.append(avg_questions)
    avg_questions_considered_per_turn = (sum(avg_questions_considered_list) / len(avg_questions_considered_list) 
                                        if avg_questions_considered_list else 0.0)
    
    # Total connection errors (use 0 as default for old files)
    total_connection_errors = sum(e["summary"].get("total_connection_errors", 0) for e in evaluations)
    
    # Group by target city
    by_target: Dict[str, Dict[str, Any]] = {}
    for e in evaluations:
        target_id = e["target"]["id"]
        target_label = e["target"]["label"]
        
        if target_id not in by_target:
            by_target[target_id] = {
                "target_id": target_id,
                "target_label": target_label,
                "conversations": 0,
                "total_turns_evaluated": 0,
                "total_optimal_choices": 0,
                "optimal_choice_rates": [],
                "avg_chosen_info_gains": [],
                "avg_optimal_info_gains": [],
                "avg_questions_considered": [],
                "total_connection_errors": 0
            }
        
        by_target[target_id]["conversations"] += 1
        by_target[target_id]["total_turns_evaluated"] += e["summary"]["total_turns_evaluated"]
        by_target[target_id]["total_optimal_choices"] += e["summary"]["optimal_choices"]
        by_target[target_id]["optimal_choice_rates"].append(e["summary"]["optimal_choice_rate"])
        
        # Filter out negative values (indicate evaluation errors)
        chosen_ig = e["summary"].get("avg_chosen_info_gain")
        if chosen_ig is not None and chosen_ig >= 0.0:
            by_target[target_id]["avg_chosen_info_gains"].append(chosen_ig)
        optimal_ig = e["summary"].get("avg_optimal_info_gain")
        if optimal_ig is not None and optimal_ig >= 0.0:
            by_target[target_id]["avg_optimal_info_gains"].append(optimal_ig)
        
        # Average questions considered per turn (use 0.0 as default for old files)
        avg_questions = e["summary"].get("avg_questions_considered_per_turn")
        if avg_questions is None:
            avg_questions = 0.0
        by_target[target_id]["avg_questions_considered"].append(avg_questions)
        
        # Connection errors (use 0 as default for old files)
        by_target[target_id]["total_connection_errors"] += e["summary"].get("total_connection_errors", 0)
    
    # Calculate per-target statistics
    by_target_summary = {}
    for target_id, data in by_target.items():
        rates = data["optimal_choice_rates"]
        chosen_igs = data["avg_chosen_info_gains"]
        optimal_igs = data["avg_optimal_info_gains"]
        avg_questions_list = data["avg_questions_considered"]
        
        # Calculate means
        avg_rate = sum(rates) / len(rates) if rates else 0.0
        avg_chosen_ig = sum(chosen_igs) / len(chosen_igs) if chosen_igs else None
        avg_optimal_ig = sum(optimal_igs) / len(optimal_igs) if optimal_igs else None
        avg_questions = sum(avg_questions_list) / len(avg_questions_list) if avg_questions_list else None
        
        # Calculate stds (populacional para cada target, já que temos todas as conversas)
        std_rate = statistics.pstdev(rates) if len(rates) > 1 else 0.0
        std_chosen_ig = statistics.pstdev(chosen_igs) if chosen_igs and len(chosen_igs) > 1 else None
        std_optimal_ig = statistics.pstdev(optimal_igs) if optimal_igs and len(optimal_igs) > 1 else None
        std_questions = statistics.pstdev(avg_questions_list) if avg_questions_list and len(avg_questions_list) > 1 else None
        
        by_target_summary[target_id] = {
            "target_label": data["target_label"],
            "conversations": data["conversations"],
            "total_turns_evaluated": data["total_turns_evaluated"],
            "total_optimal_choices": data["total_optimal_choices"],
            "avg_optimal_choice_rate": avg_rate,
            "std_optimal_choice_rate": std_rate,
            "avg_chosen_info_gain": avg_chosen_ig,
            "std_chosen_info_gain": std_chosen_ig,
            "avg_optimal_info_gain": avg_optimal_ig,
            "std_optimal_info_gain": std_optimal_ig,
            "avg_questions_considered_per_turn": avg_questions,
            "std_questions_considered_per_turn": std_questions,
            "total_connection_errors": data["total_connection_errors"]
        }
    
    # Count statuses
    success_count = sum(1 for r in results if r["status"] == "success")
    skipped_count = sum(1 for r in results if r["status"] == "skipped")
    error_count = sum(1 for r in results if r["status"] == "error")
    
    # Calculate global stds and SEs using hierarchical approach (by target)
    num_targets = len(by_target_summary)
    
    # Collect target-level means for std calculation
    target_rates = [target["avg_optimal_choice_rate"] for target in by_target_summary.values()]
    target_chosen_igs = [target["avg_chosen_info_gain"] for target in by_target_summary.values() 
                         if target["avg_chosen_info_gain"] is not None]
    target_optimal_igs = [target["avg_optimal_info_gain"] for target in by_target_summary.values() 
                          if target["avg_optimal_info_gain"] is not None]
    target_questions = [target["avg_questions_considered_per_turn"] for target in by_target_summary.values() 
                       if target["avg_questions_considered_per_turn"] is not None]
    
    # Calculate stds (amostral para inferência estatística)
    std_optimal_choice_rate = statistics.stdev(target_rates) if len(target_rates) > 1 else 0.0
    std_chosen_info_gain = statistics.stdev(target_chosen_igs) if len(target_chosen_igs) > 1 else None
    std_optimal_info_gain = statistics.stdev(target_optimal_igs) if len(target_optimal_igs) > 1 else None
    std_questions_considered_per_turn = statistics.stdev(target_questions) if len(target_questions) > 1 else None
    
    # Calculate SEs (hierarchical: std / sqrt(num_targets))
    se_optimal_choice_rate = std_optimal_choice_rate / math.sqrt(num_targets) if num_targets > 1 else 0.0
    se_chosen_info_gain = std_chosen_info_gain / math.sqrt(len(target_chosen_igs)) if std_chosen_info_gain is not None and target_chosen_igs and len(target_chosen_igs) > 1 else None
    se_optimal_info_gain = std_optimal_info_gain / math.sqrt(len(target_optimal_igs)) if std_optimal_info_gain is not None and target_optimal_igs and len(target_optimal_igs) > 1 else None
    se_questions_considered_per_turn = std_questions_considered_per_turn / math.sqrt(len(target_questions)) if std_questions_considered_per_turn is not None and target_questions and len(target_questions) > 1 else None
    
    return {
        "total_conversations": len(results),
        "total_evaluated": success_count + skipped_count,
        "total_success": success_count,
        "total_skipped": skipped_count,
        "total_errors": error_count,
        "aggregate_statistics": {
            "total_turns_evaluated": total_turns_evaluated,
            "total_optimal_choices": total_optimal_choices,
            "avg_optimal_choice_rate": avg_optimal_choice_rate,
            "std_optimal_choice_rate": std_optimal_choice_rate,
            "se_optimal_choice_rate": se_optimal_choice_rate,
            "avg_chosen_info_gain": avg_chosen_info_gain,
            "std_chosen_info_gain": std_chosen_info_gain,
            "se_chosen_info_gain": se_chosen_info_gain,
            "avg_optimal_info_gain": avg_optimal_info_gain,
            "std_optimal_info_gain": std_optimal_info_gain,
            "se_optimal_info_gain": se_optimal_info_gain,
            "avg_questions_considered_per_turn": avg_questions_considered_per_turn,
            "std_questions_considered_per_turn": std_questions_considered_per_turn,
            "se_questions_considered_per_turn": se_questions_considered_per_turn,
            "total_connection_errors": total_connection_errors
        },
        "by_target": by_target_summary
    }


def main():
    """Main entry point for batch evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate Seeker's question choices for all conversations in a runs.csv"
    )
    parser.add_argument(
        "runs_csv_path",
        type=Path,
        help="Path to the runs.csv file"
    )
    parser.add_argument(
        "--outputs-base-dir",
        type=Path,
        default=Path("outputs"),
        help="Base directory where conversation paths in runs.csv are relative to (default: ./outputs)"
    )
    parser.add_argument(
        "--graph-csv",
        type=Path,
        default=Path("data/top_40_pop_cities.csv"),
        help="Path to CSV file for loading knowledge graph (default: data/top_40_pop_cities.csv)"
    )
    parser.add_argument(
        "--oracle-model",
        type=str,
        default="Qwen3-8B",
        help="LLM model to use for Oracle simulation (default: Qwen3-8B)"
    )
    parser.add_argument(
        "--pruner-model",
        type=str,
        default="Qwen3-8B",
        help="LLM model to use for Pruner simulation (default: Qwen3-8B)"
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:8000/v1",
        help="Base URL for LLM API (default: http://localhost:8000/v1)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for LLM (defaults to OPENAI_API_KEY env var if not provided)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Temperature for LLM generation (default: None, API decides)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing question_evaluation.json files, even if valid"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Maximum number of parallel workers (default: 1)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without actually processing"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    ClaryLogger.configure()
    
    logger.info("🔍 Evaluating Seeker Question Choices from runs.csv")
    logger.info("=" * 60)
    logger.info("📁 runs.csv: %s", args.runs_csv_path)
    logger.info("📁 Outputs base: %s", args.outputs_base_dir)
    logger.info("📊 Graph CSV: %s", args.graph_csv)
    logger.info("🤖 Oracle Model: %s", args.oracle_model)
    logger.info("🤖 Pruner Model: %s", args.pruner_model)
    if args.base_url:
        logger.info("🌐 Base URL: %s", args.base_url)
    if args.temperature is not None:
        logger.info("🌡️  Temperature: %s", args.temperature)
    logger.info("⚙️  Max workers: %s", args.max_workers)
    logger.info("🔄 Force: %s", args.force)
    
    # Find all conversation directories
    try:
        conversation_dirs = find_conversation_dirs_from_runs_csv(
            args.runs_csv_path, 
            args.outputs_base_dir
        )
    except FileNotFoundError as e:
        logger.error("Error: %s", e)
        return 1
    
    if not conversation_dirs:
        logger.error("No conversation directories found from %s", args.runs_csv_path)
        return 1
    
    logger.info("📊 Found %d unique conversation directories", len(conversation_dirs))
    
    if args.dry_run:
        logger.info("\n--- Dry Run: Conversations to be processed ---")
        for i, conv_dir in enumerate(conversation_dirs, 1):
            output_path = conv_dir / "question_evaluation.json"
            status = "Process"
            if output_path.exists():
                try:
                    with output_path.open("r", encoding="utf-8") as f:
                        data = json.load(f)
                        if (data.get("turns_evaluation") and 
                            len(data.get("turns_evaluation", [])) > 0 and
                            data.get("summary")):
                            status = "Skip (exists and valid)"
                        else:
                            status = "Process (exists but invalid)"
                except (json.JSONDecodeError, Exception):
                    status = "Process (exists but corrupted)"
            logger.info("[%d/%d] %s: %s", i, len(conversation_dirs), status, conv_dir)
        logger.info("\nDry run complete. No conversations were actually processed.")
        return 0
    
    logger.info("\n🔄 Processing conversations...")
    
    # Create LLM configs for simulation
    oracle_config = LLMConfig(
        model=args.oracle_model,
        api_key=args.api_key or getenv("OPENAI_API_KEY"),
        base_url=args.base_url,
        temperature=args.temperature
    )
    pruner_config = LLMConfig(
        model=args.pruner_model,
        api_key=args.api_key or getenv("OPENAI_API_KEY"),
        base_url=args.base_url,
        temperature=args.temperature
    )
    
    results = []
    if args.max_workers > 1:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
                executor.submit(
                    process_single_conversation,
                    conv_dir,
                    args.graph_csv,
                    oracle_config,
                    pruner_config,
                    args.force
                ): conv_dir 
                for conv_dir in conversation_dirs
            }
            for i, future in enumerate(as_completed(futures), 1):
                result = future.result()
                results.append(result)
                status_emoji = "✅" if result["status"] == "success" else "⏭️" if result["status"] == "skipped" else "❌"
                logger.info(
                    "[%d/%d] %s %s: %s", 
                    i, 
                    len(conversation_dirs), 
                    status_emoji,
                    result["status"].capitalize(), 
                    result["conversation_dir"]
                )
    else:
        try:
            from tqdm import tqdm
        except ImportError:
            def tqdm(iterable, *args, **kwargs):
                return iterable
        
        for i, conv_dir in enumerate(tqdm(conversation_dirs, desc="Evaluating conversations", unit="conv"), 1):
            result = process_single_conversation(
                conv_dir,
                args.graph_csv,
                oracle_config,
                pruner_config,
                args.force
            )
            results.append(result)
            status_emoji = "✅" if result["status"] == "success" else "⏭️" if result["status"] == "skipped" else "❌"
            logger.info(
                "[%d/%d] %s %s: %s", 
                i, 
                len(conversation_dirs), 
                status_emoji,
                result["status"].capitalize(), 
                result["conversation_dir"]
            )
    
    # Calculate aggregate summary
    aggregate_summary = calculate_aggregate_summary(results)
    
    # Log summary
    logger.info("\n--- Evaluation Summary ---")
    logger.info("✅ Sucessos: %d", aggregate_summary["total_success"])
    logger.info("⏭️  Pulados (já existentes e válidos): %d", aggregate_summary["total_skipped"])
    logger.info("❌ Erros: %d", aggregate_summary["total_errors"])
    logger.info("Total processado/tentado: %d", aggregate_summary["total_conversations"])
    
    if aggregate_summary["total_evaluated"] > 0:
        stats = aggregate_summary["aggregate_statistics"]
        logger.info("\n--- Aggregate Statistics (from successful evaluations) ---")
        logger.info("Total turns evaluated: %d", stats["total_turns_evaluated"])
        logger.info("Total optimal choices: %d", stats["total_optimal_choices"])
        logger.info("Average optimal choice rate: %.2f%%", stats["avg_optimal_choice_rate"] * 100)
        logger.info("Average chosen info gain: %.4f", stats["avg_chosen_info_gain"])
        logger.info("Average optimal info gain: %.4f", stats["avg_optimal_info_gain"])
        logger.info("Average questions considered per turn: %.2f", stats["avg_questions_considered_per_turn"])
        logger.info("Total connection errors: %d", stats["total_connection_errors"])
    
    # Save aggregate summary to JSON file
    summary_output_path = args.runs_csv_path.parent / "question_evaluations_summary.json"
    try:
        with summary_output_path.open("w", encoding="utf-8") as f:
            json.dump(aggregate_summary, f, indent=2, ensure_ascii=False)
        logger.info("\n💾 Aggregate summary saved to: %s", summary_output_path)
    except Exception as e:
        logger.error("Failed to save aggregate summary: %s", e, exc_info=True)
    
    if aggregate_summary["total_errors"] > 0:
        logger.error("Algumas conversas falharam na avaliação. Verifique os logs acima para detalhes.")
        return 1
    return 0


if __name__ == "__main__":
    exit(main())

