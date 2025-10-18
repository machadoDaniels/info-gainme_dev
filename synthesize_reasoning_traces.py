#!/Users/daniel2/Documents/AKCIT-RL/clary_quest/.conda/bin/python
"""Synthesize reasoning traces from SeekerAgent conversations.

This script reads seeker.json files containing reasoning_history and creates
synthesized seeker_traces.json files with turn-based structure:
- Each turn contains: turn_index, original_reasoning, reasoning_trace (summary, options_considered, decision_rationale), question, oracle_answer
- Output format similar to seeker.json but with synthesized reasoning traces
"""

import argparse
from os import getenv
from pathlib import Path
from dotenv import load_dotenv

from src.analysis.reasoning_synthesis import create_seeker_traces_file
from src.agents.llm_adapter import LLMConfig
from src.utils import ClaryLogger

logger = ClaryLogger.get_logger(__name__)


def main():
    """Main entry point for reasoning trace synthesis."""
    load_dotenv()
    
    # Configure logging
    ClaryLogger.configure()
    
    parser = argparse.ArgumentParser(
        description="Synthesize SeekerAgent reasoning traces using LLM analysis"
    )
    parser.add_argument(
        "input_file",
        type=Path,
        help="Path to seeker.json file to process"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Output path for seeker_traces.json (default: same directory as input)"
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
        default=getenv("OPENAI_API_KEY"),
        help="API key for LLM (defaults to OPENAI_API_KEY env var)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for LLM generation (default: 0.0)"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not args.input_file.exists():
        logger.error("Input file does not exist: %s", args.input_file)
        return 1
    
    if args.input_file.name != "seeker.json":
        logger.warning("Input file is not named 'seeker.json': %s", args.input_file)
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        # Default: same directory as input, with seeker_traces.json name
        output_path = args.input_file.parent / "seeker_traces.json"
    
    logger.info("🧠 SeekerAgent Reasoning Trace Synthesis")
    logger.info("=" * 50)
    logger.info("📁 Input file: %s", args.input_file)
    logger.info("📁 Output file: %s", output_path)
    logger.info("🤖 LLM Model: %s", args.model)
    logger.info("🌡️  Temperature: %s", args.temperature)
    
    # Create LLM config
    llm_config = LLMConfig(
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        temperature=args.temperature
    )
    
    try:
        logger.info("🔄 Processing reasoning traces...")
        create_seeker_traces_file(args.input_file, output_path, llm_config)
        logger.info("✅ Successfully created synthesized traces: %s", output_path)
        
        # Show file size info
        output_size = output_path.stat().st_size
        logger.info("📊 Output file size: %,d bytes", output_size)
        
        return 0
        
    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
        return 1
    except Exception as e:
        logger.error("Error processing file: %s", e)
        return 1


if __name__ == "__main__":
    exit(main())
