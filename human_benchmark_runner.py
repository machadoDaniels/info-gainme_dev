#!/usr/bin/env python3
"""Interactive human-seeker benchmark runner.

Runs benchmark games where a human plays the Seeker role via CLI, while
Oracle and Pruner remain LLM-powered (Qwen3-8B by default).  Results are
saved in the same output/ structure as automated benchmarks so they can be
analysed with the standard analysis pipeline.

Usage
-----
    python human_benchmark_runner.py --config configs/human/geo_20_human_fo.yaml
    python human_benchmark_runner.py --config configs/human/geo_20_human_fo.yaml --num-games 5
    python human_benchmark_runner.py --config configs/human/geo_20_human_fo.yaml --seed 42

The target is chosen randomly and NOT revealed to the player up front.
Press Ctrl+C at any time to stop after the current game finishes.
"""

import argparse
import logging
import random
import signal
import sys
from datetime import datetime
from os import getenv
from pathlib import Path

from dotenv import load_dotenv

from src.logging_config import setup_logging
from src.domain.geo.loader import load_geo_candidates
from src.domain.objects import load_flat_object_candidates
from src.domain.diseases import load_flat_disease_candidates
from src.utils.config_loader import load_benchmark_config
from src.benchmark import BenchmarkRunner

logger = logging.getLogger(__name__)

_STOP_REQUESTED = False


def _handle_sigint(sig, frame):
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    print("\n\n[!] Stopping after this game finishes…")


def main() -> None:
    # Shared with _handle_sigint: user asked to stop after current game (Ctrl+C).
    global _STOP_REQUESTED

    parser = argparse.ArgumentParser(description="Human-seeker benchmark runner")
    parser.add_argument(
        "--config", type=Path, default="configs/human/geo_20_human_fo.yaml",
        help="Path to human benchmark YAML config",
    )
    parser.add_argument(
        "--num-games", type=int, default=1,
        help="Number of games to play (default: 1). Use 0 to play all targets.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for target selection (default: random)",
    )
    parser.add_argument(
        "--servers-override", type=Path, default=None,
        help="Path to servers override YAML (for custom vLLM endpoints)",
    )
    args = parser.parse_args()

    load_dotenv()
    api_key = getenv("OPENAI_API_KEY")
    if not api_key:
        setup_logging()
        logger.error("OPENAI_API_KEY is not set in .env")
        sys.exit(1)

    try:
        benchmark_config, config = load_benchmark_config(
            args.config, api_key, args.servers_override
        )
    except Exception as exc:
        setup_logging()
        logger.error("Failed to load config: %s", exc)
        sys.exit(1)

    # Logging
    output_base = Path(config["output"]["base_dir"])
    log_file = (
        output_base
        / "logs"
        / f"{benchmark_config.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    setup_logging(log_file=log_file)

    # Load dataset
    dataset_cfg = config.get("dataset", {})
    dataset_type = dataset_cfg.get("type", "geo")
    csv_path = Path(dataset_cfg["csv_path"])

    if dataset_type == "objects":
        pool, _ = load_flat_object_candidates(csv_path=csv_path)
    elif dataset_type == "diseases":
        pool, _ = load_flat_disease_candidates(csv_path=csv_path)
    else:
        pool, _ = load_geo_candidates(csv_path=csv_path)

    # Random order of secret targets (full dataset or first k). Reproducible with --seed.
    all_candidates = list(pool.candidates)
    n_candidates = len(all_candidates)
    requested = args.num_games if args.num_games > 0 else n_candidates
    k = min(requested, n_candidates)
    rng = random.Random(args.seed)
    targets = rng.sample(all_candidates, k=k)

    signal.signal(signal.SIGINT, _handle_sigint)

    mode_name = benchmark_config.observability_mode.name
    print("\n" + "═" * 60)
    print("  INFO-GAINME  —  Human Baseline")
    print("═" * 60)
    print(f"  Domain      : {dataset_type}  ({n_candidates} candidates)")
    print(f"  Observability: {mode_name}")
    print(f"  Oracle/Pruner: {benchmark_config.oracle_config.model}")
    _seed_hint = f", seed={args.seed}" if args.seed is not None else ", seed=none"
    print(f"  Games planned: {k}  (target order: random{_seed_hint})")
    print(f"  Max turns/game: {benchmark_config.max_turns}")
    print("═" * 60)
    print("  Ask yes/no questions to identify the hidden target.")
    print("  Ctrl+C to stop after the current game.")
    print("═" * 60 + "\n")

    runner = BenchmarkRunner(config=benchmark_config, output_base=output_base)

    for game_idx, target in enumerate(targets, 1):
        if _STOP_REQUESTED:
            break

        print(f"\n{'▶' * 3}  GAME {game_idx}/{len(targets)}  {'◀' * 3}")
        print(f"  (Target is secret — find the {dataset_type} item!)\n")

        try:
            runner.run(
                pool=pool,
                targets=[target],
                runs_per_target=1,
                debug=True,
                max_workers=1,   # human games must be sequential
            )
        except KeyboardInterrupt:
            _STOP_REQUESTED = True
            print("\n[!] Game interrupted.")

        print(f"\n  ✓ Game {game_idx} done. Target was: {target.label}")

        if game_idx < len(targets) and not _STOP_REQUESTED:
            try:
                input("\n  Press Enter to start the next game (Ctrl+C to quit)…")
            except (EOFError, KeyboardInterrupt):
                break

    print("\n  All done. Results saved to:", runner._output_dir())


if __name__ == "__main__":
    main()
