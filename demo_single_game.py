#!/usr/bin/env python3
"""Demo for geographic dataset.

Loads cities from CSV and runs a single game.
"""

import logging
from os import getenv
from dotenv import load_dotenv
from src.logging_config import setup_logging
from src.orchestrator import Orchestrator
from src.agents.llm_adapter import LLMConfig
from src.data_types import ObservabilityMode
from src.domain.geo.loader import load_geo_candidates
from src.benchmark_config import BenchmarkConfig
from pathlib import Path
from random import choice
import os

logger = logging.getLogger(__name__)

OPENAI_API_KEY = getenv("OPENAI_API_KEY")
OBSERVABILITY_MODE = ObservabilityMode.PARTIALLY_OBSERVABLE
MAX_TURNS = 15
CSV_PATH = Path("data/top_10_pop_cities.csv")
OUTPUT_PATH = Path("outputs")
MODEL = "gpt-4o-mini"

os.makedirs(OUTPUT_PATH, exist_ok=True)


def main() -> None:
    """Run the benchmark demonstration."""
    load_dotenv()
    setup_logging()

    logger.info("Clary Quest - Geographic Benchmark")

    pool, domain_config = load_geo_candidates(csv_path=CSV_PATH)
    candidates = pool.get_active()
    logger.info("Candidate Pool: %d cities", len(candidates))

    llm_config = LLMConfig(model=MODEL, api_key=OPENAI_API_KEY)
    bm_config = BenchmarkConfig(
        seeker_config=llm_config,
        oracle_config=llm_config,
        pruner_config=llm_config,
        observability_mode=OBSERVABILITY_MODE,
        max_turns=MAX_TURNS,
    )

    target = choice(candidates)
    logger.info(
        "Target: %s (%s) | observability=%s | max_turns=%d | model=%s",
        target.label, target.id,
        bm_config.observability_mode.name, bm_config.max_turns, bm_config.seeker_config.model,
    )

    orchestrator = Orchestrator.from_target(
        target=target,
        pool=pool,
        seeker_config=bm_config.seeker_config,
        oracle_config=bm_config.oracle_config,
        pruner_config=bm_config.pruner_config,
        observability_mode=bm_config.observability_mode,
        max_turns=bm_config.max_turns,
    )

    logger.info("Starting benchmark run...")
    orchestrator.run(debug=True)

    if orchestrator.turns:
        last_turn = orchestrator.turns[-1]
        if last_turn.answer.game_over:
            logger.info("Success! Seeker found the target in %d turns.", len(orchestrator.turns))
        else:
            logger.info("Game ended after %d turns without finding the target.", len(orchestrator.turns))

    logger.info("Benchmark completed!")


if __name__ == "__main__":
    main()
