#!/usr/bin/env python3
"""Demo for diseases dataset.

Loads diseases from CSV (diseases_test.csv or diseases_full.csv).
Preprocess first: python scripts/prepare_diseases_csv.py
"""

import logging
from os import getenv
from dotenv import load_dotenv
from random import choice

from src.logging_config import setup_logging
from src.orchestrator import Orchestrator
from src.agents.llm_config import LLMConfig
from src.data_types import ObservabilityMode
from src.domain.diseases import load_flat_disease_candidates
from src.benchmark_config import BenchmarkConfig
from pathlib import Path
import os

logger = logging.getLogger(__name__)

OPENAI_API_KEY = getenv("OPENAI_API_KEY")
OBSERVABILITY_MODE = ObservabilityMode.FULLY_OBSERVABLE
MAX_TURNS = 20
OUTPUT_PATH = Path("outputs")
DISEASES_CSV = Path("data/diseases/diseases_test.csv")
MODEL = "gpt-4o-mini"

os.makedirs(OUTPUT_PATH, exist_ok=True)


def main() -> None:
    """Run the benchmark with diseases dataset."""
    load_dotenv()
    setup_logging()

    logger.info("Clary Quest - Diseases Benchmark")

    pool, domain_config = load_flat_disease_candidates(csv_path=DISEASES_CSV)
    candidates = pool.get_active()
    logger.info("Candidate Pool: %d diseases", len(candidates))

    llm_config = LLMConfig(model=MODEL, api_key=OPENAI_API_KEY)
    bm_config = BenchmarkConfig(
        seeker_config=llm_config,
        oracle_config=llm_config,
        pruner_config=llm_config,
        observability_mode=OBSERVABILITY_MODE,
        max_turns=MAX_TURNS,
        domain_config=domain_config,
    )

    target = choice(candidates)
    symptoms = target.attrs.get("symptoms", [])
    logger.info(
        "Target: %s (%s) | symptoms=%d | observability=%s | max_turns=%d | model=%s",
        target.label, target.id, len(symptoms),
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
        domain_config=domain_config,
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
