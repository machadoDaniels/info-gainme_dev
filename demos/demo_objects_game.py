#!/usr/bin/env python3
"""Demo for flat (non-hierarchical) object dataset.

Loads objects from CSV (objects_test.csv or objects_full.csv).
"""

import logging
import os
import sys
from os import getenv
from pathlib import Path
from random import choice

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.logging_config import setup_logging
from src.orchestrator import Orchestrator
from src.agents.llm_config import LLMConfig
from src.data_types import ObservabilityMode
from src.domain.objects import load_flat_object_candidates
from src.benchmark_config import BenchmarkConfig

logger = logging.getLogger(__name__)

OPENAI_API_KEY = getenv("OPENAI_API_KEY")
OBSERVABILITY_MODE = ObservabilityMode.FULLY_OBSERVABLE
MAX_TURNS = 20
OUTPUT_PATH = Path("outputs")
OBJECTS_CSV = Path("data/objects/objects_test.csv")  # or objects_full.csv
MODEL = "gpt-4o-mini"

os.makedirs(OUTPUT_PATH, exist_ok=True)


def main() -> None:
    """Run the benchmark with flat object dataset."""
    load_dotenv()
    setup_logging()

    logger.info("Clary Quest - Objects Benchmark (non-hierarchical)")

    pool, domain_config = load_flat_object_candidates(csv_path=OBJECTS_CSV)
    candidates = pool.get_active()

    categories: dict = {}
    for c in candidates:
        cat = c.attrs.get("category", "other")
        categories.setdefault(cat, []).append(c.label)

    logger.info("Candidate Pool: %d objects | categories: %s",
                len(candidates),
                {cat: len(items) for cat, items in sorted(categories.items())})

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
    logger.info(
        "Target: %s (%s) | category=%s | observability=%s | max_turns=%d | model=%s",
        target.label, target.id, target.attrs.get("category"),
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
