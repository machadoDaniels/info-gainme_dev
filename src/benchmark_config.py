from dataclasses import dataclass, field
from typing import Optional, Dict, Any

from src.data_types import ObservabilityMode
from src.agents.llm_adapter import LLMConfig


@dataclass
class BenchmarkConfig:
    """Configuration for a single game run in the benchmark.

    This groups all agent LLM configs and orchestrator options.
    """

    seeker_config: LLMConfig
    oracle_config: LLMConfig
    pruner_config: LLMConfig

    observability_mode: ObservabilityMode = ObservabilityMode.FULLY_OBSERVED
    max_turns: int = 40

    # Optional experiment metadata
    experiment_name: Optional[str] = None
    tags: Dict[str, Any] = field(default_factory=dict)

 