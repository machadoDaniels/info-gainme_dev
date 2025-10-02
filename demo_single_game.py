#!/usr/bin/env python3
"""Main entry point for running the benchmark.

This script demonstrates the complete benchmark flow:
1. Load a simple knowledge graph
2. Set up Seeker and Oracle agents
3. Run the benchmark without pruning
4. Display results
"""

from os import getenv
from dotenv import load_dotenv
from src.orchestrator import Orchestrator
from src.agents.llm_adapter import LLMConfig
from src.data_types import ObservabilityMode
from src.domain.geo.loader import load_geo_graph
from src.benchmark_config import BenchmarkConfig
from pathlib import Path
from random import choice


OPENAI_API_KEY = getenv("OPENAI_API_KEY")
OBSERVABILITY_MODE = ObservabilityMode.PARTIALLY_OBSERVED
MAX_TURNS = 15
CSV_PATH = Path("data/top_10_pop_cities.csv")
OUTPUT_PATH = Path("output/sample_graph.png")
MODEL = "gpt-4o-mini"


def main() -> None:
    """Run the benchmark demonstration."""
    load_dotenv()
    
    print("🎮 Clary Quest - Geographic Benchmark")
    print("=" * 50)
    
    # Create knowledge graph
    graph = load_geo_graph(csv_path=CSV_PATH)
    active_nodes = graph.get_active_nodes()
    max_turns = MAX_TURNS
    
    print(f"📍 Knowledge Graph: {len(active_nodes)} nodes")
    for node in sorted(active_nodes, key=lambda n: n.id):
        attrs_str = ", ".join(f"{k}={v}" for k, v in node.attrs.items())
        print(f"   - {node.id}: {node.label} ({attrs_str})")

    graph.plot(output_path=OUTPUT_PATH)

    # Set up configuration
    llm_config = LLMConfig(model=MODEL, api_key=OPENAI_API_KEY)
    bm_config = BenchmarkConfig(
        seeker_config=llm_config,
        oracle_config=llm_config,
        pruner_config=llm_config,
        observability_mode=OBSERVABILITY_MODE,
        max_turns=max_turns
    )
    
    # Choose target randomly
    active_cities = [node for node in active_nodes if node.attrs.get("type") == "city"]
    target_node = choice(active_cities)
    
    # Create orchestrator using factory method
    orchestrator = Orchestrator.from_target(
        target_node=target_node,
        graph=graph,
        seeker_config=bm_config.seeker_config,
        oracle_config=bm_config.oracle_config,
        pruner_config=bm_config.pruner_config,
        observability_mode=bm_config.observability_mode,
        max_turns=bm_config.max_turns
    )
    
    print(f"\n🎯 Configuration:")
    print(f"   - Target: {target_node.id} ({target_node.label})")
    print(f"   - Seeker observability: {bm_config.observability_mode.name}")
    print(f"   - Max turns: {bm_config.max_turns}")
    print(f"   - Model: {bm_config.seeker_config.model}")
    
    # Run the benchmark
    print(f"\n🚀 Starting benchmark run...")
    print("   (This may take a moment as agents generate responses...)\n")
    
    orchestrator.run(debug=True)
    
    # Final result
    if orchestrator.turns:
        last_turn = orchestrator.turns[-1]
        if "paris" in last_turn.question.text.lower() and last_turn.answer.text.lower().strip() == "yes":
            print(f"\n🎉 Success! Seeker found the target in {len(orchestrator.turns)} turns!")
        else:
            print(f"\n🤔 Game ended after {len(orchestrator.turns)} turns without finding the target.")
        
    print("\n🎯 Benchmark completed!")

if __name__ == "__main__":
    main()
