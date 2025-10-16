#!/usr/bin/env python3
"""Run multi-game benchmark experiments.

This script runs a full benchmark across multiple target cities and multiple
runs per target, using the BenchmarkRunner to write incremental results to CSV.
"""

from os import getenv
from pathlib import Path
from dotenv import load_dotenv

from src.domain.geo.loader import load_geo_graph
from src.benchmark_config import BenchmarkConfig
from src.agents.llm_adapter import LLMConfig
from src.data_types import ObservabilityMode
from src.benchmark import BenchmarkRunner


# === Configuration ===
OPENAI_API_KEY = getenv("OPENAI_API_KEY")

# SEEKER_MODEL = "Qwen3-30B-A3B-Instruct-2507"
SEEKER_MODEL = "Qwen3-8B"
SEEKER_BASE_URL = "http://localhost:8000/v1"  # SSH tunnel: ssh -L 8000:localhost:8005 user_danielpedrozo@dgx-H100-02

ORACLE_MODEL = "gpt-4o-mini"
ORACLE_BASE_URL = None

PRUNER_MODEL = "gpt-4o-mini"
PRUNER_BASE_URL = None

OBSERVABILITY_MODE = ObservabilityMode.PARTIALLY_OBSERVABLE
MAX_TURNS = 15

# Dataset and targets
CSV_PATH = Path("data/top_10_pop_cities.csv")
NUM_TARGETS = None
RUNS_PER_TARGET = 2  # Number of runs per city

EXPERIMENT_NAME = "top10_test_po"

# Output
OUTPUT_BASE = Path("outputs")

DEBUG = True


def main() -> None:
    """Run the full benchmark experiment."""
    load_dotenv()
    
    print("🎯 Clary Quest - Multi-Game Benchmark")
    print("=" * 70)
    
    # Load knowledge graph
    graph = load_geo_graph(csv_path=CSV_PATH)
    active_nodes = graph.get_active_nodes()
    
    # Select target cities (deterministic by sorting by id)
    all_cities = sorted(
        [n for n in active_nodes if n.attrs.get("type") == "city"],
        key=lambda n: n.id
    )
    target_cities = all_cities[:NUM_TARGETS] if NUM_TARGETS else all_cities
    
    print(f"📊 Experiment Configuration:")
    print(f"   - Experiment Name: {EXPERIMENT_NAME}")
    print(f"   - Seeker Model: {SEEKER_MODEL}")
    print(f"   - Oracle Model: {ORACLE_MODEL}")
    print(f"   - Pruner Model: {PRUNER_MODEL}")
    print(f"   - Observability: {OBSERVABILITY_MODE.name}")
    print(f"   - Max Turns: {MAX_TURNS}")
    print(f"   - Total Nodes: {len(active_nodes)}")
    print(f"   - Target Cities: {len(target_cities)}")
    print(f"   - Runs per Target: {RUNS_PER_TARGET}")
    print(f"   - Total Games: {len(target_cities) * RUNS_PER_TARGET}")
    
    print(f"\n🎯 Target Cities:")
    for city in target_cities:
        print(f"   - {city.label} [{city.id}]")
    
    # Set up benchmark configuration
    config = BenchmarkConfig(
        seeker_config=LLMConfig(
            model=SEEKER_MODEL, 
            api_key=OPENAI_API_KEY, 
            base_url=SEEKER_BASE_URL,
            timeout=120.0  # Increase timeout for local models
        ),
        oracle_config=LLMConfig(model=ORACLE_MODEL, api_key=OPENAI_API_KEY, base_url=ORACLE_BASE_URL),
        pruner_config=LLMConfig(model=PRUNER_MODEL, api_key=OPENAI_API_KEY, base_url=PRUNER_BASE_URL),
        observability_mode=OBSERVABILITY_MODE,
        max_turns=MAX_TURNS,
        experiment_name=EXPERIMENT_NAME,
        save_conversations=True,
        save_graph_plots=True
    )
    
    # Create and run benchmark
    runner = BenchmarkRunner(config=config, output_base=OUTPUT_BASE)
    
    # Check for existing results to support resume
    csv_path = runner._csv_path()
    if csv_path.exists():
        completed_runs = runner._get_completed_runs(csv_path)
        total_runs = len(target_cities) * RUNS_PER_TARGET
        
        if completed_runs:
            print(f"\n🔄 Resuming experiment...")
            print(f"   - Already completed: {len(completed_runs)}/{total_runs} runs")
            print(f"   - Remaining: {total_runs - len(completed_runs)} runs")
        else:
            print(f"\n🚀 Starting NEW benchmark...")
            print(f"   (This will take a while as agents play {total_runs} games...)")
    else:
        print(f"\n🚀 Starting NEW benchmark...")
        print(f"   (This will take a while as agents play {len(target_cities) * RUNS_PER_TARGET} games...)")
    
    print()
    
    csv_path = runner.run(
        graph=graph,
        targets=target_cities,
        runs_per_target=RUNS_PER_TARGET,
        debug=DEBUG  # Set to True to see detailed turn-by-turn output
    )
    
    print(f"\n✅ Benchmark Complete!")
    print("=" * 70)
    print(f"📁 Results saved to: {csv_path}")
    print(f"📊 Total games played: {len(target_cities) * RUNS_PER_TARGET}")
    
    # Show quick preview of results
    print(f"\n📋 Results Preview (first 5 rows):")
    print("-" * 70)
    lines = Path(csv_path).read_text().splitlines()
    for line in lines[:6]:  # Header + first 5 rows
        print(line)
    
    print(f"\n💡 Next steps:")
    print(f"   - Analyze results: {csv_path}")
    print(f"   - Generate summary: [StatsAggregator - to be implemented]")
    print(f"   - Compare models: Run with different MODEL values")


if __name__ == "__main__":
    main()

