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
from src.runner import BenchmarkRunner
from src.graph import KnowledgeGraph, Node
from src.entropy import Entropy
from src.agents.llm_adapter import LLMAdapter, LLMConfig
from src.agents.seeker import SeekerAgent
from src.agents.oracle import OracleAgent
from src.agents.pruner import PrunerAgent
from src.data_types import ObservabilityMode
from src.domain.geo.loader import load_geo_graph
from pathlib import Path
from random import choice

def main() -> None:
    """Run the benchmark demonstration."""
    load_dotenv()
    
    print("🎮 Clary Quest - Geographic Benchmark")
    print("=" * 50)
    
    # Create knowledge graph
    graph = load_geo_graph(csv_path=Path("data/top_20_pop_cities.csv"))
    active_nodes = graph.get_active_nodes()
    
    print(f"📍 Knowledge Graph: {len(active_nodes)} nodes")
    for node in sorted(active_nodes, key=lambda n: n.id):
        attrs_str = ", ".join(f"{k}={v}" for k, v in node.attrs.items())
        print(f"   - {node.id}: {node.label} ({attrs_str})")

    graph.plot(output_path="output/sample_graph.png")

    # Set up LLM configuration
    api_key = getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Error: OPENAI_API_KEY not found in environment")
        print("   Please set your OpenAI API key in .env file or environment")
        return
    
    config = LLMConfig(
        model="gpt-4o-mini",
        api_key=api_key
    )
    
    # Create separate LLM adapters for each agent
    seeker_adapter = LLMAdapter(config)
    oracle_adapter = LLMAdapter(config)
    pruner_adapter = LLMAdapter(config, save_history=False)
    
    # Choose target randomly
    active_cities = [node for node in active_nodes if node.attrs.get("type") == "city"]
    target_node = choice(active_cities)
    
    # Create agents
    seeker = SeekerAgent(
        llm_adapter=seeker_adapter, 
        observability_mode=ObservabilityMode.FULLY_OBSERVED
        )
        
    oracle = OracleAgent(
        llm_adapter=oracle_adapter,
        target_node_id=target_node.id,
        target_node=target_node
    )
    
    pruner = PrunerAgent(llm_adapter=pruner_adapter)
    
    # Create entropy calculator
    entropy = Entropy()
    
    # Create benchmark runner
    runner = BenchmarkRunner(
        graph=graph,
        seeker=seeker,
        oracle=oracle,
        pruner=pruner,
        entropy=entropy,
        max_turns=40,
        h_threshold=None
    )
    
    print(f"\n🎯 Configuration:")
    print(f"   - Target: {oracle.target_node_id} ({target_node.label})")
    print(f"   - Seeker observability: {seeker.observability_mode.name}")
    print(f"   - Max turns: 7")
    print(f"   - Model: {config.model}")
    
    try:
        # Run the benchmark
        print(f"\n🚀 Starting benchmark run...")
        print("   (This may take a moment as agents generate responses...)\n")
        
        runner.run(debug=True)
        
        # Final result
        if runner.turns:
            last_turn = runner.turns[-1]
            if "paris" in last_turn.question.text.lower() and last_turn.answer.text.lower().strip() == "yes":
                print(f"\n🎉 Success! Seeker found the target in {len(runner.turns)} turns!")
            else:
                print(f"\n🤔 Game ended after {len(runner.turns)} turns without finding the target.")
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        print("   This might be due to:")
        print("   - OpenAI API issues or rate limits")
        print("   - Network connectivity problems")
        print("   - Invalid API key")
        print(f"   - Error details: {type(e).__name__}")
    
    print("\n🎯 Benchmark completed!")


if __name__ == "__main__":
    main()
