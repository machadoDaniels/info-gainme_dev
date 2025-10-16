"""Orchestrator for a single benchmark game.

Coordinates turns between `SeekerAgent`, `OracleAgent`, and `PrunerAgent`, 
computes entropy metrics with `Entropy`, and records `TurnState`. 
Orchestrates a single game from start to finish.
"""

from __future__ import annotations

from typing import List, Optional
from datetime import datetime
from pathlib import Path
import json

from .entropy import Entropy
from .graph import KnowledgeGraph, Node
from .data_types import TurnState, Question, Answer, ObservabilityMode
from .agents.seeker import SeekerAgent
from .agents.oracle import OracleAgent
from .agents.pruner import PrunerAgent
from .agents.llm_adapter import LLMAdapter, LLMConfig


class Orchestrator:
    """Orchestrates a single benchmark game.

    Args:
        graph: Knowledge graph with nodes/edges and pruning state.
        seeker: SeekerAgent instance responsible for generating questions.
        oracle: OracleAgent instance responsible for generating answers.
        entropy: Entropy helper for computing entropy and information gain.
        max_turns: Maximum number of turns to execute.
        pruner: Optional PrunerAgent for deterministic pruning.
    """

    def __init__(
        self,
        *,
        graph: KnowledgeGraph,
        seeker: SeekerAgent,
        oracle: OracleAgent,
        pruner: PrunerAgent,
        entropy: Entropy,
        max_turns: int
    ) -> None:
        if graph is None:
            raise ValueError("graph cannot be None")
        if seeker is None:
            raise ValueError("seeker cannot be None")
        if oracle is None:
            raise ValueError("oracle cannot be None")
        if entropy is None:
            raise ValueError("entropy cannot be None")
        if max_turns <= 0:
            raise ValueError("max_turns must be > 0")

        self._graph = graph
        self._seeker = seeker
        self._oracle = oracle
        self._entropy = entropy
        self._max_turns = max_turns
        self._pruner = pruner

        self._current_turn: int = 0
        self._turns: List[TurnState] = []

    @property
    def turns(self) -> List[TurnState]:
        return self._turns

    @property
    def current_turn(self) -> int:
        return self._current_turn
    
    @classmethod
    def from_target(
        cls,
        *,
        target_node: Node,
        graph: KnowledgeGraph,
        seeker_config: LLMConfig,
        oracle_config: LLMConfig,
        pruner_config: LLMConfig,
        observability_mode: ObservabilityMode = ObservabilityMode.FULLY_OBSERVED,
        max_turns: int = 40,
    ) -> Orchestrator:
        """Factory method to create an Orchestrator with all agents configured.
        
        Args:
            target_node: The target node that the Seeker must find.
            graph: Knowledge graph for the game.
            seeker_config: LLM configuration for SeekerAgent.
            oracle_config: LLM configuration for OracleAgent.
            pruner_config: LLM configuration for PrunerAgent.
            observability_mode: How much graph info the Seeker can see.
            max_turns: Maximum number of turns before game ends.
            
        Returns:
            Fully configured Orchestrator ready to run.
            
        Example:
            orch = Orchestrator.from_target(
                target_node=tokyo,
                graph=geo_graph,
                seeker_config=LLMConfig(model="gpt-4o-mini"),
                oracle_config=LLMConfig(model="gpt-4o-mini"),
                pruner_config=LLMConfig(model="gpt-4o-mini"),
                observability_mode=ObservabilityMode.FULLY_OBSERVED,
                max_turns=15
            )
            orch.run(debug=True)
        """
        # Create LLM adapters for each agent
        seeker_adapter = LLMAdapter(seeker_config)
        oracle_adapter = LLMAdapter(oracle_config)
        pruner_adapter = LLMAdapter(pruner_config, save_history=True)  # Save history but use stateless calls
        
        # Create agents
        seeker = SeekerAgent(
            llm_adapter=seeker_adapter,
            observability_mode=observability_mode
        )
        
        oracle = OracleAgent(
            llm_adapter=oracle_adapter,
            target_node_id=target_node.id,
            target_node=target_node
        )
        
        pruner = PrunerAgent(llm_adapter=pruner_adapter)
        
        # Create entropy calculator
        entropy = Entropy()
        
        # Return configured orchestrator
        return cls(
            graph=graph,
            seeker=seeker,
            oracle=oracle,
            pruner=pruner,
            entropy=entropy,
            max_turns=max_turns
        )

    def show_turn(self, turn: TurnState) -> None:
        """Show the turn state with detailed information."""
        print(f"\n🔄 Turn {turn.turn_index}")
        print("=" * 50)
        
        # Show question and answer
        print(f"❓ Question: {turn.question.text}")
        print(f"💬 Answer: {turn.answer.text}")
        print(f"✅ Compliant: {turn.answer.compliant}")

        # Show pruned nodes
        print(f"🔍 Pruning Rationale: {turn.pruning_result.rationale}")
        print(f"🔍 Pruned Nodes: {turn.pruning_result.pruned_ids}")

        # Show current active nodes count
        active_leaf_count = len(self._graph.get_active_leaf_nodes())
        print(f"🎯 Active Leaf Nodes: {active_leaf_count}") 
        active_count = len(self._graph.get_active_nodes())
        print(f"🎯 Active Nodes: {active_count}")

        # Show entropy metrics
        print(f"📊 Entropy: {turn.h_before:.4f} → {turn.h_after:.4f}")
        print(f"📈 Info Gain: {turn.info_gain:.4f}")
        print(f"✂️  Pruned: {turn.pruned_count} nodes")
    

        # Show progress
        progress = (turn.turn_index / self._max_turns) * 100
        print(f"⏳ Progress: {progress:.1f}% ({turn.turn_index}/{self._max_turns})")
        
        print("-" * 50)

    def run(self, debug: bool = False, save_plots: bool = False, plots_dir: Optional[Path] = None) -> None:
        """Execute the benchmark loop.

        Delegates pruning decisions to the configured `PrunerAgent`, which is LLM-driven. 
        Entropy is computed before and after each turn. Timestamps are captured for each turn.
        
        Args:
            debug: Show detailed turn-by-turn information.
            save_plots: Save graph plot for each turn.
            plots_dir: Directory to save plots. If None and save_plots=True, uses default.
        """
        # Apaga os plots existentes no diretório de plots
        if save_plots:
            plots_dir = plots_dir or Path("outputs/plots")
            for plot_file in plots_dir.glob("*.png"):
                print(f"Cleaning up residual plot file: {plot_file}")
                plot_file.unlink()
        
        for turn in range(1, self._max_turns + 1):
            self._current_turn = turn
            
            # Start timestamp
            turn_start = datetime.now()

            active_nodes = self._graph.get_active_nodes()
            active_nodes_before = len(active_nodes)
            
            # Compute entropy only over leaf nodes (cities) since only they can be targets
            active_leaf_nodes = self._graph.get_active_leaf_nodes()
            active_leaf_nodes_before = len(active_leaf_nodes)
            h_before = self._entropy.compute(active_leaf_nodes)

            # Prepare textual graph view and inject once if fully observed
            graph_text = self._graph.graph_to_text()
            if turn == 1 and self._seeker.observability_mode.name == "FULLY_OBSERVED":
                self._seeker.add_initial_graph(graph_text, turn)
            
            # Save plot before pruning if requested
            if save_plots:
                plot_dir = plots_dir
                plot_dir.mkdir(parents=True, exist_ok=True)
                plot_path = plot_dir / f"turn_{turn:02d}.png"
                self._graph.plot(
                    output_path=str(plot_path),
                    title=f"Turn {turn} ({active_nodes_before} nodes)"
                )

            # Seeker asks a question
            question: Question = self._seeker.question_to_oracle(active_nodes, turn)

            # Oracle receives and answers
            self._oracle.add_seeker_question(question)
            answer: Answer = self._oracle.answer_seeker()

            # Apply pruning via PrunerAgent (LLM-driven)
            pruned_count = 0
            pruning_result = self._pruner.analyze_and_prune(
                graph_text=graph_text,
                turn_index=turn,
                question=question,
                answer=answer,
                active_leaf_nodes=active_leaf_nodes,
            )
            if pruning_result.pruned_ids:
                self._graph.apply_pruning(pruning_result.pruned_ids)
                pruned_count = len(pruning_result.pruned_ids)
                if debug:
                    print(f"🔍 Pruning: {pruning_result.rationale}\n Pruned IDs: {pruning_result.pruned_ids}")

            # Seeker integrates the oracle's answer and (optionally) context from graph text
            self._seeker.add_oracle_answer_and_pruning(
                answer,
                graph_text=graph_text if self._seeker.observability_mode.name == "FULLY_OBSERVED" else None,
                turn=turn,
            )

            # Compute entropy after pruning (only over leaf nodes/cities)
            active_nodes_after = len(self._graph.get_active_nodes())
            active_leaf_nodes_after = self._graph.get_active_leaf_nodes()
            active_leaf_nodes_after_count = len(active_leaf_nodes_after)
            
            # If game is won, entropy should be 0 (target found, no uncertainty)
            if answer.game_over:
                h_after = 0.0
            else:
                h_after = self._entropy.compute(active_leaf_nodes_after)
                
            info_gain = self._entropy.info_gain(h_before, h_after)
            
            # End timestamp
            turn_end = datetime.now()
            duration = (turn_end - turn_start).total_seconds()

            self._turns.append(
                TurnState(
                    turn_index=turn,
                    h_before=h_before,
                    h_after=h_after,
                    info_gain=info_gain,
                    pruned_count=pruned_count,
                    question=question,
                    answer=answer,
                    pruning_result=pruning_result,
                    active_nodes_before=active_nodes_before,
                    active_nodes_after=active_nodes_after,
                    active_leaf_nodes_before=active_leaf_nodes_before,
                    active_leaf_nodes_after=active_leaf_nodes_after_count,
                    timestamp_start=turn_start.isoformat(),
                    timestamp_end=turn_end.isoformat(),
                    duration_seconds=round(duration, 6),
                    graph_snapshot=graph_text,
                )
            )

            if debug:
                self.show_turn(self._turns[-1])
            
            # Check if game is over
            if answer.game_over:
                if debug:
                    print(f"\n🎉 Game Over! Seeker found the target in {turn} turns!")
                break
        
        # Show final summary if debug is enabled
        if debug and self._turns:
            print("\n🏁 Benchmark Complete!")
            print("=" * 50)
            summary = self.get_summary()
            print(f"📊 Total Turns: {summary['turns']}")
            print(f"📈 Start Entropy: {summary['h_start']:.4f}")
            print(f"📉 End Entropy: {summary['h_end']:.4f}")
            print(f"🎯 Total Info Gain: {summary['total_info_gain']:.4f}")
            print(f"📊 Avg Info Gain/Turn: {summary['avg_info_gain_per_turn']:.4f}")
            print("=" * 50)


    def get_summary(self) -> dict:
        """Return a simple summary of the run."""
        total_info_gain = sum(t.info_gain for t in self._turns)
        num_turns = len(self._turns)
        
        return {
            "turns": num_turns,
            "current_turn": self._current_turn,
            "h_start": self._turns[0].h_before if self._turns else None,
            "h_end": self._turns[-1].h_after if self._turns else None,
            "total_info_gain": total_info_gain,
            "avg_info_gain_per_turn": total_info_gain / num_turns if num_turns > 0 else 0.0,
        }
    
    def export_conversation(self, output_dir: Path) -> None:
        """Export complete game conversation for all agents.
        
        Saves each agent's LLMAdapter.history to separate JSON files,
        along with game metadata and turn-by-turn details in JSONL format.
        
        Args:
            output_dir: Directory to save conversation files (e.g., game_001_Beijing_city19332/).
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Save Seeker history
        if self._seeker._llm_adapter._save_history:
            seeker_data = {
                "agent_type": "seeker",
                "config": {
                    "model": self._seeker._llm_adapter.config.model,
                    "temperature": self._seeker._llm_adapter.config.temperature,
                    "max_tokens": self._seeker._llm_adapter.config.max_tokens,
                    "base_url": self._seeker._llm_adapter.config.base_url,
                },
                "observability_mode": self._seeker.observability_mode.name,
                "total_messages": len(self._seeker._llm_adapter.history),
                "history": self._seeker._llm_adapter.history
            }
            with (output_dir / "seeker.json").open("w", encoding="utf-8") as f:
                json.dump(seeker_data, f, indent=2, ensure_ascii=False)
        
        # 2. Save Oracle history
        if self._oracle._llm_adapter._save_history:
            oracle_data = {
                "agent_type": "oracle",
                "config": {
                    "model": self._oracle._llm_adapter.config.model,
                    "temperature": self._oracle._llm_adapter.config.temperature,
                    "max_tokens": self._oracle._llm_adapter.config.max_tokens,
                    "base_url": self._oracle._llm_adapter.config.base_url,
                },
                "target": {
                    "id": self._oracle._target_node_id,
                    "label": self._oracle._target_node.label if self._oracle._target_node else None,
                    "attrs": dict(self._oracle._target_node.attrs) if self._oracle._target_node else {}
                },
                "total_messages": len(self._oracle._llm_adapter.history),
                "history": self._oracle._llm_adapter.history
            }
            with (output_dir / "oracle.json").open("w", encoding="utf-8") as f:
                json.dump(oracle_data, f, indent=2, ensure_ascii=False)
        
        # 3. Save Pruner history (or note if disabled)
        pruner_data = {
            "agent_type": "pruner",
            "config": {
                "model": self._pruner.llm_adapter.config.model,
                "temperature": self._pruner.llm_adapter.config.temperature,
                "max_tokens": self._pruner.llm_adapter.config.max_tokens,
                "base_url": self._pruner.llm_adapter.config.base_url,
            },
            "save_history": self._pruner.llm_adapter._save_history,
            "total_calls": len(self.turns),
        }
        
        if self._pruner.llm_adapter._save_history:
            pruner_data["total_messages"] = len(self._pruner.llm_adapter.history)
            pruner_data["history"] = self._pruner.llm_adapter.history
        else:
            pruner_data["note"] = "Pruner was configured with save_history=False. No conversation history available."
            pruner_data["history"] = []
        
        with (output_dir / "pruner.json").open("w", encoding="utf-8") as f:
            json.dump(pruner_data, f, indent=2, ensure_ascii=False)
        
        # 4. Save metadata
        summary = self.get_summary()
        win = any(t.answer.game_over for t in self.turns)
        compliance_rate = (
            sum(1 for t in self.turns if t.answer.compliant) / len(self.turns)
        ) if self.turns else 0.0
        
        total_pruned = sum(t.pruned_count for t in self._turns)
        initial_nodes = len(self._graph.nodes)
        final_active = len(self._graph.get_active_nodes())
        
        metadata = {
            "game_id": None,  # Will be set by BenchmarkRunner
            "timestamp": datetime.now().isoformat(),
            "target": {
                "id": self._oracle._target_node_id,
                "label": self._oracle._target_node.label if self._oracle._target_node else None,
                "attrs": dict(self._oracle._target_node.attrs) if self._oracle._target_node else {}
            },
            "config": {
                "experiment_name": None,  # Will be set by BenchmarkRunner
                "observability_mode": self._seeker.observability_mode.name,
                "max_turns": self._max_turns,
                "models": {
                    "seeker": self._seeker._llm_adapter.config.model,
                    "oracle": self._oracle._llm_adapter.config.model,
                    "pruner": self._pruner.llm_adapter.config.model
                }
            },
            "results": {
                "turns_played": len(self.turns),
                "win": win,
                "h_start": summary["h_start"],
                "h_end": summary["h_end"],
                "total_info_gain": summary["total_info_gain"],
                "avg_info_gain_per_turn": summary["avg_info_gain_per_turn"],
                "compliance_rate": round(compliance_rate, 4),
                "final_active_nodes": final_active
            },
            "graph_stats": {
                "initial_nodes": initial_nodes,
                "final_nodes": final_active,
                "total_pruned": total_pruned,
                "pruning_efficiency": round(total_pruned / initial_nodes, 4) if initial_nodes > 0 else 0
            }
        }
        
        with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        # 5. Save turn-by-turn details in JSONL format
        with (output_dir / "turns.jsonl").open("w", encoding="utf-8") as f:
            for turn_state in self.turns:
                turn_data = turn_state.to_export_dict()
                f.write(json.dumps(turn_data, ensure_ascii=False) + "\n")
    


