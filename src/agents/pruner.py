"""PrunerAgent implementation for deterministic rule-based pruning.

The PrunerAgent analyzes question-answer pairs and determines which nodes
should be pruned from the knowledge graph based on logical rules.
"""

from __future__ import annotations

from typing import Set

from ..data_types import Answer, PruningResult, Question, PrunerResponse
from ..graph import Node
from .llm_adapter import LLMAdapter
from ..prompts import get_pruner_system_prompt
from ..utils.utils import parse_first_json_object


class PrunerAgent:
    """Agent responsible for deterministic pruning based on question-answer analysis.
    
    The PrunerAgent uses rule-based logic to determine which nodes should be
    pruned from the knowledge graph based on the Seeker's question and the
    Oracle's answer. It maintains its own LLM adapter for potential future
    AI-assisted pruning decisions.
    """
    
    def __init__(self, llm_adapter: LLMAdapter) -> None:
        """Initialize the PrunerAgent.
        
        Args:
            llm_adapter: LLM adapter for AI-assisted pruning decisions.
        """
        self.llm_adapter = llm_adapter
        self.pruning_count = 0
        
        # Add system prompt to history for export (if save_history is enabled)
        if self.llm_adapter._save_history:
            system_prompt = get_pruner_system_prompt()
            self.llm_adapter.append_history("system", system_prompt)
    
    def analyze_and_prune(
        self,
        graph_text: str,
        turn_index: int,
        question: Question,
        answer: Answer,
        *,
        active_leaf_nodes: Set[Node] = None,
        target_node_id: str = None,
    ) -> PruningResult:
        """Delegate pruning decision to the LLM using graph text and turn context.

        The LLM must respond with a strict JSON object:
            {"rationale": "...", "pruned_ids": ["node:id", ...]}

        Args:
            graph_text: Textual representation of the active graph (graph_to_text).
            turn_index: Current turn number (1-based).
            question: Seeker's question.
            answer: Oracle's answer.
            active_leaf_nodes: Set of active leaf nodes (cities) from the graph.
            target_node_id: ID of the target node that must NEVER be pruned.
        Returns:
            PruningResult with pruned node IDs and rationale. Falls back to no
            pruning if parsing fails or the model returns an invalid response.
            
        Note:
            Uses stateless LLM calls (each request is independent) but saves to history
            for export and analysis purposes.
            CRITICAL: Only CITY nodes can be pruned, as only cities can be targets.
            CRITICAL: The target node will NEVER be included in pruned_ids.
        """
        system_prompt = get_pruner_system_prompt()

        user_prompt = (
            "GRAPH:\n" + graph_text + "\n\n" +
            f"TURN: {turn_index}\n" +
            f"QUESTION: {question.text}\n" +
            f"ANSWER: {answer.text}\n\n" 
        )

        # Build stateless messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        # Save request to history for export
        if self.llm_adapter._save_history:
            self.llm_adapter.append_history("user", user_prompt)

        # Make stateless call (don't use accumulated history)
        # Note: add_to_history defaults to True, so it will auto-save the response
        reply = self.llm_adapter.generate(
            messages=messages, 
            stateless=True,  # Use only these messages, not history
            # temperature=0.0,
            # add_to_history=True is default, so response is auto-saved
        )
        try:
            pruning_response = PrunerResponse.model_validate_json(reply)
        except Exception as e:
            raise ValueError(f"Invalid LLM response (non-JSON): {e}. Response: {reply}")
            
        if pruning_response is None:
            return PruningResult(pruned_ids=set(), rationale="Invalid LLM response (non-JSON)")
        
        pruned_ids_list = pruning_response.pruned_ids
        rationale = pruning_response.rationale

        # Filter to only include CITY nodes (targets can only be cities)
        candidate_ids = {str(x) for x in pruned_ids_list if isinstance(x, (str, int))}
        city_ids = {node_id for node_id in candidate_ids if node_id.startswith("city:")}
        
        # Additional validation: ensure city_ids are in active leaf nodes (cities)
        validated_city_ids = city_ids.copy()
        if active_leaf_nodes is not None:
            # active_nodes should be leaf nodes (cities) from the graph
            active_leaf_ids = {node.id for node in active_leaf_nodes}
            validated_city_ids = city_ids & active_leaf_ids
            
            # Update rationale if some cities were filtered out
            invalid_cities = city_ids - active_leaf_ids
            if invalid_cities:
                rationale = f"Filtered out inactive cities {invalid_cities}: {rationale}"

        # CRITICAL: Remove target node from pruned_ids to prevent accidental pruning
        if target_node_id and target_node_id in validated_city_ids:
            validated_city_ids.remove(target_node_id)

        # If LLM returned non-city IDs, raise an error
        if candidate_ids and not city_ids:
            raise ValueError("LLM returned non-city IDs")
        
        if len(candidate_ids) > len(city_ids):
            filtered_out = candidate_ids - city_ids
            raise ValueError(f"Filtered out non-city nodes {filtered_out}: {rationale}")

        self.pruning_count += len(validated_city_ids)
        return PruningResult(pruned_ids=validated_city_ids, rationale=rationale)
        