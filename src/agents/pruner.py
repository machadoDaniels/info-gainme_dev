"""PrunerAgent implementation for deterministic rule-based pruning.

The PrunerAgent analyzes question-answer pairs and determines which nodes
should be pruned from the knowledge graph based on logical rules.
"""

from __future__ import annotations

from typing import Set

from ..data_types import Answer, PruningResult, Question
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
            llm_adapter: LLM adapter for potential AI-assisted pruning decisions.
        """
        self.llm_adapter = llm_adapter
        self.pruning_count = 0
    
    def analyze_and_prune(
        self,
        graph_text: str,
        turn_index: int,
        question: Question,
        answer: Answer,
    ) -> PruningResult:
        """Delegate pruning decision to the LLM using graph text and turn context.

        The LLM must respond with a strict JSON object:
            {"pruned_ids": ["node:id", ...], "rationale": "..."}

        Args:
            graph_text: Textual representation of the active graph (graph_to_text).
            turn_index: Current turn number (1-based).
            question: Seeker's question.
            answer: Oracle's answer.

        Returns:
            PruningResult with pruned node IDs and rationale. Falls back to no
            pruning if parsing fails or the model returns an invalid response.
        """
        system_prompt = get_pruner_system_prompt()

        user_prompt = (
            "GRAPH:\n" + graph_text + "\n\n" +
            f"TURN: {turn_index}\n" +
            f"QUESTION: {question.text}\n" +
            f"ANSWER: {answer.text}\n\n" +
            "Respond with JSON only: {\"pruned_ids\": [...], \"rationale\": \"...\"}"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        reply = self.llm_adapter.generate(messages=messages, add_to_history=False, temperature=0.0)

        parsed = parse_first_json_object(reply)
        if parsed is None:
            return PruningResult(pruned_ids=set(), rationale="Invalid LLM response (non-JSON)")

        pruned_ids_list = parsed.get("pruned_ids", [])
        rationale = parsed.get("rationale", "") or "No rationale provided"

        # Normalize and validate shape
        if not isinstance(pruned_ids_list, list):
            pruned_ids_list = []
        pruned_ids: Set[str] = {str(x) for x in pruned_ids_list if isinstance(x, (str, int))}

        self.pruning_count += len(pruned_ids)
        return PruningResult(pruned_ids=pruned_ids, rationale=rationale)
        