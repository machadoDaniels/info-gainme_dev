"""OracleAgent implementation for answering questions about the target.

The OracleAgent knows the target node and answers questions truthfully
without revealing the target directly. It maintains compliance with
the game rules while providing helpful information.
"""

from __future__ import annotations

from typing import Set

from ..data_types import Answer, Question, OracleResponse
from ..graph import Node
from ..prompts import get_oracle_system_prompt
from .llm_adapter import LLMAdapter
from ..utils.utils import parse_first_json_object


class OracleAgent:
    """Agent that answers questions about a known target node.
    
    The Oracle knows the target node and answers questions truthfully
    without revealing the target directly. It maintains compliance with
    the game rules while providing helpful information.
    """

    def __init__(
        self, 
        llm_adapter: LLMAdapter, 
        target_node_id: str,
        *,
        target_node: Node = None
        ) -> None:
        """Initialize the OracleAgent.
        
        Args:
            model: Model identifier for the LLM.
            llm_adapter: LLMAdapter instance for generating answers.
            target_node_id: ID of the node the Oracle "knows" as the target.
            target_node: Optional target node object with full details.
            
        Raises:
            ValueError: If any parameter is invalid.
        """
        if llm_adapter is None:
            raise ValueError("LLMAdapter cannot be None")
        if not target_node_id:
            raise ValueError("Target node ID cannot be empty")
            
        self._model = llm_adapter.config.model
        self._llm_adapter = llm_adapter
        self._target_node_id = target_node_id
        self._target_node = target_node  # Store target node for metadata export
        self._answers_given = 0
        
        # Build system prompt with target information
        system_prompt = self._build_system_prompt_with_target(target_node)
        self._llm_adapter.append_history("system", system_prompt)

    @property
    def model(self) -> str:
        """Get the model identifier."""
        return self._model

    @property
    def target_node_id(self) -> str:
        """Get the target node ID."""
        return self._target_node_id
        
    @property
    def answers_given(self) -> int:
        """Get the number of answers given by this agent."""
        return self._answers_given

    def add_seeker_question(self, question: Question) -> None:
        """Add Seeker's question to conversation history.
        
        Args:
            question: The Seeker's question to add to history.
        """
        # Add only the Seeker's question as user message
        user_message = f"[Seeker] - {question.text}"
        self._llm_adapter.append_history("user", user_message)

    def answer_seeker(self) -> Answer:
        """Generate an answer to the Seeker's question.
        
        Returns:
            Answer object with response text, compliance flag, and game_over flag.
            
        Note:
            The Oracle must answer truthfully and return JSON with rationale, answer, and game_over status.
        """
        # Generate answer (expecting JSON response with rationale first)
        response = self._llm_adapter.generate()

        oracle_response = OracleResponse.model_validate_json(response)
        
        # Check compliance
        is_compliant = self._check_compliance(oracle_response.answer)
        
        # Track usage
        self._answers_given += 1    
        
        return Answer(text=oracle_response.answer, compliant=is_compliant, game_over=oracle_response.game_over, rationale=oracle_response.rationale)

    def _build_system_prompt_with_target(self, target_node: Node) -> str:
        """Build system prompt with target information included.
        
        Args:
            target_node: The target node object, if available.
            
        Returns:
            Complete system prompt with target details.
        """
        base_prompt = get_oracle_system_prompt()
        
        # Add target information to system prompt
        attrs_str = ""
        if target_node.attrs:
            attrs_str = f", {', '.join(f'{k}={v}' for k, v in target_node.attrs.items())}"
        
        target_info = f"\n\n## Your Target\n\nID: {target_node.id}\nLabel: {target_node.label}{attrs_str}\n\nThis is the target you know about. Answer all questions truthfully based on this target's properties."
        
        return base_prompt + target_info

    def _check_compliance(self, answer_text: str) -> bool:
        """Check if the answer complies with Oracle rules.
        
        Args:
            question_text: The original question.
            answer_text: The generated answer.
            
        Returns:
            True if the answer appears compliant, False otherwise.
            
        Note:
            This is a basic heuristic check. A more sophisticated version
            could use additional LLM calls or rule-based validation.
        """
        answer_lower = answer_text.lower().strip()
        
        # Check for direct target revelation (basic heuristic)
        if self._target_node_id.lower() in answer_lower:
            return False
            
        # Simple compliance indicators
        compliant_patterns = [
            "yes", "no"
        ]
        
        for pattern in compliant_patterns:
            if pattern in answer_lower:
                return True
                
        # Default to compliant if no clear violations detected
        return True


if __name__ == "__main__":
    """Minimal one-shot test for OracleAgent."""

    from os import getenv
    from dotenv import load_dotenv
    from .llm_adapter import LLMAdapter
    from .llm_config import LLMConfig

    load_dotenv()

    config = LLMConfig(model="gpt-4o-mini", api_key=getenv("OPENAI_API_KEY"))
    llm_adapter = LLMAdapter(config)

    target_node = Node(
        id="paris",
        label="Paris",
        attrs={
            "continent": "europe",
            "country": "france",
            "capital": "true",
            "population": "2161000",
            "coastal": "false",
        },
    )

    oracle = OracleAgent(
        llm_adapter=llm_adapter,
        target_node_id="paris",
        target_node=target_node,
    )

    print("🎮 Oracle minimal test (one question)")
    try:
        user_question = input("You (Seeker): ").strip()
    except EOFError:
        print("🚪 No input.")
        raise SystemExit(0)

    if not user_question:
        print("⚠️  Empty question.")
        raise SystemExit(0)

    question = Question(text=user_question)
    oracle.add_seeker_question(question)

    try:
        answer = oracle.answer_seeker()
        print(f"🔮 Oracle: {answer.text}")
        print(f"🧩 Game over: {answer.game_over}")
    except Exception as e:
        print(f"⚠️  Oracle response failed: {e}")