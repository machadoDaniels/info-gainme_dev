"""SeekerAgent implementation for proactive information gathering.

The SeekerAgent asks strategic questions to reduce uncertainty in the knowledge
graph, operating under different observability modes (FULLY_OBSERVABLE, 
PARTIALLY_OBSERVABLE, AUTO).
"""

from __future__ import annotations

from typing import Set, Optional
from os import getenv

from ..data_types import ObservabilityMode, Question, Answer
from ..graph import Node
from ..prompts import get_seeker_system_prompt
from .llm_adapter import LLMAdapter
from dotenv import load_dotenv

load_dotenv()

class SeekerAgent:
    """Agent that seeks information by asking strategic questions.
    
    This agent generates strategic yes/no questions to reduce uncertainty in the
    knowledge graph, operating under different observability modes.
    """

    def __init__(
        self, 
        llm_adapter: LLMAdapter, 
        observability_mode: ObservabilityMode
    ) -> None:
        """Initialize the SeekerAgent.
        
        Args:
            model: Model identifier for the LLM.
            llm: LLMAdapter instance for generating questions.
            observability_mode: Mode controlling how much graph info is visible.
            
        Raises:
            ValueError: If model is empty or llm is None.
        """
        if llm_adapter is None:
            raise ValueError("LLMAdapter cannot be None")
        if not isinstance(observability_mode, ObservabilityMode):
            raise ValueError("Invalid observability mode")
            
        self._model = llm_adapter.config.model
        self._llm_adapter = llm_adapter
        self._observability_mode = observability_mode
        self._questions_asked = 0
        self._initial_graph_injected: bool = False
        
        # Initialize conversation with system prompt
        self._llm_adapter.append_history("system", get_seeker_system_prompt())

    @property
    def model(self) -> str:
        """Get the model identifier."""
        return self._model

    @property
    def observability_mode(self) -> ObservabilityMode:
        """Get the current observability mode."""
        return self._observability_mode
        
    @property
    def questions_asked(self) -> int:
        """Get the number of questions asked by this agent."""
        return self._questions_asked

    def choose_observability(self) -> ObservabilityMode:
        """Return the current observability mode.
        
        Returns:
            The configured observability mode (FULLY_OBSERVABLE or PARTIALLY_OBSERVABLE).
        """
        return self.observability_mode

    def question_to_oracle(
        self,
        active_nodes: Set[Node],
        turn: int,
        ) -> Question:
        """Generate a strategic question to ask the Oracle.
        
        Args:
            active_nodes: Set of nodes that haven't been pruned yet.
            
        Returns:
            A Question object containing the generated question text.
            
        Raises:
            ValueError: If active_nodes is empty.
        """
        # Generate question
        question_text = self._llm_adapter.generate()
        
        # Note: LLMAdapter.generate() automatically adds the response to history,
        # so we don't need to manually append it here
        
        # Track usage
        self._questions_asked += 1
        
        return Question(text=question_text.strip())

    def add_oracle_answer_and_pruning(
        self, 
        answer: Answer,
        graph_text: Optional[str],
        turn: int,
        ) -> None:
        """Add Oracle's answer to the conversation history and pruning result.

        Args:
            answer: The Oracle's answer to add to history.
            active_nodes: Current set of active nodes.
            turn: Current turn number.

        Returns:
            None
        """
        user_answer = f"""[Oracle] - {answer.text}"""

        context = self._build_context(graph_text, turn)

        if context:
            user_answer += f"\n[Computer] - {context}"
        
        # Add Oracle's answer as user message (Seeker's perspective: Oracle is user)
        self._llm_adapter.append_history("user", user_answer)
    
    def add_initial_graph(self, graph_text: str, turn: int) -> None:
        """Inject the initial graph into the system prompt once when fully observed.

        Args:
            graph_text: Textual representation of the knowledge graph.

        Returns:
            None
        """
        if self._observability_mode != ObservabilityMode.FULLY_OBSERVABLE:
            return
        if not graph_text or self._initial_graph_injected:
            return

        context = self._build_context(graph_text, turn)
        self._llm_adapter.append_history("user", f"[Computer] - {context}")
        self._initial_graph_injected = True

    def _build_context(self, graph_text: Optional[str], turn: int) -> Optional[str]:
        """Build context prompt based on observability mode and current state.
        
        Args:
            active_nodes: Current set of active nodes.
            turn: Current turn number.
            
        Returns:
            Formatted context string for the LLM.
        """
        if self.observability_mode == ObservabilityMode.FULLY_OBSERVABLE:
            assert graph_text is not None, "graph_text cannot be None when observability mode is FULLY_OBSERVABLE"
            # Provide full graph textual view instead of listing nodes
            return graph_text

        elif self.observability_mode == ObservabilityMode.PARTIALLY_OBSERVABLE:
            # Show only basic stats
            return None

        else:
            raise ValueError(f"Unknown observability mode: {self.observability_mode}")




if __name__ == "__main__":
    """Interactive test case - simulate a conversation between Seeker and user-as-Oracle."""
    
    print("🎮 Geographic Guessing Game - Interactive Test")
    print("=" * 50)
    print("You are the Oracle! A SeekerAgent will ask you yes/no questions.")
    print("Answer 'yes', 'no', or 'quit' to exit.\n")
    
    # Create a simple LLMAdapter config for testing
    from .llm_adapter import LLMAdapter
    from .llm_config import LLMConfig
    
    # Use a minimal config (will fail LLM calls, but we can test the conversation flow)
    config = LLMConfig(
        model="gpt-4o-mini",
        api_key=getenv("OPENAI_API_KEY")
    )
    
    llm_adapter = LLMAdapter(config)
    seeker = SeekerAgent(llm_adapter, ObservabilityMode.PARTIALLY_OBSERVABLE)
    
    # Simulate some active nodes (without actual graph)
    fake_nodes = {
        Node(id="paris", label="Paris"),
        Node(id="london", label="London"), 
        Node(id="berlin", label="Berlin"),
        Node(id="rome", label="Rome")
    }
    
    turn = 1
    max_turns = 10
    
    print(f"📍 Scenario: Seeker is trying to find the target among: {', '.join(n.id for n in fake_nodes)}")
    print("Think of one as your secret target!\n")
    
    while turn <= max_turns:
        # This will fail because we don't have a real LLM, but we can show the pattern
        print(f"🤖 Turn {turn}: Seeker would ask a question here...")
        print("    (In real usage, the LLM would generate a strategic question)")
        
        # Simulate manual question for demo
        question = seeker.question_to_oracle(active_nodes=fake_nodes, turn=turn)
        
        print(f"Question: '{question.text}'")
        
        # Get user's answer as Oracle
        try:
            oracle_answer = input(f"\n🔮 You (Oracle): ").strip().lower()
        except EOFError:
            print("🚪 Input ended, exiting test.")
            break
        
        if oracle_answer == "quit":
            print("🚪 Game ended by user.")
            break
            
        if oracle_answer not in ["yes", "no"]:
            print("⚠️  Please answer 'yes' or 'no' (or 'quit' to exit)")
            continue
        
        # Create Answer object and add to Seeker's history
        answer = Answer(text=oracle_answer.capitalize(), compliant=True)
        seeker.add_oracle_answer_and_pruning(answer, fake_nodes, turn)
        
        print(f"✅ Turn {turn} completed. Oracle answered: {oracle_answer}")
        
        # Check if game won only on final targeting question
        if oracle_answer == "yes" and ("is this the target" in question.text.lower() or "is the target paris" in question.text.lower()):
            print("🎉 Game won! Target found!")
            break
                
        turn += 1
                
    print(f"\n📊 Game Summary:")
    print(f"   - Turns played: {turn-1}")
    print(f"   - Questions asked: {seeker.questions_asked}")
    print(f"   - Observability: {seeker.observability_mode.name}")
    print(f"   - Model: {seeker.model}")
    
    # Show conversation history
    print(f"\n💬 Conversation History:")
    for i, msg in enumerate(seeker._llm_adapter.history):
        role_emoji = {"system": "⚙️", "user": "🔮", "assistant": "🤖"}
        emoji = role_emoji.get(msg["role"], "❓")
        content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
        print(f"   {emoji} {msg['role']}: {content}")
        
    print("\n🎯 Test completed!")