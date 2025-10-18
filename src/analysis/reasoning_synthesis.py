"""Reasoning synthesis utilities for SeekerAgent traces."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..agents.llm_adapter import LLMAdapter, LLMConfig
from ..prompts import get_reasoning_synthesis_prompt
from ..utils import parse_first_json_object, ClaryLogger

logger = ClaryLogger.get_logger(__name__)

# Pre-compiled regex patterns for better performance
_THINK_PATTERN = re.compile(r'<think>\n(.*?)\n</think>', re.DOTALL)
_ORACLE_PATTERN = re.compile(r'\[Oracle\]\s*-\s*(.*?)(?:\n|$)')


def load_seeker_conversation(file_path: Path) -> Dict[str, Any]:
    """Load seeker conversation data from JSON file.
    
    Args:
        file_path: Path to the seeker.json file.
        
    Returns:
        Dictionary containing the conversation data.
        
    Raises:
        FileNotFoundError: If the file doesn't exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Seeker conversation file not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_reasoning_from_message(message: Dict[str, str]) -> Optional[str]:
    """Extract reasoning content from a message with <think> tags.
    
    Args:
        message: Message dictionary with 'role' and 'content' keys.
        
    Returns:
        Extracted reasoning content or None if no <think> tags found.
    """
    if message.get("role") != "assistant":
        return None
    
    content = message.get("content", "")
    
    # Look for <think> tags and extract content
    think_match = _THINK_PATTERN.search(content)
    if think_match:
        return think_match.group(1).strip()
    
    return None


def synthesize_reasoning_trace(
    reasoning_text: str,
    llm_adapter: LLMAdapter
) -> Dict[str, Any]:
    """Synthesize a single reasoning trace using LLM.
    
    Args:
        reasoning_text: The raw reasoning text to synthesize.
        llm_adapter: LLM adapter to use for synthesis.
        
    Returns:
        Dictionary containing synthesized reasoning trace.
        
    Raises:
        ValueError: If synthesis fails.
    """
    # Prepare the synthesis prompt
    synthesis_prompt = get_reasoning_synthesis_prompt()
    
    user_content = f"<think>\n{reasoning_text}\n</think>"

    # Create messages for LLM call
    messages = [
        {"role": "system", "content": synthesis_prompt},
        {"role": "user", "content": user_content}
    ]
    
    # Generate synthesis using stateless call
    try:
        response = llm_adapter.generate(
            messages=messages,
            stateless=True,
            add_to_history=False
        )
        
        # Parse JSON response
        parsed = parse_first_json_object(response)
        if not parsed:
            raise ValueError(f"Invalid JSON response: {response}")
        
        # Validate required fields
        required_fields = ["summary", "options_considered", "decision_rationale"]
        for field in required_fields:
            if field not in parsed:
                parsed[field] = f"Missing field: {field}"
        
        return parsed
        
    except Exception as e:
        # Fallback: create basic structure if synthesis fails
        return {
            "summary": "Synthesis failed",
            "options_considered": [],
            "decision_rationale": f"Error: {str(e)}"
        }


def extract_oracle_answer(message_content: str) -> str:
    """Extract oracle answer from message content.
    
    Args:
        message_content: The content of the user message.
        
    Returns:
        The oracle answer text.
    """
    # Look for [Oracle] - pattern
    oracle_match = _ORACLE_PATTERN.search(message_content)
    if oracle_match:
        return oracle_match.group(1).strip()
    
    return message_content.strip()


def create_turn_based_traces(
    seeker_data: Dict[str, Any],
    llm_adapter: LLMAdapter
) -> List[Dict[str, Any]]:
    """Create turn-based traces similar to seeker.json structure.
    
    Args:
        seeker_data: The loaded seeker conversation data.
        llm_adapter: LLM adapter for synthesis.
        
    Returns:
        List of turn-based traces.
    """
    reasoning_history = seeker_data.get("reasoning_history", [])
    history = seeker_data.get("history", [])
    
    # Pre-filter messages for better performance
    reasoning_msgs = [
        msg for msg in reasoning_history 
        if msg.get("role") == "assistant" and extract_reasoning_from_message(msg)
    ]
    
    assistant_msgs = [
        (i, msg) for i, msg in enumerate(history)
        if msg.get("role") == "assistant" and not msg.get("content", "").startswith("# SeekerAgent")
    ]
    
    # Log processing info
    logger.info("Processing %d reasoning traces for synthesis", len(reasoning_msgs))
    
    turns = []
    
    # Match each assistant question with its reasoning and oracle answer
    for i, (hist_idx, assistant_msg) in enumerate(assistant_msgs):
        question = assistant_msg.get("content", "").strip()
        
        # Find oracle answer more efficiently
        oracle_answer = _find_oracle_answer(history, hist_idx)
        
        # Get corresponding reasoning
        reasoning_text = None
        if i < len(reasoning_msgs):
            reasoning_text = extract_reasoning_from_message(reasoning_msgs[i])
        
        if reasoning_text:
            try:
                reasoning_trace = synthesize_reasoning_trace(reasoning_text, llm_adapter)
                turns.append({
                    'turn_index': i + 1,
                    "original_reasoning": reasoning_text,
                    "reasoning_trace": reasoning_trace,
                    "question": question,
                    "oracle_answer": oracle_answer
                })
            except Exception as e:
                logger.warning("Failed to synthesize reasoning for question '%s': %s", 
                              question[:50] + "..." if len(question) > 50 else question, e)
    
    return turns


def _find_oracle_answer(history: List[Dict[str, Any]], start_idx: int) -> str:
    """Find oracle answer starting from the given index.
    
    Args:
        history: List of conversation messages.
        start_idx: Starting index to search from.
        
    Returns:
        Oracle answer text or "Unknown" if not found.
    """
    for i in range(start_idx + 1, len(history)):
        msg = history[i]
        if msg.get("role") == "user" and "[Oracle]" in msg.get("content", ""):
            return extract_oracle_answer(msg.get("content", ""))
    return "Unknown"


def create_seeker_traces_file(
    input_path: Path,
    output_path: Path,
    llm_config: LLMConfig
) -> None:
    """Create seeker_traces.json from seeker.json file.
    
    Args:
        input_path: Path to input seeker.json file.
        output_path: Path where seeker_traces.json will be saved.
        llm_config: LLM configuration for synthesis.
        
    Raises:
        FileNotFoundError: If input file doesn't exist.
        ValueError: If processing fails.
    """
    # Load seeker conversation data
    seeker_data = load_seeker_conversation(input_path)
    
    # Create LLM adapter for synthesis
    llm_adapter = LLMAdapter(llm_config, save_history=False)
    
    # Create turn-based traces
    turns = create_turn_based_traces(seeker_data, llm_adapter)
    
    # Create output data structure similar to seeker.json
    output_data = {
        "agent_type": seeker_data.get("agent_type", "seeker"),
        "config": seeker_data.get("config", {}),
        "observability_mode": seeker_data.get("observability_mode"),
        "total_messages": seeker_data.get("total_messages"),
        "total_turns": len(turns),
        "history": turns
    }
    
    # Save to output file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
