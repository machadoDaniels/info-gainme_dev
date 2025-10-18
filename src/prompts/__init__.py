"""Prompt management utilities.

This module provides utilities to load and manage system prompts from markdown files.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict


_PROMPT_CACHE: Dict[str, str] = {}


def load_prompt(prompt_name: str) -> str:
    """Load a system prompt from a markdown file.
    
    Args:
        prompt_name: Name of the prompt file (without .md extension).
        
    Returns:
        The prompt content as a string.
        
    Raises:
        FileNotFoundError: If the prompt file doesn't exist.
        
    Note:
        Prompts are cached after first load for performance.
    """
    if prompt_name in _PROMPT_CACHE:
        return _PROMPT_CACHE[prompt_name]
    
    # Get the directory where this module is located
    prompts_dir = Path(__file__).parent
    prompt_file = prompts_dir / f"{prompt_name}.md"
    
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    
    with open(prompt_file, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    
    _PROMPT_CACHE[prompt_name] = content
    return content


def get_seeker_system_prompt() -> str:
    """Get the SeekerAgent system prompt."""
    return load_prompt("seeker_system")


def get_oracle_system_prompt() -> str:
    """Get the OracleAgent system prompt."""
    return load_prompt("oracle_system")


def get_pruner_system_prompt() -> str:
    """Get the PrunerAgent system prompt."""
    return load_prompt("pruner_system")


def get_reasoning_synthesis_prompt() -> str:
    """Get the reasoning synthesis system prompt."""
    return load_prompt("reasoning_synthesis")


def clear_cache() -> None:
    """Clear the prompt cache. Useful for testing or reloading prompts."""
    _PROMPT_CACHE.clear()


__all__ = [
    "load_prompt",
    "get_seeker_system_prompt", 
    "get_oracle_system_prompt",
    "get_pruner_system_prompt",
    "get_reasoning_synthesis_prompt",
    "clear_cache",
]


