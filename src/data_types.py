"""Shared types (skeleton).

Provides `TurnState`, `PruningResult`, and enums per UML.
"""

from __future__ import annotations

from ast import List
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Optional, Any, List
from pydantic import BaseModel
import json


class ObservabilityMode(Enum):
    FULLY_OBSERVABLE = "FO"
    PARTIALLY_OBSERVABLE = "PO"

@dataclass
class Question:
    text: str


class OracleResponse(BaseModel):
    rationale: str
    answer: str
    game_over: bool

@dataclass
class Answer:
    rationale: str
    text: str
    compliant: bool
    game_over: bool = False


class PrunerResponse(BaseModel):
    pruned_ids: List[str]
    rationale: str

@dataclass
class PruningResult:
    pruned_ids: set[str]
    rationale: str


@dataclass
class TurnState:
    turn_index: int
    h_before: float
    h_after: float
    info_gain: float
    pruned_count: int
    question: Question
    answer: Answer
    
    # Additional metadata for conversation export
    pruning_result: Optional[PruningResult] = None
    active_nodes_before: Optional[int] = None
    active_nodes_after: Optional[int] = None
    active_leaf_nodes_before: Optional[int] = None
    active_leaf_nodes_after: Optional[int] = None
    timestamp_start: Optional[str] = None
    timestamp_end: Optional[str] = None
    duration_seconds: Optional[float] = None
    graph_snapshot: Optional[str] = None
    
    def to_export_dict(self) -> dict[str, Any]:
        """Convert TurnState to dictionary for JSONL export.
        
        Uses dataclasses.asdict() to preserve all attributes, then converts
        sets to lists for JSON serialization.
        
        Returns:
            Dictionary with all turn data ready for JSON export.
        """
        data = asdict(self)
        
        # Convert pruning_result.pruned_ids from set to list for JSON serialization
        if data.get("pruning_result") and "pruned_ids" in data["pruning_result"]:
            data["pruning_result"]["pruned_ids"] = list(data["pruning_result"]["pruned_ids"])
        
        return data


