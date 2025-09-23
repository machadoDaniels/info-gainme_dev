"""Shared types (skeleton).

Provides `TurnState`, `PruningResult`, and enums per UML.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ObservabilityMode(Enum):
    FULLY_OBSERVED = "FO"
    PARTIALLY_OBSERVED = "PO"


@dataclass
class Question:
    text: str


@dataclass
class Answer:
    text: str
    compliant: bool
    game_over: bool = False


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


