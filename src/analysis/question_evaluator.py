"""Question evaluator for analyzing Seeker's question choices.

This module evaluates whether the Seeker made optimal choices by comparing
the information gain of the chosen question against all considered questions.

IMPORTANT: This module is READ-ONLY. It does NOT:
- Save or modify any plots
- Save or modify any turn files (turns.jsonl, seeker.json, etc.)
- Export conversations
- Save graph snapshots

The only output is the evaluation results dictionary returned by evaluate_seeker_choices(),
which should be saved separately by the caller.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from ..candidates import Candidate, CandidatePool
from ..entropy import Entropy
from ..data_types import Question, Answer, PruningResult
from ..agents.oracle import OracleAgent
from ..agents.pruner import PrunerAgent
from ..agents.llm_adapter import LLMAdapter
from ..agents.llm_config import LLMConfig
from ..domain.types import GEO_DOMAIN, DISEASES_DOMAIN, OBJECTS_DOMAIN, DomainConfig
from ..domain.geo.loader import load_geo_candidates
from ..domain.diseases.loader import load_flat_disease_candidates
from ..domain.objects.loader import load_flat_object_candidates
from ..utils import ClaryLogger

logger = ClaryLogger.get_logger(__name__)


# ---------------------------------------------------------------------------
# Domain detection & dataset loading
# ---------------------------------------------------------------------------

def _detect_domain(target_id: str) -> str:
    """Detect domain from target_id prefix (disease:, object:, city:)."""
    if target_id.startswith("disease:"):
        return "diseases"
    if target_id.startswith("object:"):
        return "objects"
    return "geo"


def _find_dataset_csv(domain: str, project_root: Path) -> Path:
    """Return the default dataset CSV for the given domain."""
    candidates = {
        "diseases": [
            project_root / "data" / "diseases" / "diseases_160.csv",
            project_root / "data" / "diseases" / "diseases_full.csv",
        ],
        "objects": [
            project_root / "data" / "objects" / "objects_full.csv",
        ],
        "geo": [
            project_root / "data" / "geo" / "top_160_pop_cities.csv",
            project_root / "data" / "geo" / "top_20_pop_cities.csv",
            project_root / "data" / "top_40_pop_cities.csv",
        ],
    }
    for path in candidates.get(domain, []):
        if path.exists():
            return path
    raise FileNotFoundError(
        f"No dataset CSV found for domain '{domain}'. "
        "Pass --dataset-csv explicitly."
    )


def _load_pool(domain: str, csv_path: Path) -> tuple[CandidatePool, DomainConfig]:
    if domain == "diseases":
        return load_flat_disease_candidates(csv_path)
    if domain == "objects":
        return load_flat_object_candidates(csv_path)
    pool, domain_config = load_geo_candidates(csv_path)
    return pool, domain_config


# ---------------------------------------------------------------------------
# State reconstruction
# ---------------------------------------------------------------------------

def load_turns_history(turns_jsonl_path: Path) -> List[Dict[str, Any]]:
    turns = []
    with turns_jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                turns.append(json.loads(line))
    return turns


_JSONL_INDEX: Optional[Dict[str, List[Dict[str, Any]]]] = None
_JSONL_INDEX_PATH: Optional[Path] = None


def _get_jsonl_index(unified_jsonl: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Load and cache the unified seeker_traces.jsonl as a {seeker_path → turns} dict."""
    global _JSONL_INDEX, _JSONL_INDEX_PATH
    if _JSONL_INDEX is not None and _JSONL_INDEX_PATH == unified_jsonl:
        return _JSONL_INDEX
    index: Dict[str, List[Dict[str, Any]]] = {}
    for line in unified_jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except Exception:
            continue
        key = record.get("seeker_path", "")
        if key:
            index[key] = record.get("turns") or record.get("history") or []
    _JSONL_INDEX = index
    _JSONL_INDEX_PATH = unified_jsonl
    logger.info("Loaded unified JSONL index: %d entries from %s", len(index), unified_jsonl)
    return index


def _load_seeker_history(conversation_dir: Path) -> List[Dict[str, Any]]:
    """Load synthesized reasoning turns for a conversation.

    Tries, in order:
    1. Per-conversation ``seeker_traces.json`` (old pipeline, key ``"history"``)
    2. Unified ``outputs/seeker_traces.jsonl`` (new pipeline, key ``"turns"``)
       matched by the conversation's seeker.json path.

    Returns an empty list if neither source is found.
    """
    # --- option 1: per-conversation file (old format) ---
    per_conv = conversation_dir / "seeker_traces.json"
    if per_conv.exists():
        with per_conv.open(encoding="utf-8") as f:
            data = json.load(f)
        return data.get("history") or data.get("turns") or []

    # --- option 2: unified JSONL (new format) with cached index ---
    project_root = Path(__file__).parent.parent.parent
    unified_jsonl = project_root / "outputs" / "seeker_traces.jsonl"
    if not unified_jsonl.exists():
        logger.warning("No seeker_traces.json and no unified seeker_traces.jsonl for %s", conversation_dir)
        return []

    index = _get_jsonl_index(unified_jsonl)
    target_key = str(conversation_dir / "seeker.json")
    turns = index.get(target_key)
    if turns is None:
        logger.warning("Conversation %s not found in %s", conversation_dir.name, unified_jsonl)
        return []
    return turns


def reconstruct_pool_state(
    pool: CandidatePool,
    turns_history: List[Dict[str, Any]],
    up_to_turn: int,
) -> CandidatePool:
    """Return a deep-copied pool with prunings from turns < up_to_turn applied."""
    pool_copy = copy.deepcopy(pool)
    for turn in turns_history:
        if turn["turn_index"] < up_to_turn:
            pruning_result = turn.get("pruning_result", {})
            pruned_labels = set(pruning_result.get("pruned_labels", []))
            if pruned_labels:
                pool_copy.prune(pruned_labels)
    return pool_copy


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------

def simulate_oracle_answer(
    question_text: str,
    target: Candidate,
    oracle_config: LLMConfig,
    domain_config: DomainConfig,
) -> Answer:
    oracle_adapter = LLMAdapter(oracle_config, save_history=True)
    oracle = OracleAgent(
        llm_adapter=oracle_adapter,
        target=target,
        domain_config=domain_config,
    )
    oracle.add_seeker_question(Question(text=question_text))
    return oracle.answer_seeker()


def simulate_pruning(
    pool: CandidatePool,
    question_text: str,
    answer: Answer,
    turn_index: int,
    target_label: str,
    pruner_config: LLMConfig,
    domain_config: DomainConfig,
) -> PruningResult:
    pruner_adapter = LLMAdapter(pruner_config, save_history=False)
    pruner = PrunerAgent(pruner_adapter, domain_config=domain_config)
    return pruner.analyze_and_prune(
        candidate_pool=pool,
        turn_index=turn_index,
        question=Question(text=question_text),
        answer=answer,
        target_label=target_label,
    )


# ---------------------------------------------------------------------------
# Per-question / per-turn evaluation
# ---------------------------------------------------------------------------

def evaluate_question(
    question_text: str,
    pool: CandidatePool,
    turn_index: int,
    target: Candidate,
    oracle_config: LLMConfig,
    pruner_config: LLMConfig,
    domain_config: DomainConfig,
) -> Dict[str, Any]:
    active_before = pool.get_active()
    h_before = Entropy.compute(len(active_before))

    answer = simulate_oracle_answer(question_text, target, oracle_config, domain_config)

    pool_copy = copy.deepcopy(pool)
    pruning_result = simulate_pruning(
        pool_copy, question_text, answer, turn_index,
        target.label, pruner_config, domain_config,
    )
    if pruning_result.pruned_labels:
        pool_copy.prune(pruning_result.pruned_labels)

    active_after = pool_copy.get_active()
    h_after = Entropy.compute(len(active_after))
    info_gain = Entropy.info_gain(h_before, h_after)

    return {
        "question": question_text,
        "simulated_answer": answer.text,
        "simulated_pruned_labels": list(pruning_result.pruned_labels),
        "pruned_count": len(pruning_result.pruned_labels),
        "h_before": h_before,
        "h_after": h_after,
        "info_gain": info_gain,
        "active_candidates_before": len(active_before),
        "active_candidates_after": len(active_after),
    }


def evaluate_turn(
    turn_data: Dict[str, Any],
    pool: CandidatePool,
    turns_history: List[Dict[str, Any]],
    target: Candidate,
    oracle_config: LLMConfig,
    pruner_config: LLMConfig,
    domain_config: DomainConfig,
    questions_considered: List[str],
) -> Dict[str, Any]:
    turn_index = turn_data["turn_index"]
    # Per synthesis prompt: the chosen question is the last item in questions_considered
    chosen_question = questions_considered[-1] if questions_considered else turn_data.get("question", "")

    logger.info("  Reconstructing pool state for turn %d...", turn_index)
    pool_state = reconstruct_pool_state(pool, turns_history, turn_index)
    active_count = len(pool_state.get_active())
    logger.info("  Active candidates at turn %d: %d", turn_index, active_count)

    logger.info("  Evaluating %d considered questions...", len(questions_considered))
    questions_evaluation = []
    for question_text in tqdm(questions_considered, desc=f"    Turn {turn_index}", leave=False):
        try:
            eval_result = evaluate_question(
                question_text, pool_state, turn_index,
                target,
                oracle_config, pruner_config, domain_config,
            )
            questions_evaluation.append(eval_result)
        except Exception as e:
            questions_evaluation.append({
                "question": question_text,
                "error": str(e),
                "info_gain": -1.0,
            })

    questions_evaluation.sort(key=lambda x: x.get("info_gain", -1.0), reverse=True)

    chosen_info_gain = None
    chosen_rank = None
    for i, ev in enumerate(questions_evaluation):
        if ev["question"] == chosen_question:
            chosen_info_gain = ev["info_gain"]
            chosen_rank = i + 1
            break

    optimal_question = questions_evaluation[0]["question"] if questions_evaluation else None
    optimal_info_gain = questions_evaluation[0].get("info_gain", 0.0) if questions_evaluation else 0.0

    return {
        "turn_index": turn_index,
        "chosen_question": chosen_question,
        "chosen_info_gain": chosen_info_gain,
        "chosen_rank": chosen_rank,
        "optimal_question": optimal_question,
        "optimal_info_gain": optimal_info_gain,
        "was_optimal": chosen_rank == 1 if chosen_rank else False,
        "questions_evaluation": questions_evaluation,
        "total_considered": len(questions_considered),
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def evaluate_seeker_choices(
    conversation_dir: Path,
    oracle_config: LLMConfig,
    pruner_config: LLMConfig,
    dataset_csv_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Evaluate Seeker's question choices for a complete game.

    READ-ONLY: never writes files (except stateless LLM API calls).

    Args:
        conversation_dir: Directory containing seeker_traces.json, turns.jsonl, metadata.json.
        oracle_config: LLM configuration for Oracle simulation.
        pruner_config: LLM configuration for Pruner simulation.
        dataset_csv_path: Path to domain CSV. Auto-detected if None.
    """
    # Load metadata
    logger.info("Loading metadata...")
    metadata_path = conversation_dir / "metadata.json"
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    target_info = metadata["target"]
    target_id = target_info["id"]
    logger.info("Target: %s (%s)", target_info["label"], target_id)

    # Detect domain and load candidates
    domain = _detect_domain(target_id)
    project_root = Path(__file__).parent.parent.parent
    if dataset_csv_path is None:
        dataset_csv_path = _find_dataset_csv(domain, project_root)
    logger.info("Domain: %s — loading candidates from %s", domain, dataset_csv_path)
    pool, domain_config = _load_pool(domain, dataset_csv_path)
    logger.info("Pool loaded: %d candidates", len(pool.get_active()))

    # Find target Candidate in pool
    target_label = target_info["label"]
    target_candidate = next(
        (c for c in pool.candidates if c.id == target_id or c.label == target_label),
        None,
    )
    if target_candidate is None:
        # Fallback: build a Candidate from metadata
        target_candidate = Candidate(
            id=target_id,
            label=target_label,
            attrs=target_info.get("attrs", {}),
        )
        logger.warning("Target '%s' not found in pool — using metadata attrs", target_label)

    # Load seeker traces — prefer per-conversation seeker_traces.json,
    # fall back to the unified seeker_traces.jsonl index (new pipeline).
    logger.info("Loading seeker traces...")
    history = _load_seeker_history(conversation_dir)
    logger.info("Found %d turns in seeker traces", len(history))

    # Load turns history
    logger.info("Loading turns history...")
    turns_jsonl_path = conversation_dir / "turns.jsonl"
    turns_history = load_turns_history(turns_jsonl_path)
    logger.info("Loaded %d turns from turns.jsonl", len(turns_history))

    # Filter turns that have questions_considered
    turns_with_questions = [
        td for td in history
        if td.get("reasoning_trace", {}).get("questions_considered")
    ]
    logger.info("Found %d turns with questions_considered to evaluate", len(turns_with_questions))

    turns_evaluation = []
    for td in tqdm(turns_with_questions, desc="Evaluating turns", unit="turn"):
        turn_index = td.get("turn_index", 0)
        questions_considered = td["reasoning_trace"]["questions_considered"]
        logger.info(
            "\nEvaluating turn %d (%d questions)...", turn_index, len(questions_considered)
        )
        try:
            turn_eval = evaluate_turn(
                td, pool, turns_history,
                target_candidate,
                oracle_config, pruner_config, domain_config,
                questions_considered,
            )
            was_optimal = turn_eval.get("was_optimal", False)
            chosen_ig = turn_eval.get("chosen_info_gain")
            optimal_ig = turn_eval.get("optimal_info_gain", 0.0)
            rank = turn_eval.get("chosen_rank")
            rank_str = str(rank) if rank is not None else "?"
            ig_str = f"{chosen_ig:.3f}" if chosen_ig is not None else "?"
            logger.info(
                "  Turn %d: %s (rank %s/%d, IG: %s, optimal: %.3f)",
                turn_index,
                "✅ Optimal" if was_optimal else "❌ Suboptimal",
                rank_str, len(questions_considered), ig_str, optimal_ig,
            )
            turns_evaluation.append(turn_eval)
        except Exception as e:
            logger.error("  Turn %d failed: %s", turn_index, e, exc_info=True)
            turns_evaluation.append({"turn_index": turn_index, "error": str(e)})

    # Summary statistics
    optimal_choices = sum(1 for t in turns_evaluation if t.get("was_optimal", False))
    total_evaluated = sum(1 for t in turns_evaluation if "error" not in t)

    valid_chosen_ig = [
        t["chosen_info_gain"] for t in turns_evaluation
        if "error" not in t
        and t.get("chosen_info_gain") is not None
        and t["chosen_info_gain"] >= 0.0
    ]
    avg_chosen_ig = sum(valid_chosen_ig) / len(valid_chosen_ig) if valid_chosen_ig else 0.0

    valid_optimal_ig = [
        t.get("optimal_info_gain", 0.0) for t in turns_evaluation if "error" not in t
    ]
    avg_optimal_ig = sum(valid_optimal_ig) / len(valid_optimal_ig) if valid_optimal_ig else 0.0

    valid_considered = [
        t["total_considered"] for t in turns_evaluation
        if "error" not in t and t.get("total_considered") is not None
    ]
    avg_questions = sum(valid_considered) / len(valid_considered) if valid_considered else 0.0

    connection_errors = sum(
        1
        for t in turns_evaluation
        if "error" not in t
        for q in t.get("questions_evaluation", [])
        if "Connection" in q.get("error", "")
    )

    return {
        "conversation_dir": str(conversation_dir),
        "target": {"id": target_id, "label": target_label},
        "domain": domain,
        "turns_evaluation": turns_evaluation,
        "summary": {
            "total_turns_evaluated": total_evaluated,
            "optimal_choices": optimal_choices,
            "optimal_choice_rate": optimal_choices / total_evaluated if total_evaluated > 0 else 0.0,
            "avg_chosen_info_gain": avg_chosen_ig,
            "avg_optimal_info_gain": avg_optimal_ig,
            "avg_questions_considered_per_turn": avg_questions,
            "total_connection_errors": connection_errors,
        },
    }
