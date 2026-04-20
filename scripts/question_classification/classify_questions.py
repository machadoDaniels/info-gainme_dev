#!/usr/bin/env python3
"""Classify seeker questions from Info Gainme benchmark runs.

For each seeker turn, the classifier labels:

    question_type_rationale    one sentence explaining the primary class
    question_type              semantic | lexical | direct_guess | malformed | other
    subclasses_rationale       one sentence explaining the sub-class tag(s) (may be empty)
    subclasses                 list of snake_case tags (may be empty — tags are orthogonal
                               and a single question can carry more than one, e.g.
                               ["comparative", "quantitative_threshold"])
    redundancy                 none | exact_duplicate | semantic_equivalent | strictly_implied
    redundant_with_turn        1-indexed turn this one duplicates, or null

Rationale fields always come BEFORE the label they justify so the model
reasons chain-of-thought style inside its structured output.

This script issues ONE LLM request per conversation: it feeds the whole
Q&A history of a game and gets back an array of classifications, one per
turn. That's roughly 10× fewer requests than classifying turn-by-turn,
and redundancy detection stays confident because the model already sees
every prior turn in context.

Usage
-----
    # Full sweep (default)
    python3 scripts/question_classification/classify_questions.py \
        --max-concurrency 32

    # Cap per stratum
    python3 scripts/question_classification/classify_questions.py --per-stratum 30

    # Single conversation (debug)
    python3 scripts/question_classification/classify_questions.py \
        --seeker-file outputs/models/.../conversations/.../seeker.json

Output
------
    outputs/question_classification/
        <experiment>/<target>/classification.json   per-conversation
        summary.json                                 aggregated counts
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class QuestionType(str, Enum):
    SEMANTIC = "semantic"
    LEXICAL = "lexical"
    DIRECT_GUESS = "direct_guess"
    MALFORMED = "malformed"
    OTHER = "other"


class RedundancyType(str, Enum):
    NONE = "none"
    EXACT_DUPLICATE = "exact_duplicate"
    SEMANTIC_EQUIVALENT = "semantic_equivalent"
    STRICTLY_IMPLIED = "strictly_implied"


class QuestionClassification(BaseModel):
    """One classification per seeker turn.

    Field order is deliberate: the echoed ``question`` anchors the entry to a
    specific input turn (so we can detect the model dropping / reordering /
    inventing turns), then every rationale precedes its label so the model
    generates the justification first and commits to a class after.
    """

    question: str = Field(
        description=(
            "The seeker's question for this turn, echoed VERBATIM from the input. "
            "Used as an integrity anchor to verify the classification lines up with "
            "the correct turn — do not paraphrase, summarise, or translate."
        ),
    )
    question_type_rationale: str = Field(
        default="",
        description="One short sentence on why this question_type fits.",
    )
    question_type: QuestionType
    subclasses_rationale: str = Field(
        default="",
        description="One short sentence on why these sub-class tags fit. Empty when subclasses is [].",
    )
    subclasses: list[str] = Field(
        default_factory=list,
        description=(
            "Orthogonal snake_case tags supplementing question_type. A single question "
            "may carry more than one (e.g. ['comparative', 'quantitative_threshold']). "
            "Empty list when no sharper label applies."
        ),
    )
    redundancy: RedundancyType
    redundant_with_turn: int | None = Field(
        default=None,
        description="1-indexed earlier turn this one is redundant with; null otherwise.",
    )


class BatchClassification(BaseModel):
    """Wrapper for one LLM response: one classification per turn, in order."""

    classifications: list[QuestionClassification]


# ---------------------------------------------------------------------------
# Seeker.json parsing
# ---------------------------------------------------------------------------


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
# Oracle responses include a "[Computer] - Active candidates..." block after
# the first line in FO mode; capture only the Oracle's one-line answer.
_ORACLE_RE = re.compile(r"\[Oracle\]\s*-\s*([^\n]+)")


@dataclass
class TurnQA:
    turn: int
    question: str
    oracle_answer: str


def extract_turns(seeker_json: dict[str, Any]) -> list[TurnQA]:
    """Pair each seeker question with the Oracle's one-line reply."""
    turns: list[TurnQA] = []
    pending_q: str | None = None
    idx = 0
    for msg in seeker_json.get("history", []):
        role = msg.get("role")
        content = msg.get("content", "") or ""
        if role == "assistant":
            pending_q = _THINK_RE.sub("", content).strip()
        elif role == "user" and pending_q is not None:
            m = _ORACLE_RE.search(content)
            answer = m.group(1).strip() if m else content.strip()
            idx += 1
            turns.append(TurnQA(idx, pending_q, answer))
            pending_q = None
    return turns


# ---------------------------------------------------------------------------
# Experiment discovery + stratified sampling
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Stratum:
    model_slug: str
    experiment: str
    domain: str       # geo / objects / diseases
    mode: str         # fo / po
    cot: bool


def _parse_experiment(model_slug: str, experiment: str) -> Stratum | None:
    exp = experiment.lower()
    for prefix in ("geo", "objects", "diseases"):
        if exp.startswith(prefix):
            domain = prefix
            break
    else:
        return None
    mode = "po" if "_po" in exp else "fo"
    # "_no_cot" contains "_cot" — exclude it explicitly.
    has_no_cot = exp.endswith("_no_cot") or "_no_cot_" in exp
    has_cot = exp.endswith("_cot") or "_cot_" in exp
    return Stratum(model_slug, experiment, domain, mode, has_cot and not has_no_cot)


def discover_conversations(outputs_root: Path) -> dict[Stratum, list[Path]]:
    buckets: dict[Stratum, list[Path]] = defaultdict(list)
    models_root = outputs_root / "models"
    if not models_root.exists():
        return buckets
    for model_dir in sorted(p for p in models_root.iterdir() if p.is_dir()):
        for exp_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            stratum = _parse_experiment(model_dir.name, exp_dir.name)
            if stratum is None:
                continue
            buckets[stratum].extend(exp_dir.glob("conversations/*/seeker.json"))
    return buckets


def stratified_sample(
    buckets: dict[Stratum, list[Path]],
    per_stratum: int | None,
    rng: random.Random,
) -> list[tuple[Stratum, Path]]:
    """per_stratum=None → take everything; otherwise cap each stratum."""
    picks: list[tuple[Stratum, Path]] = []
    for stratum, paths in buckets.items():
        if not paths:
            continue
        chosen = list(paths) if per_stratum is None else rng.sample(paths, min(per_stratum, len(paths)))
        picks.extend((stratum, p) for p in chosen)
    return picks


# ---------------------------------------------------------------------------
# LLM classification (one request per whole conversation)
# ---------------------------------------------------------------------------


SYSTEM_PROMPT = """You are a taxonomist labelling yes/no questions asked by a seeker agent playing an information-gain guessing game.

The seeker tries to identify a secret target (a city, an object, or a disease) by asking yes/no questions. You will receive the FULL Q&A history of one game and you must classify EVERY turn, in order.

For each turn, produce SEVEN fields IN THIS EXACT ORDER:

1. **question** — the seeker's question for this turn, copied VERBATIM from the input (no paraphrasing, no trimming, no translation). This anchors the entry to the correct turn.
2. **question_type_rationale** — one short sentence explaining why your question_type fits.
3. **question_type** — one of:
   - `semantic`     uses domain knowledge (location, symptoms, attributes, taxonomy). E.g. "Is it in Asia?", "Does it affect the brain?".
   - `lexical`      asks about the NAME/STRING only, no domain meaning. E.g. "Does the name start with A?".
   - `direct_guess` names a specific candidate. E.g. "Is it Mumbai?", "Is it dengue?".
   - `malformed`    not a valid yes/no question: bare statement, open-ended, echoing the Oracle/Computer, etc.
   - `other`        genuinely does not fit any of the above (rare).
4. **subclasses_rationale** — one short sentence justifying the chosen sub-class tag(s). Use `""` when `subclasses` is an empty list.
5. **subclasses** — a LIST of snake_case sub-tags, orthogonal to question_type. The list may be empty (`[]`) when nothing sharper applies, or contain one OR MORE tags when multiple apply (e.g. a question that is both a comparison AND a numeric threshold: `["comparative", "quantitative_threshold"]`). Do not duplicate tags. Common tags (you may coin new ones):
   - `hierarchical_category`    broad taxonomic level ("Is it an animal?")
   - `fine_grained_category`    narrow taxonomic level ("Is it a citrus fruit?")
   - `comparative`              any comparison to another entity — by size, age, frequency, etc. ("bigger than a microwave", "older than Rome")
   - `relational`               defined relative to another entity ("near India?", "borders France?")
   - `quantitative_threshold`   numeric threshold ("population > 1M?", "more than 4 legs?")
   - `compound_predicate`       conjoined conditions ("in Asia AND coastal?")
   - `meta_strategy`            about the game state, not the target
   - `statement`, `open_ended`  sub-types of `malformed`
6. **redundancy** — one of:
   - `none`                brings new information.
   - `exact_duplicate`     literally the same wording as an earlier turn.
   - `semantic_equivalent` same question in different words (answer is determined).
   - `strictly_implied`    answer is already determined by prior Q&A.
7. **redundant_with_turn** — 1-indexed earliest prior turn that makes this one redundant; `null` when redundancy is `none`. Redundancy for turn K is measured ONLY against turns 1..K-1 — never use information from later turns.

Return a JSON object with a single key `classifications` containing an array with EXACTLY ONE entry per turn, in the same order as the turns shown. No prose, no markdown fences.

### Worked example (3-turn game, diseases)

    Turn 1: Q "Is the target disease primarily affecting the respiratory system?" / A "Yes"
    Turn 2: Q "Is it bigger than the common cold in mortality?" / A "Yes"
    Turn 3: Q "Is it the common cold?" / A "No"

Expected output:

    {
      "classifications": [
        {
          "question": "Is the target disease primarily affecting the respiratory system?",
          "question_type_rationale": "Uses medical-domain knowledge to categorise the target by affected organ system.",
          "question_type": "semantic",
          "subclasses_rationale": "Asks about a broad organ-system category, a high-level taxonomic bucket.",
          "subclasses": ["hierarchical_category"],
          "redundancy": "none",
          "redundant_with_turn": null
        },
        {
          "question": "Is it bigger than the common cold in mortality?",
          "question_type_rationale": "Probes a domain attribute (mortality) by comparison against a reference disease.",
          "question_type": "semantic",
          "subclasses_rationale": "Frames the attribute as a comparison with another entity, and the attribute itself is a numeric-like magnitude.",
          "subclasses": ["comparative", "quantitative_threshold"],
          "redundancy": "none",
          "redundant_with_turn": null
        },
        {
          "question": "Is it the common cold?",
          "question_type_rationale": "Names a specific candidate rather than probing an attribute.",
          "question_type": "direct_guess",
          "subclasses_rationale": "",
          "subclasses": [],
          "redundancy": "strictly_implied",
          "redundant_with_turn": 2
        }
      ]
    }

The `question` field MUST match the input turn's Q text character-for-character. If it does not, the classification will be flagged as misaligned and rejected.
"""


def build_user_message(turns: list[TurnQA], domain: str) -> str:
    lines = [f"Domain: {domain}", "", f"Game turns ({len(turns)} total — classify every one, in order):"]
    for t in turns:
        lines.append(f"  Turn {t.turn}: {t.question}")
    lines.append("")
    lines.append(
        f"Return a JSON object with key `classifications` — an array of EXACTLY {len(turns)} entries in turn order."
    )
    return "\n".join(lines)


_WS_RE = re.compile(r"\s+")


def _norm(s: str) -> str:
    """Loose comparison for the question-echo integrity check: strips
    surrounding whitespace and collapses internal whitespace runs, so a
    stray extra space or newline does not trip the mismatch check."""
    return _WS_RE.sub(" ", (s or "")).strip().lower()


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def _extract_json_payload(raw: str) -> str:
    """Strip <think> blocks + markdown fences, isolate the outer JSON object."""
    cleaned = _THINK_RE.sub("", raw).strip()
    fence = _CODE_FENCE_RE.search(cleaned)
    if fence:
        cleaned = fence.group(1).strip()
    if cleaned.startswith("{"):
        return cleaned
    match = _JSON_OBJECT_RE.search(cleaned)
    return match.group(0) if match else cleaned


async def classify_conversation_batch(
    client: AsyncOpenAI,
    model: str,
    turns: list[TurnQA],
    domain: str,
    thinking: bool,
    semaphore: asyncio.Semaphore,
) -> list[QuestionClassification]:
    """Single LLM request: classify every turn in the conversation at once."""
    extra_body: dict[str, Any] = {"guided_json": BatchClassification.model_json_schema()}
    if thinking:
        extra_body["chat_template_kwargs"] = {"thinking": True}

    async with semaphore:
        completion = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_message(turns, domain)},
            ],
            response_format={"type": "json_object"},
            extra_body=extra_body,
        )

    msg = completion.choices[0].message
    raw = msg.content or getattr(msg, "reasoning_content", "") or ""
    payload = _extract_json_payload(raw)
    try:
        batch = BatchClassification.model_validate_json(payload)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"{type(e).__name__}: {e} | raw={raw[:400]!r}") from e

    got, want = len(batch.classifications), len(turns)
    if got != want:
        raise RuntimeError(f"model returned {got} classifications, expected {want}")

    return batch.classifications


async def classify_conversation(
    seeker_path: Path,
    stratum: Stratum,
    client: AsyncOpenAI,
    model: str,
    thinking: bool,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    turns = extract_turns(json.loads(seeker_path.read_text()))

    try:
        classifications = await classify_conversation_batch(
            client, model, turns, stratum.domain, thinking, semaphore
        )
        turn_payloads = []
        for t, cls in zip(turns, classifications):
            payload: dict[str, Any] = {
                "turn": t.turn,
                "question": t.question,
                "oracle_answer": t.oracle_answer,
                "classification": cls.model_dump(mode="json"),
            }
            if _norm(cls.question) != _norm(t.question):
                payload["question_echo_warning"] = (
                    f"echoed={cls.question!r} differs from input={t.question!r}"
                )
            turn_payloads.append(payload)
    except Exception as e:  # noqa: BLE001 — bubble up as per-conversation error
        turn_payloads = [
            {
                "turn": t.turn,
                "question": t.question,
                "oracle_answer": t.oracle_answer,
                "classification": {"error": f"{type(e).__name__}: {e}"},
            }
            for t in turns
        ]

    return {
        "seeker_path": str(seeker_path),
        "experiment": stratum.experiment,
        "model_slug": stratum.model_slug,
        "domain": stratum.domain,
        "mode": stratum.mode,
        "cot": stratum.cot,
        "target": seeker_path.parent.name,
        "num_turns": len(turns),
        "turns": turn_payloads,
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def summarise(conv_results: list[dict[str, Any]]) -> dict[str, Any]:
    type_counts: Counter[str] = Counter()
    redundancy_counts: Counter[str] = Counter()
    subclass_counts: Counter[str] = Counter()
    subclass_by_type: dict[str, Counter[str]] = defaultdict(Counter)
    per_stratum: dict[str, Counter[str]] = defaultdict(Counter)
    errors = 0
    total_turns = 0

    for conv in conv_results:
        stratum_key = f"{conv['domain']}/{conv['mode']}/{'cot' if conv['cot'] else 'no_cot'}"
        for turn in conv["turns"]:
            total_turns += 1
            cls = turn["classification"]
            if "error" in cls:
                errors += 1
                continue
            qt = cls["question_type"]
            type_counts[qt] += 1
            per_stratum[stratum_key][qt] += 1
            redundancy_counts[cls["redundancy"]] += 1
            for tag in cls.get("subclasses") or []:
                label = str(tag).strip().lower()
                if not label:
                    continue
                subclass_counts[label] += 1
                subclass_by_type[qt][label] += 1

    return {
        "total_conversations": len(conv_results),
        "total_turns": total_turns,
        "errors": errors,
        "question_type_counts": dict(type_counts),
        "redundancy_counts": dict(redundancy_counts),
        "per_stratum_question_type_counts": {k: dict(v) for k, v in per_stratum.items()},
        "subclass_counts": dict(subclass_counts.most_common()),
        "subclass_by_question_type": {k: dict(v) for k, v in subclass_by_type.items()},
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _out_path(out_dir: Path, stratum: Stratum, seeker: Path) -> Path:
    return out_dir / stratum.experiment / seeker.parent.name / "classification.json"


def _stratum_from_seeker_file(path: Path) -> Stratum:
    parts = path.resolve().parts
    try:
        i = parts.index("models")
        return _parse_experiment(parts[i + 1], parts[i + 2]) or Stratum(
            parts[i + 1], parts[i + 2], "unknown", "fo", False
        )
    except (ValueError, IndexError):
        return Stratum("unknown", "unknown", "unknown", "fo", False)


async def _amain(args: argparse.Namespace) -> int:
    rng = random.Random(args.seed)

    if args.seeker_file is not None:
        picks = [(_stratum_from_seeker_file(args.seeker_file), args.seeker_file)]
    else:
        buckets = discover_conversations(args.outputs_root)
        print(f"Discovered {sum(len(v) for v in buckets.values())} conversations across {len(buckets)} strata.")
        for st, paths in sorted(buckets.items(), key=lambda kv: (kv[0].domain, kv[0].mode, kv[0].cot)):
            cot = "cot" if st.cot else "no_cot"
            print(f"  {st.domain}/{st.mode}/{cot} [{st.model_slug}/{st.experiment}]: {len(paths)}")
        picks = stratified_sample(buckets, args.per_stratum, rng)
        cap = "all" if args.per_stratum is None else args.per_stratum
        print(f"Sampled {len(picks)} conversations (per_stratum={cap}, seed={args.seed}).")

    if args.dry_run:
        for st, path in picks:
            print(f"  would classify: {st.domain}/{st.mode}/{'cot' if st.cot else 'no_cot'} :: {path}")
        return 0
    if not picks:
        print("No conversations to classify.", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    existing: list[dict[str, Any]] = []
    todo: list[tuple[Stratum, Path]] = []
    for st, path in picks:
        p = _out_path(args.out_dir, st, path)
        if not args.force and p.exists():
            try:
                existing.append(json.loads(p.read_text()))
                continue
            except Exception as e:  # noqa: BLE001
                print(f"  warning: could not reload {p} ({e}); will re-classify", file=sys.stderr)
        todo.append((st, path))
    print(f"Resume: {len(existing)} already classified, {len(todo)} to do.")

    if not todo:
        results: list[dict[str, Any]] = []
    else:
        client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key)
        thinking = not args.no_thinking
        semaphore = asyncio.Semaphore(args.max_concurrency)
        total = len(todo)
        done = 0
        lock = asyncio.Lock()

        async def _run(stratum: Stratum, path: Path) -> dict[str, Any]:
            nonlocal done
            res = await classify_conversation(path, stratum, client, args.model, thinking, semaphore)
            _out_path(args.out_dir, stratum, path).parent.mkdir(parents=True, exist_ok=True)
            _out_path(args.out_dir, stratum, path).write_text(
                json.dumps(res, indent=2, ensure_ascii=False)
            )
            async with lock:
                done += 1
                errs = sum(1 for t in res["turns"] if "error" in t["classification"])
                tag = f"({errs} turn errors)" if errs else "ok"
                print(f"[{done}/{total}] {stratum.experiment}/{path.parent.name} — {tag}")
            return res

        try:
            results = list(await asyncio.gather(*(_run(st, p) for st, p in todo)))
        finally:
            await client.close()

    conv_results = existing + results
    summary = summarise(conv_results)
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print("\n=== Summary ===")
    print(f"  conversations: {summary['total_conversations']}")
    print(f"  turns:         {summary['total_turns']} (errors: {summary['errors']})")
    print(f"  question_type: {summary['question_type_counts']}")
    print(f"  redundancy:    {summary['redundancy_counts']}")
    if summary["subclass_counts"]:
        print("  subclass tags (top):")
        for name, count in list(summary["subclass_counts"].items())[:20]:
            print(f"    {count:>4}  {name}")
    print(f"\nWrote: {summary_path}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default="http://200.137.197.131:60002/v1")
    p.add_argument("--api-key", default="NINGUEM-TA-PURO-2K26")
    p.add_argument("--model", default="nvidia/Kimi-K2.5-NVFP4")
    p.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    p.add_argument("--out-dir", type=Path, default=Path("outputs/question_classification"))
    p.add_argument("--seeker-file", type=Path, default=None, help="Classify a single seeker.json (overrides sampling).")
    p.add_argument(
        "--per-stratum",
        type=int,
        default=None,
        help="Conversations per stratum. Default: None = full sweep; pass an int to cap.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-concurrency", type=int, default=16, help="Max in-flight LLM requests.")
    p.add_argument("--no-thinking", action="store_true", help="Disable reasoning mode on the classifier.")
    p.add_argument("--dry-run", action="store_true", help="List what would be classified and exit.")
    p.add_argument("--force", action="store_true", help="Re-classify even when classification.json exists.")
    args = p.parse_args()
    return asyncio.run(_amain(args))


if __name__ == "__main__":
    sys.exit(main())
