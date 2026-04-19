#!/usr/bin/env python3
"""Question-type classifier for Info Gainme seeker questions.

Classifies each seeker question into one of:

    semantic      — domain-knowledge categorisation
    lexical       — property of the name/string (no domain meaning)
    direct_guess  — names a specific candidate
    malformed     — not a valid yes/no question at all
    other         — escape hatch with a required rationale + proposed_class

Plus redundancy relative to prior Q&A (``none`` / ``exact_duplicate`` /
``semantic_equivalent`` / ``strictly_implied``) with a ``redundant_with_turn``
pointer.

The script walks ``outputs/models/**/conversations/**/seeker.json``, buckets
each conversation into a stratum ``(model, experiment, domain, mode, cot)``,
optionally samples ``--per-stratum`` conversations from each, and calls an
OpenAI-compatible vLLM endpoint (Kimi-K2.5 by default) with optional reasoning
mode. Runs are resumable — existing ``classification.json`` files are reused
unless ``--force`` is passed.

Usage
-----
    # Sample 30 conversations per stratum (recommended for headline numbers)
    python3 scripts/classify_questions.py \
        --per-stratum 30 \
        --max-concurrency 32

    # Full sweep (all conversations, capped per stratum)
    python3 scripts/classify_questions.py --per-stratum 99999

    # Classify a single seeker.json (debug)
    python3 scripts/classify_questions.py \
        --seeker-file outputs/models/.../conversations/.../seeker.json

Output
------
    outputs/question_classification/
        <experiment>/<target>/classification.json   # per-conversation
        summary.json                                # aggregated counts + OTHER proposals
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
from pydantic import AliasChoices, BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class QuestionType(str, Enum):
    SEMANTIC = "semantic"          # domain-knowledge categorisation
    LEXICAL = "lexical"            # property of the name/string (no domain semantics)
    DIRECT_GUESS = "direct_guess"  # names a specific candidate
    MALFORMED = "malformed"        # not a valid yes/no question at all
    OTHER = "other"                # genuinely doesn't fit any of the above


class RedundancyType(str, Enum):
    NONE = "none"
    EXACT_DUPLICATE = "exact_duplicate"        # same question asked literally before
    SEMANTIC_EQUIVALENT = "semantic_equivalent"  # asks the same thing in different words
    STRICTLY_IMPLIED = "strictly_implied"      # answer is determined by prior Q&A


class SubClass(BaseModel):
    """Optional supplementary tag — a more specific sub-class worth surfacing.

    Orthogonal to ``question_type``: a ``semantic`` question can also carry a
    ``subclass`` like ``comparative_size`` or ``relational``. Use whenever a
    more specific label is relevant, regardless of the primary type.
    """

    # Rationale is nice-to-have but models sometimes omit it; don't reject.
    rationale: str = Field(
        default="",
        description="Why this specific sub-class is the right tag for this question.",
    )
    # Models use various field names for the label — accept the common ones.
    proposed_class: str = Field(
        ...,
        validation_alias=AliasChoices(
            "proposed_class", "class_name", "class", "label", "name", "tag"
        ),
        serialization_alias="proposed_class",
        description=(
            "Short snake_case name for the sub-class "
            "(e.g. 'comparative_size', 'relational', 'quantitative_threshold')."
        ),
    )


class QuestionClassification(BaseModel):
    """Structured output — one per question."""

    question_type: QuestionType
    subclass: SubClass | None = Field(
        default=None,
        description=(
            "OPTIONAL supplementary tag. Fill whenever a more specific sub-class is "
            "relevant, independent of question_type. Leave null if no useful sub-class applies."
        ),
    )
    redundancy: RedundancyType
    redundant_with_turn: int | None = Field(
        default=None,
        description=(
            "1-indexed turn number of the earlier question this one is redundant with. "
            "Null when redundancy == 'none'."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_subclass(cls, v: Any) -> Any:
        """Accept a bare string for ``subclass`` (shorthand) and convert to object."""
        if isinstance(v, dict) and isinstance(v.get("subclass"), str):
            label = v["subclass"].strip()
            v["subclass"] = {"proposed_class": label, "rationale": ""} if label else None
        return v


# ---------------------------------------------------------------------------
# Seeker.json parsing
# ---------------------------------------------------------------------------


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
# Match only the Oracle's one-line answer — FO mode appends a "[Computer] - Active candidates..."
# block on subsequent lines that we do NOT want in the classifier prompt.
_ORACLE_RE = re.compile(r"\[Oracle\]\s*-\s*([^\n]+)")


@dataclass
class TurnQA:
    turn: int
    question: str
    oracle_answer: str  # "Yes" / "No" / "Invalid" etc.


def strip_think(text: str) -> str:
    """Remove <think>...</think> blocks from a seeker message."""
    return _THINK_RE.sub("", text).strip()


def extract_turns(seeker_json: dict[str, Any]) -> list[TurnQA]:
    """Pair each seeker question with the Oracle's response."""
    history = seeker_json.get("history", [])
    turns: list[TurnQA] = []
    pending_q: str | None = None
    turn_idx = 0

    for msg in history:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "assistant":
            pending_q = strip_think(content)
        elif role == "user" and pending_q is not None:
            m = _ORACLE_RE.search(content)
            answer = m.group(1).strip() if m else content.strip()
            turn_idx += 1
            turns.append(TurnQA(turn=turn_idx, question=pending_q, oracle_answer=answer))
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
    if exp.startswith("geo"):
        domain = "geo"
    elif exp.startswith("objects"):
        domain = "objects"
    elif exp.startswith("diseases"):
        domain = "diseases"
    else:
        return None
    mode = "po" if "_po" in exp else "fo"  # default fo
    # "_no_cot" also contains "_cot", so exclude it explicitly.
    has_no_cot = exp.endswith("_no_cot") or "_no_cot_" in exp
    has_cot = exp.endswith("_cot") or "_cot_" in exp
    cot = has_cot and not has_no_cot
    return Stratum(model_slug=model_slug, experiment=experiment, domain=domain, mode=mode, cot=cot)


def discover_conversations(outputs_root: Path) -> dict[Stratum, list[Path]]:
    """Walk outputs/models/**/conversations/** and bucket by stratum."""
    buckets: dict[Stratum, list[Path]] = defaultdict(list)
    models_root = outputs_root / "models"
    if not models_root.exists():
        return buckets

    for model_dir in sorted(models_root.iterdir()):
        if not model_dir.is_dir():
            continue
        for exp_dir in sorted(model_dir.iterdir()):
            if not exp_dir.is_dir():
                continue
            stratum = _parse_experiment(model_dir.name, exp_dir.name)
            if stratum is None:
                continue
            for seeker in exp_dir.glob("conversations/*/seeker.json"):
                buckets[stratum].append(seeker)
    return buckets


def stratified_sample(
    buckets: dict[Stratum, list[Path]],
    per_stratum: int,
    rng: random.Random,
) -> list[tuple[Stratum, Path]]:
    picks: list[tuple[Stratum, Path]] = []
    for stratum, paths in buckets.items():
        if not paths:
            continue
        chosen = rng.sample(paths, min(per_stratum, len(paths)))
        picks.extend((stratum, p) for p in chosen)
    return picks


# ---------------------------------------------------------------------------
# LLM classification
# ---------------------------------------------------------------------------


SYSTEM_PROMPT = """You are a taxonomist labelling yes/no questions asked by a seeker agent playing an information-gain guessing game.

The seeker tries to identify a secret target (a city, a physical object, or a disease) by asking yes/no questions. You will be given one question at a time, together with the questions and Oracle answers that came before it in the same game.

For each question, decide THREE things.

### 1. question_type (the primary class — pick exactly one)

- **semantic**     — uses domain knowledge to categorise the target. Attributes, properties, function, location, symptoms, taxonomy. Examples: "Is it in Asia?", "Is it edible?", "Does it affect the brain?".
- **lexical**      — asks about a property of the NAME/STRING only, with no domain meaning. Examples: "Does it start with A?", "Is the name 5 letters long?".
- **direct_guess** — names a specific candidate. Examples: "Is it Mumbai?", "Is it a guitar?", "Is it dengue?".
- **malformed**    — not a valid yes/no question at all. Examples: "Yes.", "intertrigo", "[Oracle] - No", "What is the disease?" (open-ended), a statement instead of a question. Pick this when the seeker violated the game's Q&A format.
- **other**        — truly does not fit any of the above. Rare.

### 2. subclass (OPTIONAL — supplementary tag, orthogonal to question_type)

The `subclass` field is **not** an escape hatch. It is an additional, more specific label you may attach to ANY question (including `semantic`, `direct_guess`, etc.) whenever a more precise sub-category is relevant and worth surfacing.

Fill the `subclass` field whenever a sharper label applies. Good candidates include (non-exhaustive):

- **comparative_size**       — "Is it larger than a microwave?", "Is it bigger than Europe?"
- **comparative_other**      — non-size comparisons ("older than...", "more common than...")
- **quantitative_threshold** — "Population > 1M?", "More than 4 legs?", "Mortality > 10%?"
- **relational**             — relative to another entity ("near India?", "shares a border with France?")
- **hierarchical_category**  — asks about a broad taxonomic level ("Is it an animal?", "Is it a continent-level region?")
- **fine_grained_category**  — asks about a narrow taxonomic level ("Is it a citrus fruit?", "Is it a striker position?")
- **compound_predicate**     — multiple conjoined conditions ("in Asia AND coastal?")
- **meta_strategy**          — about the game state itself, not the target
- **open_ended**              — malformed because it's open-ended (sub-type of `malformed`)
- **statement**              — malformed because it's a statement (sub-type of `malformed`)

You may also coin a NEW snake_case label if the question clearly warrants one not in the list. Keep it short and descriptive.

When `subclass` is filled, include a short `rationale` explaining why that tag fits. Leave `subclass` null if no sharper label is useful.

### 3. redundancy (relative to the prior Q&A in this game)

- **none**                — brings new information.
- **exact_duplicate**     — literally the same wording as an earlier question.
- **semantic_equivalent** — asks the same thing in different words (answer would always match).
- **strictly_implied**    — the answer is already determined by prior Q&A.

If redundancy is not `none`, set `redundant_with_turn` to the 1-indexed turn of the earliest earlier question that makes this one redundant. Otherwise leave it null. When in doubt, pick `none`.

Return ONLY the structured JSON object. No prose."""


def build_user_message(prior: list[TurnQA], current: TurnQA, domain: str) -> str:
    lines = [f"Domain: {domain}", "", "Prior Q&A in this game:"]
    if not prior:
        lines.append("  (none — this is the first question)")
    else:
        for t in prior:
            lines.append(f"  Turn {t.turn}: Q: {t.question}")
            lines.append(f"           A: {t.oracle_answer}")
    lines.append("")
    lines.append(f"Current question to classify (Turn {current.turn}):")
    lines.append(f"  {current.question}")
    return "\n".join(lines)


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
# Markdown code fences: ```json\n...\n``` or ```\n...\n```
_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def _extract_json_payload(raw: str) -> str:
    """Strip ``<think>...</think>`` blocks, code fences, and isolate the JSON object.

    When a reasoning model emits its chain-of-thought inside ``content`` instead
    of ``reasoning_content`` (i.e. the vLLM server was not started with a
    reasoning parser), ``response_format`` parsing fails because the string
    starts with ``<think>`` rather than ``{``. Some models additionally wrap
    the JSON in a Markdown code fence (```` ```json ... ``` ````). Both are
    handled here.
    """
    cleaned = _THINK_RE.sub("", raw).strip()
    # If wrapped in a fenced code block, extract the fence body.
    fence = _CODE_FENCE_RE.search(cleaned)
    if fence:
        cleaned = fence.group(1).strip()
    if cleaned.startswith("{"):
        return cleaned
    match = _JSON_OBJECT_RE.search(cleaned)
    if match:
        return match.group(0)
    return cleaned  # let json.loads raise with the original-ish string


async def classify_turn(
    client: AsyncOpenAI,
    model: str,
    prior: list[TurnQA],
    current: TurnQA,
    domain: str,
    thinking: bool,
    semaphore: asyncio.Semaphore,
) -> QuestionClassification:
    """Single classification call with structured output + optional reasoning.

    Gated by ``semaphore`` so the total number of in-flight requests to the
    vLLM server never exceeds the user-configured concurrency cap.

    Uses ``chat.completions.create`` (not ``.parse``) so we can salvage
    responses where the thinking block leaks into ``content``.
    """
    extra_body: dict[str, Any] = {
        # Ask vLLM to guide decoding to our schema — defence in depth against
        # models that ignore the ``response_format`` hint.
        "guided_json": QuestionClassification.model_json_schema(),
    }
    if thinking:
        # vLLM/SGLang switch for Kimi-K2.5 and similar thinking-capable models.
        extra_body["chat_template_kwargs"] = {"thinking": True}

    async with semaphore:
        completion = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_message(prior, current, domain)},
            ],
            response_format={"type": "json_object"},
            extra_body=extra_body,
        )

    msg = completion.choices[0].message
    raw = msg.content or ""
    if not raw.strip():
        # vLLM with a reasoning parser puts everything in reasoning_content
        # when the model produces no post-think output. Fall back to that.
        raw = getattr(msg, "reasoning_content", "") or ""
    payload = _extract_json_payload(raw)
    try:
        return QuestionClassification.model_validate_json(payload)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"{type(e).__name__}: {e} | raw={raw[:300]!r}") from e


# ---------------------------------------------------------------------------
# Per-conversation pipeline
# ---------------------------------------------------------------------------


async def classify_conversation(
    seeker_path: Path,
    stratum: Stratum,
    client: AsyncOpenAI,
    model: str,
    thinking: bool,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    seeker_json = json.loads(seeker_path.read_text())
    turns = extract_turns(seeker_json)

    async def _one(idx: int) -> dict[str, Any]:
        prior = turns[:idx]
        current = turns[idx]
        try:
            cls = await classify_turn(
                client, model, prior, current, stratum.domain, thinking, semaphore
            )
            payload: dict[str, Any] = cls.model_dump(mode="json")
        except Exception as e:  # noqa: BLE001 — surface all errors as payloads
            payload = {"error": f"{type(e).__name__}: {e}"}
        return {
            "turn": current.turn,
            "question": current.question,
            "oracle_answer": current.oracle_answer,
            "classification": payload,
        }

    results = await asyncio.gather(*(_one(i) for i in range(len(turns))))

    target = seeker_path.parent.name
    return {
        "seeker_path": str(seeker_path),
        "experiment": stratum.experiment,
        "model_slug": stratum.model_slug,
        "domain": stratum.domain,
        "mode": stratum.mode,
        "cot": stratum.cot,
        "target": target,
        "num_turns": len(turns),
        "turns": results,
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def summarise(conv_results: list[dict[str, Any]]) -> dict[str, Any]:
    type_counts: Counter[str] = Counter()
    redundancy_counts: Counter[str] = Counter()
    subclass_examples: list[dict[str, Any]] = []
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
            if cls.get("subclass"):
                sub = cls["subclass"]
                pc = sub.get("proposed_class")
                if pc:
                    subclass_by_type[qt][pc.strip().lower()] += 1
                    subclass_examples.append(
                        {
                            "proposed_class": pc,
                            "rationale": sub.get("rationale", ""),
                            "question_type": qt,
                            "question": turn["question"],
                            "domain": conv["domain"],
                            "experiment": conv["experiment"],
                            "target": conv["target"],
                            "turn": turn["turn"],
                        }
                    )

    subclass_counts = Counter(e["proposed_class"].strip().lower() for e in subclass_examples)

    return {
        "total_conversations": len(conv_results),
        "total_turns": total_turns,
        "errors": errors,
        "question_type_counts": dict(type_counts),
        "redundancy_counts": dict(redundancy_counts),
        "per_stratum_question_type_counts": {k: dict(v) for k, v in per_stratum.items()},
        "subclass_counts": dict(subclass_counts.most_common()),
        "subclass_by_question_type": {k: dict(v) for k, v in subclass_by_type.items()},
        "subclass_examples": subclass_examples,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


async def _amain(args: argparse.Namespace) -> int:
    rng = random.Random(args.seed)

    if args.seeker_file is not None:
        # single file mode — infer stratum from path
        parts = args.seeker_file.resolve().parts
        try:
            models_idx = parts.index("models")
            model_slug = parts[models_idx + 1]
            exp_name = parts[models_idx + 2]
        except (ValueError, IndexError):
            model_slug, exp_name = "unknown", "unknown"
        stratum = _parse_experiment(model_slug, exp_name) or Stratum(
            model_slug=model_slug, experiment=exp_name, domain="unknown", mode="fo", cot=False
        )
        picks = [(stratum, args.seeker_file)]
    else:
        buckets = discover_conversations(args.outputs_root)
        print(f"Discovered {sum(len(v) for v in buckets.values())} conversations across {len(buckets)} strata.")
        for st, paths in sorted(buckets.items(), key=lambda kv: (kv[0].domain, kv[0].mode, kv[0].cot)):
            print(f"  {st.domain}/{st.mode}/{'cot' if st.cot else 'no_cot'} [{st.model_slug}/{st.experiment}]: {len(paths)}")
        picks = stratified_sample(buckets, args.per_stratum, rng)
        print(f"Sampled {len(picks)} conversations (per_stratum={args.per_stratum}, seed={args.seed}).")

    if args.dry_run:
        for st, path in picks:
            print(f"  would classify: {st.domain}/{st.mode}/{'cot' if st.cot else 'no_cot'} :: {path}")
        return 0

    if not picks:
        print("No conversations to classify. Exiting.", file=sys.stderr)
        return 1

    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key)
    thinking = not args.no_thinking
    semaphore = asyncio.Semaphore(args.max_concurrency)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    total = len(picks)
    done = 0
    lock = asyncio.Lock()

    def _out_path(stratum: Stratum, path: Path) -> Path:
        target = path.parent.name
        return args.out_dir / stratum.experiment / target / "classification.json"

    # Resumability: split into already-done (reloaded) vs to-do (need LLM calls).
    existing: list[dict[str, Any]] = []
    todo: list[tuple[Stratum, Path]] = []
    for st, path in picks:
        p = _out_path(st, path)
        if not args.force and p.exists():
            try:
                existing.append(json.loads(p.read_text()))
                continue
            except Exception as e:  # noqa: BLE001 — corrupt file, re-run
                print(f"  warning: could not reload {p} ({e}); will re-classify", file=sys.stderr)
        todo.append((st, path))
    print(f"Resume: {len(existing)} already classified, {len(todo)} to do.")

    todo_total = len(todo)

    async def _run_conv(stratum: Stratum, path: Path) -> dict[str, Any] | None:
        nonlocal done
        try:
            res = await classify_conversation(
                seeker_path=path,
                stratum=stratum,
                client=client,
                model=args.model,
                thinking=thinking,
                semaphore=semaphore,
            )
        except Exception as e:  # noqa: BLE001
            async with lock:
                done += 1
                print(f"[{done}/{todo_total}] FAILED {path}: {type(e).__name__}: {e}", file=sys.stderr)
            return None

        out_path = _out_path(stratum, path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(res, indent=2, ensure_ascii=False))
        async with lock:
            done += 1
            print(f"[{done}/{todo_total}] done: {stratum.experiment}/{path.parent.name}")
        return res

    try:
        results = await asyncio.gather(*(_run_conv(st, path) for st, path in todo))
    finally:
        await client.close()

    conv_results = existing + [r for r in results if r is not None]

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
    p.add_argument(
        "--seeker-file",
        type=Path,
        default=None,
        help="Classify a single seeker.json (overrides sampling).",
    )
    p.add_argument("--per-stratum", type=int, default=3, help="Conversations sampled per stratum.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--max-concurrency",
        type=int,
        default=16,
        help="Max in-flight LLM requests at any moment (global cap across all conversations).",
    )
    p.add_argument("--no-thinking", action="store_true", help="Disable reasoning mode on the classifier.")
    p.add_argument("--dry-run", action="store_true", help="List what would be classified and exit.")
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-classify conversations even if classification.json already exists.",
    )
    args = p.parse_args()
    return asyncio.run(_amain(args))


if __name__ == "__main__":
    sys.exit(main())
