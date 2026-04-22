"""Cold-start test: system → assistant (no user turn).

Tests whether each available model degenerates when the first LLM call
contains only a system message — exactly the bug in PO mode (seeker.py:56-65 +
orchestrator.py:193-197).

Three conditions per model:
  T1  system only            (reproduces the PO bug)
  T2  system + user kickoff  (proposed fix / control)
  T3  system + user + assistant + user  (turn 2 baseline)

Usage:
    python3 scripts/test_cold_start.py
    python3 scripts/test_cold_start.py --reps 10
    python3 scripts/test_cold_start.py --endpoints '{"Qwen3-8B": "http://10.100.0.112:8481/v1"}'
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path
from typing import Optional

from openai import OpenAI

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.prompts import get_seeker_system_prompt

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_ENDPOINTS: dict[str, str] = {
    "Qwen3-8B":                   "http://10.100.0.112:8481/v1",
    "Qwen3-30B-A3B-Thinking-2507": "http://10.100.0.112:8480/v1",
    "Qwen3-4B-Thinking-2507":     "http://10.100.0.113:9830/v1",
    "Nemotron-Cascade-8B":        "http://10.100.0.112:8479/v1",
}

SYSTEM_PROMPT = get_seeker_system_prompt(
    target_noun="city",
    domain_description="geographic (cities, countries, regions)",
    max_turns=25,
    observability_mode="PARTIALLY_OBSERVABLE",
)

KICKOFF_USER   = "Start the game. Ask your first question."
MOCK_ASSISTANT = "Is it in Europe?"
TURN2_USER     = "[Turn 1/25] [Oracle] - Yes"

CONDITIONS: dict[str, list[dict]] = {
    "T1_system_only": [
        {"role": "system", "content": SYSTEM_PROMPT},
    ],
    "T2_with_kickoff": [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": KICKOFF_USER},
    ],
    "T3_turn2_baseline": [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": KICKOFF_USER},
        {"role": "assistant", "content": MOCK_ASSISTANT},
        {"role": "user",      "content": TURN2_USER},
    ],
}

# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

REFUSAL_TOKENS = ("i cannot", "i'm sorry", "i am sorry", "as an ai", "i don't know",
                  "i am not able", "i'm unable", "i am unable")
FALLBACK_TOKENS = ("what is the target", "what is the secret", "what are we looking for")


def classify(text: str) -> str:
    low = text.lower().strip()
    if not low:
        return "EMPTY"
    if any(t in low for t in REFUSAL_TOKENS):
        return "REFUSAL"
    if any(t in low for t in FALLBACK_TOKENS):
        return "FALLBACK"
    # A valid yes/no question ends with "?" and is short
    has_question_mark = "?" in text
    is_short = len(text.split()) <= 25
    if has_question_mark and is_short:
        return "VALID"
    if has_question_mark:
        return "LONG_QUESTION"
    return "FORMAT_BROKEN"


def has_think_tag(text: str) -> bool:
    return "<think>" in text.lower()


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def call_model(
    client: OpenAI,
    model: str,
    messages: list[dict],
    temperature: float = 0.6,
    max_tokens: int = 512,
) -> tuple[str, Optional[str]]:
    """Return (final_content_without_think, raw_with_think_or_None).

    Handles two vLLM layouts:
      - reasoning in model_extra["reasoning"] / ["reasoning_content"]
      - reasoning embedded directly in msg.content as <think>...</think>
    """
    import re

    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    msg = completion.choices[0].message
    content = msg.content or ""
    extras = msg.model_extra or {}
    reasoning = extras.get("reasoning") or extras.get("reasoning_content")

    if reasoning:
        # Layout 1: reasoning in separate field
        raw = f"<think>{reasoning}</think>{content}"
        final = content.strip()
    elif "<think>" in content:
        # Layout 2: reasoning embedded in content
        raw = content
        final = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    else:
        raw = None
        final = content.strip()

    return final, raw


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_tests(endpoints: dict[str, str], reps: int) -> None:
    results: dict[str, dict] = {}

    for model_name, base_url in endpoints.items():
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"URL:   {base_url}")
        print(f"{'='*60}")
        client = OpenAI(api_key="EMPTY", base_url=base_url)
        results[model_name] = {}

        for cond_name, messages in CONDITIONS.items():
            print(f"\n  [{cond_name}]  ({reps} reps)")
            responses: list[str] = []
            classifications: list[str] = []
            think_present: list[bool] = []

            for i in range(reps):
                try:
                    content, raw = call_model(client, model_name, messages)
                    label = classify(content)
                    think = has_think_tag(raw) if raw else has_think_tag(content)
                    responses.append(content)
                    classifications.append(label)
                    think_present.append(think)

                    short = textwrap.shorten(content, width=80, placeholder="…")
                    think_tag = " [<think>✓]" if think else ""
                    print(f"    rep {i+1:2d}: [{label}]{think_tag}  {short!r}")
                except Exception as exc:
                    print(f"    rep {i+1:2d}: ERROR — {exc}")
                    responses.append("")
                    classifications.append("ERROR")
                    think_present.append(False)

            # Aggregate
            from collections import Counter
            counts = Counter(classifications)
            unique_responses = len(set(r for r in responses if r))
            think_rate = sum(think_present) / len(think_present) if think_present else 0.0

            print(f"\n  Summary:")
            print(f"    classifications : {dict(counts)}")
            print(f"    unique responses: {unique_responses}/{reps}  (1 = pure deterministic fallback)")
            print(f"    <think> rate    : {think_rate:.0%}")

            results[model_name][cond_name] = {
                "classifications": dict(counts),
                "unique_responses": unique_responses,
                "think_rate": round(think_rate, 2),
                "responses": responses,
            }

    # ---------------------------------------------------------------------------
    # Cross-model summary table
    # ---------------------------------------------------------------------------
    print(f"\n\n{'='*60}")
    print("CROSS-MODEL SUMMARY")
    print(f"{'='*60}")
    header = f"{'Model':<35} {'Condition':<22} {'VALID':>6} {'FALLBACK':>9} {'REFUSAL':>8} {'OTHER':>6} {'Unique':>7} {'Think':>6}"
    print(header)
    print("-" * len(header))

    for model_name, conds in results.items():
        for cond_name, data in conds.items():
            c = data["classifications"]
            valid    = c.get("VALID", 0) + c.get("LONG_QUESTION", 0)
            fallback = c.get("FALLBACK", 0)
            refusal  = c.get("REFUSAL", 0)
            other    = reps - valid - fallback - refusal
            think    = f"{data['think_rate']:.0%}"
            unique   = data["unique_responses"]
            print(f"{model_name:<35} {cond_name:<22} {valid:>6} {fallback:>9} {refusal:>8} {other:>6} {unique:>7} {think:>6}")

    # Save JSON
    out_path = Path("outputs/cold_start_test_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cold-start system→assistant test")
    parser.add_argument("--reps", type=int, default=5,
                        help="Repetitions per condition (default: 5)")
    parser.add_argument("--endpoints", type=str, default=None,
                        help='JSON dict of model→url overrides')
    args = parser.parse_args()

    endpoints = DEFAULT_ENDPOINTS.copy()
    if args.endpoints:
        endpoints.update(json.loads(args.endpoints))

    run_tests(endpoints, args.reps)


if __name__ == "__main__":
    main()
