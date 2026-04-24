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
    "Qwen3-8B":                    "http://10.100.0.112:9851/v1",
    "Qwen3-4B-Thinking-2507":      "http://10.100.0.113:9830/v1",
    "Nemotron-Cascade-8B":         "http://10.100.0.113:9831/v1",
    "Llama-3.2-1B-Instruct": "http://10.100.0.112:9850/v1",
    "Llama-3.2-3B-Instruct": "http://10.100.0.112:9860/v1",
    "Qwen3-0.6B": "http://10.100.0.121:10420/v1"
}



# Models that default to thinking ON — get enable_thinking=True unless --no-thinking is passed.
THINKING_MODELS: set[str] = {
    "Qwen3-4B-Thinking-2507",
    "Qwen3-8B",
    "Qwen3-0.6B",
    "Nemotron-Cascade-8B",
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


def has_think_tag(text: Optional[str]) -> bool:
    if not text:
        return False
    # vLLM may return reasoning directly in content (no tags) — treat long raw as thinking
    return "<think>" in text.lower() or len(text) > 500


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def call_model(
    client: OpenAI,
    model: str,
    messages: list[dict],
    max_tokens: int = 24000,
    enable_thinking: bool | None = None,
    temperature: float | None = None,
) -> dict:
    """Call the model and return a dict with all captured fields.

    Handles two vLLM layouts:
      - reasoning in model_extra["reasoning"] / ["reasoning_content"]
      - reasoning embedded directly in msg.content as <think>...</think>

    enable_thinking:
      True  → send enable_thinking=True
      False → send enable_thinking=False (explicit disable for capable models)
      None  → omit the field entirely
    """
    kwargs: dict = dict(model=model, messages=messages, max_tokens=max_tokens)
    if enable_thinking is not None:
        kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": enable_thinking}}
    if temperature is not None:
        kwargs["temperature"] = temperature

    completion = client.chat.completions.create(**kwargs)
    msg = completion.choices[0].message
    choice = completion.choices[0]
    content = msg.content or ""
    extras = msg.model_extra or {}
    reasoning = extras.get("reasoning") or extras.get("reasoning_content")

    # Final content: strip reasoning tags if embedded
    final = content.split("</think>")[-1].strip() if content else ""

    usage = getattr(completion, "usage", None)
    usage_dict = usage.model_dump() if usage is not None else None

    # Full raw response dump (all provider fields) for downstream analysis.
    try:
        completion_full = completion.model_dump()
    except Exception:
        completion_full = None

    return {
        "content_raw": content,
        "content_final": final,
        "reasoning": reasoning,
        "finish_reason": getattr(choice, "finish_reason", None),
        "usage": usage_dict,
        "enable_thinking": enable_thinking,
        "request_kwargs": kwargs,
        "completion_full": completion_full,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _slug(name: str) -> str:
    """Filesystem-safe version of a model/condition name."""
    return name.replace("/", "_").replace(" ", "_")


def _dump_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)


def run_tests(
    endpoints: dict[str, str],
    reps: int,
    out_dir: Path,
    *,
    disable_thinking: bool = False,
    temperature: float | None = None,
) -> None:
    from collections import Counter
    from datetime import datetime

    run_started = datetime.now().isoformat(timespec="seconds")
    temp_tag = f"_t{temperature:.2f}".replace(".", "_") if temperature is not None else ""
    suffix = ("_no_think" if disable_thinking else "") + temp_tag
    results: dict[str, dict] = {}
    manifest_reps: list[dict] = []

    for model_name, base_url in endpoints.items():
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"URL:   {base_url}")
        print(f"{'='*60}")
        client = OpenAI(api_key="EMPTY", base_url=base_url)
        if disable_thinking:
            thinking: bool | None = False  # explicit disable for all models
        elif model_name in THINKING_MODELS:
            thinking = True  # explicit enable for known thinking models
        else:
            thinking = None  # omit flag — model's default behavior
        model_folder = _slug(model_name) + ("_no_think" if disable_thinking else "") + temp_tag
        results[model_name] = {
            "base_url": base_url,
            "enable_thinking": thinking,
            "conditions": {},
        }

        for cond_name, messages in CONDITIONS.items():
            print(f"\n  [{cond_name}]  ({reps} reps)")
            cond_dir = out_dir / "by_model" / model_folder / _slug(cond_name)
            cond_dir.mkdir(parents=True, exist_ok=True)
            _dump_json(cond_dir / "messages.json", messages)

            reps_data: list[dict] = []
            classifications: list[str] = []
            think_present: list[bool] = []

            for i in range(reps):
                rep_ts = datetime.now().isoformat(timespec="milliseconds")
                rep_num = i + 1
                try:
                    call = call_model(client, model_name, messages, enable_thinking=thinking, temperature=temperature)
                    final = call["content_final"]
                    raw = call["content_raw"]
                    reasoning = call["reasoning"]
                    label = classify(final)
                    think = has_think_tag(reasoning) or has_think_tag(raw) or has_think_tag(final)

                    rep_entry = {
                        "rep": rep_num,
                        "timestamp": rep_ts,
                        "classification": label,
                        "think_present": think,
                        "content_final": final,
                        "content_raw": raw,
                        "reasoning": reasoning,
                        "finish_reason": call["finish_reason"],
                        "usage": call["usage"],
                        "request_kwargs": call["request_kwargs"],
                        "completion_full": call["completion_full"],
                    }
                    reps_data.append(rep_entry)
                    classifications.append(label)
                    think_present.append(think)

                    short = textwrap.shorten(final, width=80, placeholder="…")
                    think_tag = " [<think>✓]" if think else ""
                    print(f"    rep {rep_num:2d}: [{label}]{think_tag}  {short!r}")
                except Exception as exc:
                    print(f"    rep {rep_num:2d}: ERROR — {exc}")
                    rep_entry = {
                        "rep": rep_num,
                        "timestamp": rep_ts,
                        "classification": "ERROR",
                        "think_present": False,
                        "error": repr(exc),
                        "content_final": "",
                        "content_raw": "",
                        "reasoning": None,
                    }
                    reps_data.append(rep_entry)
                    classifications.append("ERROR")
                    think_present.append(False)

                rep_path = cond_dir / f"rep_{rep_num:02d}.json"
                _dump_json(rep_path, rep_entry)
                manifest_reps.append({
                    "model": model_name,
                    "condition": cond_name,
                    "rep": rep_num,
                    "classification": rep_entry["classification"],
                    "think_present": rep_entry["think_present"],
                    "path": str(rep_path.relative_to(out_dir)),
                })

            counts = Counter(classifications)
            unique_responses = len(set(r["content_final"] for r in reps_data if r.get("content_final")))
            think_rate = sum(think_present) / len(think_present) if think_present else 0.0

            print(f"\n  Summary:")
            print(f"    classifications : {dict(counts)}")
            print(f"    unique responses: {unique_responses}/{reps}  (1 = pure deterministic fallback)")
            print(f"    <think> rate    : {think_rate:.0%}")

            results[model_name]["conditions"][cond_name] = {
                "messages_sent": messages,
                "classifications": dict(counts),
                "unique_responses": unique_responses,
                "think_rate": round(think_rate, 2),
                "reps": reps_data,
            }

            # Also dump per-condition plain-text log for easy eyeballing
            txt_path = cond_dir / "log.txt"
            with open(txt_path, "w") as f:
                f.write(f"Model: {model_name}\nURL: {base_url}\nCondition: {cond_name}\n")
                f.write(f"Enable thinking: {thinking}\n")
                f.write(f"{'='*60}\nMESSAGES SENT\n{'='*60}\n")
                for m in messages:
                    f.write(f"\n--- {m['role']} ---\n{m['content']}\n")
                f.write(f"\n{'='*60}\nREPS\n{'='*60}\n")
                for r in reps_data:
                    f.write(f"\n--- rep {r['rep']} [{r['classification']}] "
                            f"think={r.get('think_present')} finish={r.get('finish_reason')} ---\n")
                    if r.get("error"):
                        f.write(f"ERROR: {r['error']}\n")
                        continue
                    if r.get("reasoning"):
                        f.write(f"[reasoning]\n{r['reasoning']}\n")
                    f.write(f"[content_raw]\n{r.get('content_raw','')}\n")
                    f.write(f"[content_final]\n{r.get('content_final','')}\n")

    # ---------------------------------------------------------------------------
    # Cross-model summary table
    # ---------------------------------------------------------------------------
    print(f"\n\n{'='*60}")
    print("CROSS-MODEL SUMMARY")
    print(f"{'='*60}")
    header = f"{'Model':<35} {'Condition':<22} {'VALID':>6} {'FALLBACK':>9} {'REFUSAL':>8} {'OTHER':>6} {'Unique':>7} {'Think':>6}"
    print(header)
    print("-" * len(header))

    summary_rows: list[dict] = []
    for model_name, model_data in results.items():
        for cond_name, data in model_data["conditions"].items():
            c = data["classifications"]
            valid    = c.get("VALID", 0) + c.get("LONG_QUESTION", 0)
            fallback = c.get("FALLBACK", 0)
            refusal  = c.get("REFUSAL", 0)
            other    = reps - valid - fallback - refusal
            think    = f"{data['think_rate']:.0%}"
            unique   = data["unique_responses"]
            print(f"{model_name:<35} {cond_name:<22} {valid:>6} {fallback:>9} {refusal:>8} {other:>6} {unique:>7} {think:>6}")
            summary_rows.append({
                "model": model_name, "condition": cond_name,
                "valid": valid, "fallback": fallback, "refusal": refusal, "other": other,
                "unique": unique, "think_rate": data["think_rate"],
            })

    # Save full JSON (everything)
    full_path = out_dir / f"cold_start_test_results{suffix}.json"
    _dump_json(full_path, {
        "run_started": run_started,
        "reps_per_condition": reps,
        "system_prompt": SYSTEM_PROMPT,
        "endpoints": endpoints,
        "results": results,
    })

    # Compact summary CSV for quick grepping/plotting
    import csv
    csv_path = out_dir / f"cold_start_summary{suffix}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()) if summary_rows else [])
        w.writeheader()
        w.writerows(summary_rows)

    # Index manifest: one row per rep, with relative path to its JSON file.
    manifest_path = out_dir / f"manifest{suffix}.json"
    _dump_json(manifest_path, {
        "run_started": run_started,
        "reps_per_condition": reps,
        "conditions": list(CONDITIONS),
        "endpoints": endpoints,
        "disable_thinking": disable_thinking,
        "reps": manifest_reps,
    })

    print(f"\nOut dir       : {out_dir}")
    print(f"Full results  : {full_path}")
    print(f"Summary CSV   : {csv_path}")
    print(f"Manifest      : {manifest_path}")
    model_suffix = "[_no_think]" if disable_thinking else ""
    print(f"Per-rep JSON  : {out_dir / 'by_model'}/<model>{model_suffix}/<condition>/rep_XX.json")
    print(f"Per-cond logs : {out_dir / 'by_model'}/<model>{model_suffix}/<condition>/log.txt")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cold-start system→assistant test")
    parser.add_argument("--reps", type=int, default=10,
                        help="Repetitions per condition (default: 10)")
    parser.add_argument("--endpoints", type=str, default=None,
                        help='JSON dict of model→url overrides')
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/cold_start"),
                        help="Output directory (default: outputs/cold_start)")
    parser.add_argument("--model", type=str, default=None,
                        help="Run only this model (must exist in DEFAULT_ENDPOINTS or --endpoints)")
    parser.add_argument("--no-thinking", action="store_true",
                        help="Disable chat_template_kwargs.enable_thinking for all models")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Sampling temperature (e.g. 0.7). Omitted by default (model default).")
    args = parser.parse_args()

    endpoints = DEFAULT_ENDPOINTS.copy()
    if args.endpoints:
        endpoints.update(json.loads(args.endpoints))

    if args.model:
        if args.model not in endpoints:
            parser.error(f"model {args.model!r} not found. Available: {sorted(endpoints)}")
        endpoints = {args.model: endpoints[args.model]}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_tests(endpoints, args.reps, args.out_dir, disable_thinking=args.no_thinking, temperature=args.temperature)


if __name__ == "__main__":
    main()
