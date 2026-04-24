#!/usr/bin/env python3
"""Validate Oracle outputs against the response format defined in the prompt.

Scans every ``oracle.json`` under an outputs directory and flags turns where
the Oracle deviated from the expected ``Yes``/``No`` format (see
``src/prompts/oracle_system.md``).

Severities:
  - error  : JSON inválido, answer não inicia com yes/no (e game_over=false),
             ou vazamento do target (label/alias presente no answer com
             game_over=false).
  - warning: answer começa com yes/no mas tem texto extra
             ("Yes, but ...", "No. It is not ..."). O prompt permite info
             mínima em casos raros; marcamos para revisão.

Outputs:
  - <base>/oracle_validation.csv    — uma linha por issue
  - <base>/oracle_validation.json   — sumário por experimento
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


# Mirrors src/utils/utils.py — inlined to keep this script self-contained
# (importing the package pulls yaml/dotenv unnecessarily).
def llm_final_content(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<think>.*", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r".*</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def parse_first_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    try:
        obj = json.loads(stripped)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    match = re.search(r"\{[\s\S]*\}", stripped)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


# "Yes" / "No" possibly followed by punctuation and more text.
YES_NO_PREFIX = re.compile(r"^\s*(yes|no)\b", re.IGNORECASE)
YES_NO_ONLY = re.compile(r"^\s*(yes|no)[\s.!]*$", re.IGNORECASE)


def iter_oracle_files(base_dir: Path) -> Iterable[Path]:
    return sorted(base_dir.rglob("oracle.json"))


def collect_aliases(target: dict[str, Any]) -> list[str]:
    labels = [target.get("label", "")]
    aliases = (target.get("attrs") or {}).get("aliases") or []
    if isinstance(aliases, (list, tuple)):
        labels.extend(str(a) for a in aliases)
    return [lbl.lower().strip() for lbl in labels if lbl]


def leaks_target(answer: str, target_labels: list[str]) -> str | None:
    low = answer.lower()
    for lbl in target_labels:
        if lbl and lbl in low:
            return lbl
    return None


def validate_oracle_file(oracle_path: Path) -> list[dict[str, Any]]:
    """Return a list of issue dicts for the given oracle.json."""
    with oracle_path.open(encoding="utf-8") as f:
        data = json.load(f)

    target = data.get("target", {})
    target_labels = collect_aliases(target)
    history = data.get("history", [])
    conversation_dir = str(oracle_path.parent)
    experiment_dir = str(oracle_path.parent.parent.parent)

    issues: list[dict[str, Any]] = []
    last_question = ""
    turn_index = 0

    for msg in history:
        role = msg.get("role")
        content = msg.get("content", "")

        if role == "user":
            # "[Seeker] - <question>"
            last_question = content.split(" - ", 1)[-1] if " - " in content else content
            turn_index += 1
            continue

        if role != "assistant":
            continue

        base = {
            "experiment_dir": experiment_dir,
            "conversation_dir": conversation_dir,
            "target_id": target.get("id", ""),
            "target_label": target.get("label", ""),
            "turn_index": turn_index,
            "question": last_question,
        }

        # 1) JSON validity — strip <think> / code fences first (same as runtime)
        cleaned = llm_final_content(content)
        parsed = parse_first_json_object(cleaned)
        if parsed is None:
            issues.append({
                **base,
                "severity": "error",
                "issue_type": "invalid_json",
                "answer": cleaned[:200] if cleaned else content[:200],
                "detail": "could not parse JSON object",
            })
            continue

        answer = str(parsed.get("answer", ""))
        game_over = bool(parsed.get("game_over", False))

        # 2) yes/no prefix
        if not YES_NO_PREFIX.match(answer):
            # Only an error if game hasn't ended — clarification/info requests
            # are technically allowed but should be rare and are worth seeing.
            issues.append({
                **base,
                "severity": "error" if not game_over else "warning",
                "issue_type": "no_yes_no_prefix",
                "answer": answer[:200],
                "detail": f"game_over={game_over}",
            })
        elif not YES_NO_ONLY.match(answer) and not game_over:
            # Starts with yes/no but has extra text — flag as warning
            issues.append({
                **base,
                "severity": "warning",
                "issue_type": "yes_no_with_extra_text",
                "answer": answer[:200],
                "detail": "",
            })

        # 3) Target leak (only meaningful when game_over=false)
        if not game_over:
            leak = leaks_target(answer, target_labels)
            if leak:
                issues.append({
                    **base,
                    "severity": "error",
                    "issue_type": "target_leak",
                    "answer": answer[:200],
                    "detail": f"contains '{leak}'",
                })

    return issues


def summarize(all_issues: list[dict[str, Any]]) -> dict[str, Any]:
    by_experiment: dict[str, Counter] = defaultdict(Counter)
    by_type = Counter()
    conversations_with_errors: set[str] = set()

    for issue in all_issues:
        by_experiment[issue["experiment_dir"]][issue["issue_type"]] += 1
        by_type[issue["issue_type"]] += 1
        if issue["severity"] == "error":
            conversations_with_errors.add(issue["conversation_dir"])

    return {
        "total_issues": len(all_issues),
        "total_errors": sum(1 for i in all_issues if i["severity"] == "error"),
        "total_warnings": sum(1 for i in all_issues if i["severity"] == "warning"),
        "conversations_with_errors": len(conversations_with_errors),
        "issues_by_type": dict(by_type),
        "issues_by_experiment": {
            exp: dict(counter) for exp, counter in sorted(by_experiment.items())
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "base_dir",
        nargs="?",
        type=Path,
        default=Path("outputs"),
        help="Root directory to scan for oracle.json files (default: outputs)",
    )
    ap.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="CSV output path (default: <base_dir>/oracle_validation.csv)",
    )
    ap.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Summary JSON output path (default: <base_dir>/oracle_validation.json)",
    )
    ap.add_argument(
        "--errors-only",
        action="store_true",
        help="Write only errors to CSV (warnings always counted in summary).",
    )
    args = ap.parse_args()

    base = args.base_dir
    if not base.exists():
        print(f"error: {base} does not exist")
        return 1

    csv_path = args.csv or base / "oracle_validation.csv"
    json_path = args.json or base / "oracle_validation.json"

    all_issues: list[dict[str, Any]] = []
    files = list(iter_oracle_files(base))
    print(f"Scanning {len(files)} oracle.json files under {base}...")

    for i, path in enumerate(files, 1):
        try:
            issues = validate_oracle_file(path)
        except Exception as e:
            print(f"  [{i}/{len(files)}] ERROR reading {path}: {e}")
            continue
        all_issues.extend(issues)

    csv_rows = all_issues if not args.errors_only else [
        i for i in all_issues if i["severity"] == "error"
    ]

    fieldnames = [
        "severity", "issue_type", "experiment_dir", "conversation_dir",
        "target_id", "target_label", "turn_index", "question", "answer", "detail",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    summary = summarize(all_issues)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Scan complete: {len(files)} files, {summary['total_issues']} issues")
    print(f"   errors:   {summary['total_errors']}")
    print(f"   warnings: {summary['total_warnings']}")
    print(f"   conversations with errors: {summary['conversations_with_errors']}")
    print(f"   issue types: {summary['issues_by_type']}")
    print(f"\n💾 CSV:     {csv_path}")
    print(f"💾 Summary: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
