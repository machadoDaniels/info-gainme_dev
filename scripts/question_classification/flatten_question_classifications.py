#!/usr/bin/env python3
"""Flatten per-conversation classification JSONs into one wide CSV.

Each row of the output CSV is a single seeker turn with its full label set.
The resulting file feeds pandas / notebooks / SQL directly — no per-file
parsing needed downstream.

Input:
    outputs/question_classification/<experiment>/<target>/classification.json

Output (default):
    outputs/question_classifications.csv

Columns:
    seeker, oracle, pruner, model_slug,
    experiment, domain, mode (fo/po), cot (0/1),
    target, run_index,
    turn, question, oracle_answer,
    question_type_rationale, question_type, subclass, subclass_rationale,
    redundancy, redundant_with_turn,
    error (empty on success)
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any


_MODEL_SLUG_RE = re.compile(r"^s_(?P<seeker>.+?)__o_(?P<oracle>.+?)__p_(?P<pruner>.+)$")
_RUN_SUFFIX_RE = re.compile(r"_run(\d+)$")


def parse_model_slug(slug: str) -> tuple[str, str, str]:
    m = _MODEL_SLUG_RE.match(slug)
    if m:
        return m["seeker"], m["oracle"], m["pruner"]
    return slug, "", ""


def parse_run_index(target_folder: str) -> tuple[str, int | None]:
    """Split ``disease-xxx-N_run02`` into (``disease-xxx-N``, 2).

    Human baselines have no ``_runNN`` suffix — returns (target, None) for those.
    """
    m = _RUN_SUFFIX_RE.search(target_folder)
    if m:
        return target_folder[: m.start()], int(m.group(1))
    return target_folder, None


COLUMNS = [
    "seeker",
    "oracle",
    "pruner",
    "model_slug",
    "experiment",
    "domain",
    "mode",
    "cot",
    "target",
    "run_index",
    "turn",
    "question",
    "oracle_answer",
    "question_type_rationale",
    "question_type",
    "subclass",
    "subclass_rationale",
    "redundancy",
    "redundant_with_turn",
    "error",
]


def iter_rows(conv: dict[str, Any]) -> list[dict[str, Any]]:
    seeker, oracle, pruner = parse_model_slug(conv.get("model_slug", ""))
    target_folder = conv.get("target", "")
    target_base, run_idx = parse_run_index(target_folder)

    base = {
        "seeker": seeker,
        "oracle": oracle,
        "pruner": pruner,
        "model_slug": conv.get("model_slug", ""),
        "experiment": conv.get("experiment", ""),
        "domain": conv.get("domain", ""),
        "mode": conv.get("mode", ""),
        "cot": 1 if conv.get("cot") else 0,
        "target": target_base,
        "run_index": run_idx if run_idx is not None else "",
    }

    rows: list[dict[str, Any]] = []
    for t in conv.get("turns", []):
        cls = t.get("classification", {}) or {}
        sub = cls.get("subclass") or {}
        row = dict(base)
        row.update(
            {
                "turn": t.get("turn", ""),
                "question": (t.get("question") or "").replace("\n", " ").strip(),
                "oracle_answer": (t.get("oracle_answer") or "").replace("\n", " ").strip(),
                "question_type_rationale": cls.get("question_type_rationale", "") if "error" not in cls else "",
                "question_type": cls.get("question_type", "") if "error" not in cls else "",
                "subclass": sub.get("proposed_class", "") if isinstance(sub, dict) else "",
                "subclass_rationale": (sub.get("subclass_rationale") or sub.get("rationale", "")) if isinstance(sub, dict) else "",
                "redundancy": cls.get("redundancy", "") if "error" not in cls else "",
                "redundant_with_turn": cls.get("redundant_with_turn", "") if "error" not in cls else "",
                "error": cls.get("error", ""),
            }
        )
        rows.append(row)
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--input-dir",
        type=Path,
        default=Path("outputs/question_classification"),
        help="Root holding <experiment>/<target>/classification.json files.",
    )
    p.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/question_classifications.csv"),
        help="Destination CSV.",
    )
    args = p.parse_args()

    files = sorted(args.input_dir.glob("*/*/classification.json"))
    if not files:
        print(f"No classification.json files found under {args.input_dir}", file=sys.stderr)
        return 1

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    total_turns = 0
    skipped = 0
    with args.output_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        for i, path in enumerate(files, 1):
            try:
                conv = json.loads(path.read_text())
            except Exception as e:  # noqa: BLE001 — corrupt files should not abort the batch
                print(f"  skip {path}: {e}", file=sys.stderr)
                skipped += 1
                continue
            for row in iter_rows(conv):
                writer.writerow(row)
                total_turns += 1
            if i % 500 == 0:
                print(f"  processed {i}/{len(files)} conversations ({total_turns} rows so far)")

    print(f"\nWrote {total_turns} rows from {len(files) - skipped} conversations to {args.output_csv}")
    if skipped:
        print(f"Skipped {skipped} unreadable file(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
