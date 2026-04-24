#!/usr/bin/env python3
"""Aggregate per-conversation judge results into two experiment-level CSVs.

Walks ``outputs/**/conversations/*/{oracle,pruner}_judge_eval.json`` and emits:

  - ``outputs/judge_oracle_summary.csv``
  - ``outputs/judge_pruner_summary.csv``

One row per conversation. Columns are designed to be joined with
``outputs/unified_experiments.csv`` via ``experiment`` + a
``s_<seeker>__o_<oracle>__p_<pruner>`` tag parsed from the conversation path.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUTS_BASE = PROJECT_ROOT / "outputs"

logger = logging.getLogger("aggregate_judge")


TRIPLE_RE = re.compile(r"s_(?P<seeker>[^/]+)__o_(?P<oracle>[^/]+)__p_(?P<pruner>[^/]+)")


def _triple_from_path(conv_dir: Path) -> tuple[str, str, str]:
    m = TRIPLE_RE.search(str(conv_dir))
    if not m:
        return "", "", ""
    return m.group("seeker"), m.group("oracle"), m.group("pruner")


def _collect_oracle_rows(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for p in paths:
        try:
            data = json.loads(p.read_text())
        except Exception as exc:
            logger.warning("skip %s: %s", p, exc)
            continue
        seeker, oracle, pruner = _triple_from_path(p.parent)
        summary = data.get("summary", {}) or {}
        rows.append({
            "conversation_path": str(p.parent.relative_to(OUTPUTS_BASE)),
            "experiment": data.get("experiment", ""),
            "target_id": data.get("target_id", ""),
            "target_label": data.get("target_label", ""),
            "seeker_model": seeker,
            "oracle_model": oracle or data.get("oracle_model", ""),
            "pruner_model": pruner,
            "judge_model": data.get("judge_model", ""),
            "n_turns": summary.get("n_turns", 0),
            "n_matches": summary.get("n_matches", 0),
            "agreement": summary.get("agreement", 0.0),
            "n_errors": summary.get("n_errors", 0),
            "YY": summary.get("yes_no_confusion", {}).get("YY", 0),
            "NN": summary.get("yes_no_confusion", {}).get("NN", 0),
            "YN": summary.get("yes_no_confusion", {}).get("YN", 0),
            "NY": summary.get("yes_no_confusion", {}).get("NY", 0),
            "conf_other": summary.get("yes_no_confusion", {}).get("other", 0),
        })
    return rows


def _collect_pruner_rows(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for p in paths:
        try:
            data = json.loads(p.read_text())
        except Exception as exc:
            logger.warning("skip %s: %s", p, exc)
            continue
        seeker, oracle, pruner = _triple_from_path(p.parent)
        summary = data.get("summary", {}) or {}
        rows.append({
            "conversation_path": str(p.parent.relative_to(OUTPUTS_BASE)),
            "experiment": data.get("experiment", ""),
            "target_id": data.get("target_id", ""),
            "target_label": data.get("target_label", ""),
            "seeker_model": seeker,
            "oracle_model": oracle,
            "pruner_model": pruner or data.get("pruner_model", ""),
            "judge_model": data.get("judge_model", ""),
            "n_turns": summary.get("n_turns", 0),
            "n_ok": summary.get("n_ok", 0),
            "n_errors": summary.get("n_errors", 0),
            "mean_jaccard": summary.get("mean_jaccard", 0.0),
            "n_target_removed_by_qwen": summary.get("n_target_removed_by_qwen", 0),
            "n_target_removed_by_judge": summary.get("n_target_removed_by_judge", 0),
        })
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        logger.info("No rows for %s — skipping.", path.name)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote %d rows → %s", len(rows), path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-dir", type=Path, default=OUTPUTS_BASE)
    parser.add_argument("--oracle-csv", type=Path,
                        default=OUTPUTS_BASE / "judge_oracle_summary.csv")
    parser.add_argument("--pruner-csv", type=Path,
                        default=OUTPUTS_BASE / "judge_pruner_summary.csv")
    args = parser.parse_args()

    oracle_files = sorted(args.outputs_dir.rglob("oracle_judge_eval.json"))
    pruner_files = sorted(args.outputs_dir.rglob("pruner_judge_eval.json"))
    logger.info("Found %d oracle, %d pruner judge files.", len(oracle_files), len(pruner_files))

    _write_csv(args.oracle_csv, _collect_oracle_rows(oracle_files))
    _write_csv(args.pruner_csv, _collect_pruner_rows(pruner_files))


if __name__ == "__main__":
    sys.exit(main())
