#!/usr/bin/env python3
"""Tag every conversation as ``ont`` (oracle ran without thinking) or ``clean``.

Detection criterion: in the oracle's ``reasoning_history``, the FIRST
assistant message must contain a ``<think>`` block. If it doesn't, the run
is marked ``is_ont=True``.

False-positive risk: oracles that legitimately don't think (e.g.,
``Nemotron-Cascade-8B`` without a reasoning parser) will be flagged as
``is_ont`` even though they ran correctly. Use ``--oracle-only`` to limit
detection to a single oracle model (e.g., ``--oracle-only Qwen3-8B``) — for
ablation runs with non-thinking oracles, those rows will be marked
``not_applicable`` instead.

Usage:
    # Default: scan all conversations, write outputs/ont_detection.csv
    python3 scripts/maintenance/detect_ont_runs.py

    # Only flag Qwen3-8B oracles (recommended — avoids false positives on ablation)
    python3 scripts/maintenance/detect_ont_runs.py --oracle-only Qwen3-8B

    # Custom output path / verbose progress
    python3 scripts/maintenance/detect_ont_runs.py --csv /tmp/det.csv -v

The CSV has columns:
    triple, exp, conv_dir, target_id, run_index, oracle_model,
    is_canonical_dir, is_ont, reason

``is_canonical_dir`` is False if the conversation was found inside an
``<exp>_ont/`` directory (audit trail; flagged ``is_ont=True`` automatically).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def first_assistant_thinking(oracle_path: Path) -> tuple[str, str]:
    """Return ('present'|'absent'|'no_history'|'parse_error', reason).

    Looks at oracle.json's reasoning_history list and finds the first entry
    with role='assistant'. Checks if its content starts with or contains
    a ``<think>`` block.
    """
    try:
        with open(oracle_path) as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        return "parse_error", str(e)

    rh = data.get("reasoning_history")
    if not isinstance(rh, list):
        return "no_history", "reasoning_history missing or not a list"

    for entry in rh:
        if not isinstance(entry, dict):
            continue
        if entry.get("role") != "assistant":
            continue
        content = entry.get("content", "") or ""
        # Allow whitespace before the <think> tag
        if "<think>" in content[:200]:
            return "present", "<think> found in first 200 chars"
        return "absent", f"first 80 chars: {content[:80]!r}"
    return "no_history", "no assistant entry in reasoning_history"


_TRIPLE_RE = re.compile(r"^s_(?P<seeker>.+?)__o_(?P<oracle>.+?)__p_(?P<pruner>.+)$")


def oracle_from_triple_name(triple_name: str) -> str | None:
    m = _TRIPLE_RE.match(triple_name)
    return m.group("oracle") if m else None


def oracle_model_from_metadata(meta_path: Path) -> str | None:
    try:
        with open(meta_path) as f:
            data = json.load(f)
        return data.get("config", {}).get("models", {}).get("oracle")
    except Exception:
        return None


def walk_conversations(out_root: Path, exp_filter: re.Pattern | None = None):
    """Yield (triple_dir, exp_dir, conv_dir, is_canonical_dir)."""
    for triple in sorted(out_root.iterdir()):
        if not triple.is_dir() or not triple.name.startswith("s_"):
            continue
        for exp in sorted(triple.iterdir()):
            if not exp.is_dir():
                continue
            if exp_filter and not exp_filter.search(f"{triple.name}/{exp.name}"):
                continue
            is_canon = not exp.name.endswith("_ont")
            convs_dir = exp / "conversations"
            if not convs_dir.is_dir():
                continue
            for conv in sorted(convs_dir.iterdir()):
                if conv.is_dir():
                    yield triple, exp, conv, is_canon


def parse_conv_name(name: str) -> tuple[str, str]:
    """Parse '<target_id_slug>_runNN' → (target_id_slug, run_index)."""
    m = re.match(r"^(.+)_run(\d+)$", name)
    if m:
        return m.group(1), m.group(2).lstrip("0") or "0"
    return name, ""


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=None, help="Project root (default: repo root).")
    p.add_argument("--csv", type=Path, default=None,
                   help="Output CSV (default: <root>/outputs/ont_detection.csv).")
    p.add_argument("--oracle-only", type=str, default=None,
                   help="Only flag is_ont when oracle_model matches this exact name "
                        "(else mark as not_applicable). Recommended: 'Qwen3-8B'.")
    p.add_argument("--filter", type=str, default=None,
                   help="Regex to limit which triple/exp paths are scanned.")
    p.add_argument("-v", "--verbose", action="store_true", help="Print per-conversation progress.")
    p.add_argument("--workers", type=int, default=16,
                   help="Thread pool size for per-conversation IO (default: 16).")
    args = p.parse_args()

    if args.root is None:
        args.root = Path(__file__).resolve().parents[2]
    out_root = args.root / "outputs" / "models"
    out_csv = args.csv or (args.root / "outputs" / "ont_detection.csv")
    pat = re.compile(args.filter) if args.filter else None

    print(f"=== Detect ONT runs ===")
    print(f"Root: {args.root}")
    print(f"Filter: {pat.pattern if pat else '(none)'}")
    print(f"Oracle filter: {args.oracle_only or '(any oracle)'}")
    print(f"Output: {out_csv}\n")

    def classify(triple, exp, conv, is_canon):
        # Fast path when --oracle-only is set: derive oracle from the triple
        # dirname instead of opening metadata.json. The triple slug is built
        # by BenchmarkRunner._safe_name from the served-model-name, so it
        # round-trips for the canonical names we filter on.
        if args.oracle_only:
            oracle_model = oracle_from_triple_name(triple.name) or "?"
        else:
            oracle_model = oracle_model_from_metadata(conv / "metadata.json") or "?"

        if args.oracle_only and oracle_model != args.oracle_only:
            label = "not_applicable"
            reason = f"oracle is {oracle_model}, not {args.oracle_only}"
        else:
            status, reason = first_assistant_thinking(conv / "oracle.json")
            label = {
                "present": "clean",
                "absent": "ont",
                "parse_error": "parse_error",
                "no_history": "no_history",
            }[status]

        target_slug, run_idx = parse_conv_name(conv.name)
        return {
            "triple": triple.name,
            "exp": exp.name,
            "conv_dir": conv.name,
            "target_id_slug": target_slug,
            "run_index": run_idx,
            "oracle_model": oracle_model,
            "is_canonical_dir": is_canon,
            "label": label,
            "is_ont": label == "ont",
            "reason": reason,
        }

    convs = list(walk_conversations(out_root, pat))
    rows = []
    counts = {"clean": 0, "ont": 0, "not_applicable": 0, "parse_error": 0, "no_history": 0}

    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for r in pool.map(lambda args_: classify(*args_), convs):
                rows.append(r)
                counts[r["label"]] = counts.get(r["label"], 0) + 1
                if args.verbose:
                    print(f"  [{r['label']:<14}] {r['triple']}/{r['exp']}/{r['conv_dir']}  oracle={r['oracle_model']}")
    else:
        for triple, exp, conv, is_canon in convs:
            r = classify(triple, exp, conv, is_canon)
            rows.append(r)
            counts[r["label"]] = counts.get(r["label"], 0) + 1
            if args.verbose:
                print(f"  [{r['label']:<14}] {r['triple']}/{r['exp']}/{r['conv_dir']}  oracle={r['oracle_model']}")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    print(f"=== Done — {len(rows)} conversations scanned ===")
    for k, v in counts.items():
        print(f"  {k:<15} {v:>5}")
    print(f"\nWrote {len(rows)} rows -> {out_csv}")

    by_exp = defaultdict(lambda: {"clean": 0, "ont": 0, "not_applicable": 0, "other": 0})
    for r in rows:
        bucket = r["label"] if r["label"] in ("clean", "ont", "not_applicable") else "other"
        by_exp[(r["triple"], r["exp"])][bucket] += 1

    print("\n=== By experiment (only those with any 'ont' flag) ===")
    print(f"{'triple/exp':<90} {'clean':>6} {'ont':>5} {'N/A':>5} {'other':>5}")
    print("-" * 120)
    flagged = [(k, v) for k, v in by_exp.items() if v["ont"] > 0]
    for (triple, exp), stats in sorted(flagged):
        path = f"{triple}/{exp}"
        print(f"{path:<90} {stats['clean']:>6} {stats['ont']:>5} "
              f"{stats['not_applicable']:>5} {stats['other']:>5}")
    print(f"\n{len(flagged)} experiments contain at least one ont conversation.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
