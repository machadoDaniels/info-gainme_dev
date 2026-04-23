"""Aggregate cold-start test results into a single summary CSV + stdout table.

Reads all per-rep JSONs under outputs/cold_start/by_model/<model>/<condition>/rep_XX.json
and produces:
  - stdout: human-readable table (one row per model×condition)
  - outputs/cold_start/aggregate.csv: same data in CSV for plotting/sharing

Usage:
    python3 scripts/aggregate_cold_start.py
    python3 scripts/aggregate_cold_start.py --base outputs/cold_start
    python3 scripts/aggregate_cold_start.py --top 5     # show top N responses per cell
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


CONDITIONS = ("T1_system_only", "T2_with_kickoff", "T3_turn2_baseline")


def load_reps(cond_dir: Path) -> list[dict]:
    return [
        json.loads(p.read_text())
        for p in sorted(cond_dir.glob("rep_*.json"))
    ]


def summarize(reps: list[dict]) -> dict:
    if not reps:
        return {}
    labels = Counter(r.get("classification", "?") for r in reps)
    finals = [r.get("content_final", "") for r in reps]
    usages = [r.get("usage") or {} for r in reps]
    prompt = [u.get("prompt_tokens") for u in usages if u.get("prompt_tokens")]
    comp = [u.get("completion_tokens") for u in usages if u.get("completion_tokens")]
    return {
        "n": len(reps),
        "valid": labels.get("VALID", 0) + labels.get("LONG_QUESTION", 0),
        "fallback": labels.get("FALLBACK", 0),
        "refusal": labels.get("REFUSAL", 0),
        "broken": labels.get("FORMAT_BROKEN", 0) + labels.get("EMPTY", 0),
        "error": labels.get("ERROR", 0),
        "unique": len(set(finals)),
        "think_rate": sum(r.get("think_present", False) for r in reps) / len(reps),
        "prompt_avg": sum(prompt) / len(prompt) if prompt else 0,
        "comp_avg": sum(comp) / len(comp) if comp else 0,
        "top_responses": Counter(finals).most_common(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate cold-start test results")
    parser.add_argument("--base", type=Path, default=Path("outputs/cold_start"),
                        help="Base dir containing by_model/ (default: outputs/cold_start)")
    parser.add_argument("--top", type=int, default=3,
                        help="Show top N responses per model/condition (default: 3)")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output CSV path (default: <base>/aggregate.csv)")
    args = parser.parse_args()

    by_model_dir = args.base / "by_model"
    if not by_model_dir.is_dir():
        parser.error(f"no by_model/ dir under {args.base}")

    rows: list[dict] = []
    for model_dir in sorted(by_model_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        for cond in CONDITIONS:
            cond_dir = model_dir / cond
            if not cond_dir.is_dir():
                continue
            s = summarize(load_reps(cond_dir))
            if not s:
                continue
            rows.append({"model": model_dir.name, "condition": cond, **s})

    if not rows:
        print("No rep data found.")
        return

    # --- stdout table ---
    hdr = f"{'Model':<40} {'Cond':<20} {'VALID':>5} {'BRKN':>4} {'Uniq':>4} {'Think':>5} {'PrmpT':>5} {'CompT':>6}"
    print(hdr)
    print("-" * len(hdr))
    prev_model = None
    for r in rows:
        if prev_model and r["model"] != prev_model:
            print()
        prev_model = r["model"]
        print(f"{r['model']:<40} {r['condition']:<20} "
              f"{r['valid']:>5} {r['broken']:>4} {r['unique']:>4} "
              f"{r['think_rate']*100:>4.0f}% {r['prompt_avg']:>5.0f} {r['comp_avg']:>6.0f}")

    # --- top responses ---
    if args.top > 0:
        print("\n" + "=" * 60)
        print(f"TOP {args.top} responses per cell")
        print("=" * 60)
        prev_model = None
        for r in rows:
            if prev_model and r["model"] != prev_model:
                print()
            prev_model = r["model"]
            print(f"\n{r['model']} / {r['condition']}")
            for txt, cnt in r["top_responses"][: args.top]:
                short = txt.replace("\n", " ")[:90]
                marker = "★" if cnt >= 8 else " "
                print(f"  {marker} x{cnt:>2}: {short!r}")

    # --- CSV ---
    out = args.out or (args.base / "aggregate.csv")
    cols = ["model", "condition", "n", "valid", "fallback", "refusal", "broken",
            "error", "unique", "think_rate", "prompt_avg", "comp_avg"]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\nCSV: {out}")


if __name__ == "__main__":
    main()
