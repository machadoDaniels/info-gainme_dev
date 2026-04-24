#!/usr/bin/env python3
"""Delete runs flagged by oracle validation so they can be re-executed.

Reads ``outputs/oracle_validation.csv`` and, for every affected
``conversation_dir``:

  1. Reads ``metadata.json`` to get ``target_id`` and ``run_index``
  2. Removes the matching row from the experiment's ``runs.csv``
     (otherwise ``BenchmarkRunner`` would skip the run as "completed")
  3. Deletes the conversation directory
  4. Removes stale ``summary.json`` / ``variance.json`` from affected
     experiments so analysis is regenerated afterwards

Defaults are conservative: ``--dry-run`` prints what would happen without
touching anything.

Usage:
  python3 scripts/delete_affected_runs.py --dry-run                        # preview
  python3 scripts/delete_affected_runs.py                                  # deletes errors + warnings
  python3 scripts/delete_affected_runs.py --errors-only                    # skip warning-only
  python3 scripts/delete_affected_runs.py --csv path/to/oracle_validation.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path


RUN_SUFFIX_RE = re.compile(r"_run(\d+)$")


def load_affected(csv_path: Path, errors_only: bool) -> dict[str, set[str]]:
    """Return {experiment_dir: {conversation_dir, ...}}."""
    by_experiment: dict[str, set[str]] = defaultdict(set)
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            if errors_only and row["severity"] != "error":
                continue
            by_experiment[row["experiment_dir"]].add(row["conversation_dir"])
    return by_experiment


def read_conversation_key(conv_dir: Path) -> tuple[str, int] | None:
    """Return (target_id, run_index).

    target_id comes from metadata.json (``target.id``). run_index is parsed
    from the directory name suffix ``_run<NN>``.
    """
    # run_index from folder name
    m = RUN_SUFFIX_RE.search(conv_dir.name)
    if not m:
        return None
    run_index = int(m.group(1))

    # target_id from metadata.json
    metadata_path = conv_dir / "metadata.json"
    if not metadata_path.exists():
        return None
    try:
        meta = json.loads(metadata_path.read_text())
    except Exception:
        return None
    target_id = meta.get("target", {}).get("id") or meta.get("target_id")
    if target_id is None:
        return None
    return str(target_id), run_index


def rewrite_runs_csv(
    runs_csv_path: Path,
    keys_to_drop: set[tuple[str, int]],
    dry_run: bool,
) -> tuple[int, int]:
    """Drop rows whose (target_id, run_index) is in keys_to_drop.

    Returns (removed_count, kept_count).
    """
    if not runs_csv_path.exists():
        return 0, 0

    with runs_csv_path.open() as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        all_rows = list(reader)

    removed = 0
    kept_rows = []
    for row in all_rows:
        key = (row.get("target_id", ""), int(row.get("run_index", -1)))
        if key in keys_to_drop:
            removed += 1
        else:
            kept_rows.append(row)

    if not dry_run and removed > 0:
        tmp = runs_csv_path.with_suffix(".csv.tmp")
        with tmp.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(kept_rows)
        tmp.replace(runs_csv_path)

    return removed, len(kept_rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv", type=Path, default=Path("outputs/oracle_validation.csv"),
                    help="Path to oracle_validation.csv")
    ap.add_argument("--errors-only", action="store_true",
                    help="Only delete runs with severity=error (skip warning-only)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print what would be deleted without touching files")
    ap.add_argument("--keep-stale-summaries", action="store_true",
                    help="Do not delete summary.json / variance.json from affected experiments")
    args = ap.parse_args()

    if not args.csv.exists():
        print(f"error: {args.csv} not found")
        return 1

    affected = load_affected(args.csv, args.errors_only)
    if not affected:
        print("No affected runs found.")
        return 0

    total_convs = sum(len(v) for v in affected.values())
    total_exps = len(affected)
    mode = "DRY-RUN" if args.dry_run else "DELETE"
    scope = "errors only" if args.errors_only else "errors + warnings"

    print(f"[{mode}] {total_convs} conversas em {total_exps} experimentos "
          f"({scope})")
    print("=" * 70)

    # Count total runs across ALL experiments (for global %), not just affected ones.
    outputs_root = Path("outputs")
    global_total_rows = 0
    for csv_file in outputs_root.rglob("runs.csv"):
        try:
            with csv_file.open() as f:
                global_total_rows += sum(1 for _ in csv.DictReader(f))
        except Exception:
            pass

    totals = {
        "convs_deleted": 0,
        "rows_removed": 0,
        "rows_missing": 0,
        "rows_total_before": 0,
        "metadata_missing": 0,
        "summaries_removed": 0,
    }

    for exp_dir, convs in sorted(affected.items()):
        exp_path = Path(exp_dir)
        runs_csv = exp_path / "runs.csv"

        # Collect (target_id, run_index) keys from metadata files
        keys_to_drop: set[tuple[str, int]] = set()
        valid_convs: list[Path] = []
        for conv in sorted(convs):
            conv_path = Path(conv)
            key = read_conversation_key(conv_path)
            if key is None:
                totals["metadata_missing"] += 1
                print(f"  ⚠  no metadata: {conv_path}")
                continue
            keys_to_drop.add(key)
            valid_convs.append(conv_path)

        removed, kept = rewrite_runs_csv(runs_csv, keys_to_drop, args.dry_run)
        missing_in_csv = len(keys_to_drop) - removed
        totals["rows_removed"] += removed
        totals["rows_missing"] += missing_in_csv
        total_before = removed + kept
        totals["rows_total_before"] += total_before

        # Delete conversation dirs
        for conv_path in valid_convs:
            if conv_path.exists():
                if not args.dry_run:
                    shutil.rmtree(conv_path)
                totals["convs_deleted"] += 1

        # Remove stale summaries
        if not args.keep_stale_summaries:
            for name in ("summary.json", "variance.json", "question_evaluations_summary.json"):
                stale = exp_path / name
                if stale.exists():
                    if not args.dry_run:
                        stale.unlink()
                    totals["summaries_removed"] += 1

        pct = (removed / total_before * 100) if total_before else 0.0
        print(f"  {exp_path.name}: {removed} rows removidas "
              f"({pct:5.1f}% de {total_before}), "
              f"{len(valid_convs)} pastas {'seriam ' if args.dry_run else ''}apagadas "
              f"({kept} runs restantes)")

    print("=" * 70)
    pct_affected_exp = (
        totals["rows_removed"] / totals["rows_total_before"] * 100
        if totals["rows_total_before"] else 0.0
    )
    pct_global = (
        totals["rows_removed"] / global_total_rows * 100
        if global_total_rows else 0.0
    )
    action = "que seriam " if args.dry_run else ""
    print(f"Conversas {action}deletadas:                 {totals['convs_deleted']}")
    print(f"Linhas {action}removidas de runs.csv:        {totals['rows_removed']}")
    print(f"  % dentro dos experimentos afetados:     {pct_affected_exp:.2f}% "
          f"({totals['rows_removed']}/{totals['rows_total_before']})")
    print(f"  % do total de runs (todos experimentos): {pct_global:.2f}% "
          f"({totals['rows_removed']}/{global_total_rows})")
    print(f"Linhas já ausentes em runs.csv:           {totals['rows_missing']}")
    print(f"Metadata ausentes (ignoradas):            {totals['metadata_missing']}")
    print(f"Summaries stale {action}removidos:           {totals['summaries_removed']}")

    if args.dry_run:
        print("\nNada foi alterado. Rode sem --dry-run para aplicar.")
    else:
        print("\n✅ Feito. Re-submeta os jobs e o BenchmarkRunner vai rodar só os faltantes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
