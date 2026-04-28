#!/usr/bin/env python3
"""Audit benchmark configs vs runs.csv on disk.

Walks ``configs/full/**/*.yaml``, computes ``expected = num_targets × runs_per_target``
from each YAML, then counts unique ``(target_id, run_index)`` pairs in the
corresponding ``outputs/models/<triple>/<exp>/runs.csv``.

Outputs:
    - prints a per-folder summary to stdout
    - writes a per-config CSV (default: ``outputs/configs_progress.csv``)

Usage:
    python3 scripts/audit_experiments.py                     # default project dir + outputs/configs_progress.csv
    python3 scripts/audit_experiments.py --root /custom/dir
    python3 scripts/audit_experiments.py --csv /tmp/foo.csv
    python3 scripts/audit_experiments.py --no-csv            # only print summary

The script never modifies any data — it only reads YAMLs and CSVs.
"""
from __future__ import annotations

import argparse
import csv
import functools
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml


def slugify(name: str) -> str:
    """Match ``BenchmarkRunner._safe_name`` exactly (src/benchmark.py)."""
    return (
        str(name)
        .replace("/", "-")
        .replace("\\", "-")
        .replace(":", "-")
        .replace(" ", "_")
    )


def get_path(d, *keys, default=None):
    for k in keys:
        if isinstance(d, dict) and k in d:
            d = d[k]
        else:
            return default
    return d


def find_runs_csv(cfg_data, out_root: Path):
    seeker = get_path(cfg_data, "models", "seeker", "model", default="?")
    oracle = get_path(cfg_data, "models", "oracle", "model", default="?")
    pruner = get_path(cfg_data, "models", "pruner", "model", default="?")
    triple = f"s_{slugify(seeker)}__o_{slugify(oracle)}__p_{slugify(pruner)}"
    exp = get_path(cfg_data, "experiment", "name")
    runs_csv = out_root / triple / exp / "runs.csv" if exp else None
    return seeker, oracle, pruner, triple, exp, runs_csv


def count_unique_runs(p: Path) -> int:
    """Count unique (target_id, run_index) pairs in a runs.csv."""
    if not p or not p.exists():
        return 0
    seen = set()
    with open(p, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tid = row.get("target_id") or row.get("target")
            ri = row.get("run_index") or row.get("run")
            if tid is None or ri is None:
                continue
            seen.add((tid, ri))
    return len(seen)


@functools.lru_cache(maxsize=None)
def _dataset_row_count(p: Path) -> int | None:
    if not p.exists():
        return None
    with open(p) as f:
        return sum(1 for _ in f) - 1


def expected_runs(cfg_data, root: Path):
    rpt = get_path(cfg_data, "dataset", "runs_per_target", default=1) or 1
    csv_path = get_path(cfg_data, "dataset", "csv_path")
    num_targets = get_path(cfg_data, "dataset", "num_targets")
    if not csv_path:
        return None
    p = root / csv_path if not os.path.isabs(csv_path) else Path(csv_path)
    n = _dataset_row_count(p)
    if n is None:
        return None
    if num_targets:
        n = min(n, num_targets)
    return n * rpt


def audit(root: Path, include_ont: bool = False):
    """Audit configs vs runs.csv on disk.

    When ``include_ont`` is True, also looks for ``<exp>_ont/runs.csv`` (data
    from the oracle-no-thinking era preserved by ``rename_ont_experiments.sh``)
    and reports both ``actual`` (canonical) and ``ont_actual`` columns. The
    ``status`` / ``pct`` columns then reflect the **best of canonical vs ont**
    (effectively counting _ont data as valid).
    """
    cfg_root = root / "configs" / "full"
    out_root = root / "outputs" / "models"
    rows = []
    for yml in sorted(cfg_root.rglob("*.yaml")):
        rel = str(yml.relative_to(cfg_root))
        folder = rel.split("/")[0]
        is_ablation = any("ablation" in part for part in yml.parts)
        try:
            with open(yml) as f:
                data = yaml.safe_load(f)
        except Exception as e:
            row = {
                "folder": folder, "config": rel, "status": "ERROR",
                "actual": 0, "expected": 0, "pct": 0.0,
                "seeker": "?", "oracle": "?", "pruner": "?",
                "exp_name": str(e), "is_ablation": is_ablation, "runs_csv": "",
            }
            if include_ont:
                row.update({"ont_actual": 0, "ont_runs_csv": "", "best_actual": 0})
            rows.append(row)
            continue
        seeker, oracle, pruner, triple, exp_name, runs_csv = find_runs_csv(data, out_root)
        actual = count_unique_runs(runs_csv) if runs_csv else 0
        exp_total = expected_runs(data, root) or 0

        ont_actual = 0
        ont_runs_csv = None
        if include_ont and exp_name:
            ont_csv = out_root / triple / f"{exp_name}_ont" / "runs.csv"
            if ont_csv.exists():
                ont_actual = count_unique_runs(ont_csv)
                ont_runs_csv = ont_csv

        effective = max(actual, ont_actual) if include_ont else actual

        if exp_total == 0:
            status, pct = "?", 0.0
        elif effective == 0:
            status, pct = "MISSING", 0.0
        elif effective >= exp_total:
            status, pct = "DONE", 100.0
        else:
            status, pct = "PARTIAL", round(100.0 * effective / exp_total, 1)

        row = {
            "folder": folder,
            "config": rel,
            "status": status,
            "actual": actual,
            "expected": exp_total,
            "pct": pct,
            "seeker": seeker,
            "oracle": oracle,
            "pruner": pruner,
            "exp_name": exp_name or "?",
            "is_ablation": is_ablation,
            "runs_csv": str(runs_csv.relative_to(root)) if runs_csv else "",
        }
        if include_ont:
            row["ont_actual"] = ont_actual
            row["ont_runs_csv"] = str(ont_runs_csv.relative_to(root)) if ont_runs_csv else ""
            row["best_actual"] = effective
        rows.append(row)
    return rows


def print_summary(rows, include_ont: bool = False):
    by_folder = defaultdict(lambda: {"DONE": 0, "PARTIAL": 0, "MISSING": 0, "?": 0,
                                     "ERROR": 0, "total": 0, "actual": 0, "expected": 0,
                                     "ont": 0})
    for r in rows:
        s = by_folder[r["folder"]]
        s[r["status"]] += 1
        s["total"] += 1
        # When including _ont, sum the effective (best) count for the rows-progress column.
        s["actual"] += r.get("best_actual", r["actual"]) if include_ont else r["actual"]
        s["expected"] += r["expected"]
        if include_ont:
            s["ont"] += r.get("ont_actual", 0)

    if include_ont:
        print(f"{'folder':<28} {'DONE':>5} {'PART':>5} {'MISS':>5} {'tot':>5}  "
              f"{'best_rows':>13}  {'pct':>6}  {'_ont rows':>10}")
    else:
        print(f"{'folder':<28} {'DONE':>5} {'PART':>5} {'MISS':>5} {'tot':>5}  "
              f"{'rows':>13}  {'pct':>6}")
    print("-" * (90 if include_ont else 80))
    for f in sorted(by_folder):
        s = by_folder[f]
        pct = 100.0 * s["actual"] / s["expected"] if s["expected"] else 0.0
        line = (f"{f:<28} {s['DONE']:>5} {s['PARTIAL']:>5} {s['MISSING']:>5} "
                f"{s['total']:>5}  {s['actual']:>5}/{s['expected']:<6} {pct:>5.1f}%")
        if include_ont:
            line += f"  {s['ont']:>10}"
        print(line)
    print("-" * (90 if include_ont else 80))

    overall = Counter(r["status"] for r in rows)
    main = Counter(r["status"] for r in rows if not r["is_ablation"])
    print(f"\nALL configs        ({len(rows):>3}): {dict(overall)}")
    print(f"Excluding ablation ({sum(1 for r in rows if not r['is_ablation']):>3}): {dict(main)}")


def write_csv(rows, path: Path):
    if not rows:
        print(f"no rows to write — {path} skipped")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=None,
                   help="Project root (defaults to repo root, computed from this script's location).")
    p.add_argument("--csv", type=Path, default=None,
                   help="Output CSV path (default: <root>/outputs/configs_progress.csv).")
    p.add_argument("--no-csv", action="store_true", help="Skip CSV writing — only print summary.")
    p.add_argument("--include-ont", action="store_true",
                   help="Treat <exp>_ont/runs.csv as valid fallback data (status uses max of canonical vs ont).")
    args = p.parse_args()

    if args.root is None:
        args.root = Path(__file__).resolve().parent.parent

    rows = audit(args.root, include_ont=args.include_ont)
    print_summary(rows, include_ont=args.include_ont)
    if not args.no_csv:
        out_csv = args.csv or (
            args.root / "outputs"
            / ("configs_progress_with_ont.csv" if args.include_ont else "configs_progress.csv")
        )
        write_csv(rows, out_csv)


if __name__ == "__main__":
    sys.exit(main())
