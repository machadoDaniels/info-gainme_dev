#!/usr/bin/env python3
"""Merge ``<exp>_ont/`` data into the canonical ``<exp>/`` directory.

For each ``(target_id, run_index)`` pair present in the ``_ont`` runs.csv
that is **not yet** in the canonical runs.csv, this script:

  1. Copies the conversation directory (seeker.json, oracle.json, pruner.json,
     turns.jsonl, metadata.json, etc.) from ``_ont/conversations/<conv>/`` to
     ``<exp>/conversations/<conv>/``.
  2. Appends the corresponding row to ``<exp>/runs.csv`` (creating the file
     with header if needed).

The ``_ont`` directory itself is **never modified** — it remains as audit
trail. When the same ``(target_id, run_index)`` exists in both, canonical
wins (it's assumed to be the post-fix re-run with proper oracle thinking).

After merge, the resumable benchmark runner will see those targets as
"complete" and skip them on subsequent submissions. To re-run a contaminated
subset cleanly, use ``detect_ont_runs.py`` to identify the rows then delete
them via ``maintenance/delete_affected_runs.py`` before re-submitting.

Usage:
    # Default: dry-run (no changes), prints what would be merged
    python3 scripts/maintenance/merge_ont_into_canonical.py

    # Apply for real
    python3 scripts/maintenance/merge_ont_into_canonical.py --apply

    # Limit to a regex of triple/exp paths
    python3 scripts/maintenance/merge_ont_into_canonical.py --filter 'olmo3-32b' --apply

Safety:
    Refuses to overwrite an existing conversation dir in canonical (would
    indicate the same conv name already exists, which shouldn't happen if
    the dedup logic is correct — but the check protects against config
    bugs). If you need to override, pass ``--force-conv-overwrite``.

    Refuses to merge when canonical and _ont runs.csv have different column
    sets — that would silently drop data. Resolve by aligning headers (e.g.,
    re-run ``analyze_results`` to regenerate canonical) before merging.

    No file-locking: do **not** run while a benchmark is actively writing to
    the same canonical runs.csv. The intra-process lock in ``BenchmarkRunner``
    won't protect against this script's concurrent appends.
"""
from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
from pathlib import Path
from typing import Iterable


def find_ont_dirs(out_root: Path, pattern: re.Pattern | None = None) -> Iterable[Path]:
    for p in sorted(out_root.glob("*/*_ont")):
        if not p.is_dir():
            continue
        if pattern and not pattern.search(str(p.relative_to(out_root))):
            continue
        yield p


def load_runs_csv(p: Path) -> tuple[list[str], list[dict]]:
    """Return (header, rows). Empty if file doesn't exist."""
    if not p.exists():
        return [], []
    with open(p, newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        rows = list(reader)
    return header, rows


def keys_in(rows: list[dict]) -> set[tuple[str, str]]:
    out = set()
    for r in rows:
        tid = r.get("target_id") or r.get("target")
        ri = r.get("run_index") or r.get("run")
        if tid is not None and ri is not None:
            out.add((str(tid), str(ri)))
    return out


def conv_dirname(row: dict) -> str | None:
    """Extract the conversation dir name from a runs.csv row."""
    cp = row.get("conversation_path") or row.get("conv_path") or ""
    if cp:
        # path may be absolute, relative to project, or just the basename
        return Path(cp).name
    # fallback: build it from target_id + run_index — best-effort
    tid = row.get("target_id", "")
    ri = row.get("run_index", "")
    if tid and ri:
        # mirror BenchmarkRunner: <target_id_slug>_run<NN>
        # target_id like "disease:abscess_of_nose:0" → "disease-abscess_of_nose-0"
        slug = tid.replace(":", "-").replace("/", "-")
        return f"{slug}_run{int(ri):02d}"
    return None


def append_row(canonical_csv: Path, header: list[str], row: dict) -> None:
    """Append a row to canonical_csv, writing header if file is new."""
    write_header = not canonical_csv.exists()
    canonical_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(canonical_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if write_header:
            writer.writeheader()
        # Drop keys not in header to avoid ValueError
        clean = {k: row.get(k, "") for k in header}
        writer.writerow(clean)


def merge_one(ont_dir: Path, *, apply: bool, force_conv_overwrite: bool) -> dict:
    """Merge one <exp>_ont/ into its canonical sibling. Returns stats dict."""
    triple_dir = ont_dir.parent
    exp_name = ont_dir.name[:-len("_ont")]
    canon_dir = triple_dir / exp_name
    rel_label = f"{triple_dir.name}/{exp_name}"

    stats = {
        "exp": rel_label,
        "ont_rows": 0,
        "canon_rows_before": 0,
        "to_copy": 0,
        "skipped_already_canon": 0,
        "skipped_no_conv_dir": 0,
        "errors": [],
    }

    ont_csv = ont_dir / "runs.csv"
    canon_csv = canon_dir / "runs.csv"

    ont_header, ont_rows = load_runs_csv(ont_csv)
    canon_header, canon_rows = load_runs_csv(canon_csv)
    stats["ont_rows"] = len(ont_rows)
    stats["canon_rows_before"] = len(canon_rows)

    if not ont_rows:
        stats["errors"].append("ont runs.csv empty or missing")
        return stats

    canon_keys = keys_in(canon_rows)
    # Header strategy: when canonical doesn't exist yet, inherit _ont's header (full
    # column set). When canonical exists with a different shape than _ont, refuse
    # to merge — silently dropping columns would corrupt downstream analyses.
    if canon_header and ont_header and set(canon_header) != set(ont_header):
        only_in_ont = set(ont_header) - set(canon_header)
        only_in_canon = set(canon_header) - set(ont_header)
        stats["errors"].append(
            f"header mismatch — only in _ont: {sorted(only_in_ont)}, "
            f"only in canonical: {sorted(only_in_canon)}; refusing to merge"
        )
        return stats
    header = canon_header or ont_header

    plans: list[tuple[Path, Path, dict]] = []  # (src, dst, row)
    for row in ont_rows:
        tid = row.get("target_id") or row.get("target")
        ri = row.get("run_index") or row.get("run")
        if tid is None or ri is None:
            continue
        if (str(tid), str(ri)) in canon_keys:
            stats["skipped_already_canon"] += 1
            continue
        cn = conv_dirname(row)
        if not cn:
            stats["skipped_no_conv_dir"] += 1
            continue
        src = ont_dir / "conversations" / cn
        dst = canon_dir / "conversations" / cn
        if not src.exists():
            stats["skipped_no_conv_dir"] += 1
            stats["errors"].append(f"missing source conv: {src.relative_to(ont_dir.parents[2])}")
            continue
        if dst.exists() and not force_conv_overwrite:
            stats["errors"].append(f"canonical conv already exists, skipping: {cn}")
            continue
        plans.append((src, dst, row))

    stats["to_copy"] = len(plans)

    if apply and plans:
        for src, dst, row in plans:
            if dst.exists() and force_conv_overwrite:
                shutil.rmtree(dst)
            dst.parent.mkdir(parents=True, exist_ok=True)
            # Atomic-ish: copy conv first, then append row. If append fails, roll
            # back the conv copy to avoid orphan dirs (CSV row is the source of
            # truth for downstream scripts).
            shutil.copytree(src, dst)
            try:
                append_row(canon_csv, header, row)
            except Exception:
                shutil.rmtree(dst, ignore_errors=True)
                raise

    return stats


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=None, help="Project root (default: repo root).")
    p.add_argument("--apply", action="store_true", help="Actually perform the merge (default: dry-run).")
    p.add_argument("--filter", type=str, default=None, help="Regex to limit which exps to process.")
    p.add_argument("--force-conv-overwrite", action="store_true",
                   help="If a canonical conv dir already exists, overwrite it (use with care).")
    args = p.parse_args()

    if args.root is None:
        args.root = Path(__file__).resolve().parents[2]

    out_root = args.root / "outputs" / "models"
    pat = re.compile(args.filter) if args.filter else None

    print(f"=== Merge _ont → canonical {'(APPLY)' if args.apply else '(DRY-RUN)'} ===")
    print(f"Root: {args.root}")
    print(f"Filter: {pat.pattern if pat else '(none)'}\n")

    ont_dirs = list(find_ont_dirs(out_root, pat))
    if not ont_dirs:
        print("No _ont dirs found. Nothing to do.")
        return 0

    summary = []
    for d in ont_dirs:
        s = merge_one(d, apply=args.apply, force_conv_overwrite=args.force_conv_overwrite)
        summary.append(s)
        flag = "[+]" if s["to_copy"] > 0 else "[ ]"
        print(f"  {flag} {s['exp']:<70} ont={s['ont_rows']:>4}  "
              f"canon_before={s['canon_rows_before']:>4}  to_copy={s['to_copy']:>4}  "
              f"skipped(already)={s['skipped_already_canon']:>4}  "
              f"skipped(no_conv)={s['skipped_no_conv_dir']:>3}")
        for err in s["errors"]:
            print(f"      WARN: {err}")

    total_to_copy = sum(s["to_copy"] for s in summary)
    print(f"\nTotal: {len(summary)} _ont dirs, {total_to_copy} runs would be merged"
          f"{' (no changes made)' if not args.apply else ' — DONE'}.")
    if not args.apply and total_to_copy > 0:
        print("Re-run with --apply to execute.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
