#!/usr/bin/env python3
"""Aggregate the flat question-classification CSV into analysis artifacts.

Reads the CSV produced by ``flatten_question_classifications.py`` plus all
``outputs/models/**/runs.csv`` files, then writes a handful of CSVs designed
for plotting and reporting in a notebook:

    by_model.csv              seeker × question_type counts + fractions
    by_model_turn.csv         seeker × turn × question_type (for the
                               "binary-search curve" plot)
    redundancy_by_model.csv   seeker × redundancy counts + rates
    subclass_by_model.csv     seeker × subclass counts (top subclasses only)
    by_model_summary.csv      headline table per seeker (the paper's Table 1):
                                 malformed_rate, redundancy_rate,
                                 hierarchical_rate, fine_grained_rate,
                                 direct_guess_rate, win_rate, mean_turns,
                                 mean_info_gain, mean_ig_per_turn
    joined_with_runs.csv      flat turn-level CSV joined with per-game metrics
                               (win, total_info_gain, turns, compliance_rate)

Run order:

    python3 scripts/flatten_question_classifications.py
    python3 scripts/analyze_question_classifications.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_flat(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype={"run_index": "Int64", "redundant_with_turn": "Int64"})
    # Drop rows that couldn't be classified (non-empty error column).
    errored = df["error"].notna() & (df["error"].astype(str) != "") & (df["error"].astype(str) != "nan")
    if errored.any():
        print(f"  note: {int(errored.sum())} turn(s) had classification errors and are excluded from rates.")
    return df[~errored].copy()


def load_runs(outputs_root: Path) -> pd.DataFrame:
    """Concatenate every runs.csv under outputs/models/."""
    files = sorted((outputs_root / "models").glob("**/runs.csv"))
    if not files:
        print(f"  warning: no runs.csv under {outputs_root}/models — skipping join step.", file=sys.stderr)
        return pd.DataFrame()
    frames = []
    for f in files:
        try:
            frames.append(pd.read_csv(f))
        except Exception as e:  # noqa: BLE001
            print(f"  skip {f}: {e}", file=sys.stderr)
    if not frames:
        return pd.DataFrame()
    runs = pd.concat(frames, ignore_index=True)
    # Derive the conv folder name so we can join on (experiment, target, run_index).
    runs["target_folder"] = runs["conversation_path"].astype(str).apply(
        lambda p: Path(p).name
    )
    # Mirror the parser in flatten_question_classifications.py.
    runs["target_base"] = runs["target_folder"].str.replace(r"_run\d+$", "", regex=True)
    return runs


# ---------------------------------------------------------------------------
# Aggregations
# ---------------------------------------------------------------------------


QUESTION_TYPES = ["semantic", "lexical", "direct_guess", "malformed", "other"]
REDUNDANCY_TYPES = ["none", "exact_duplicate", "semantic_equivalent", "strictly_implied"]


def _crosstab_with_rates(df: pd.DataFrame, row: str, col: str, values: list[str]) -> pd.DataFrame:
    """Cross-tab of counts + fractions, with a stable column order."""
    ct = pd.crosstab(df[row], df[col])
    # Ensure every expected label appears, even with zero.
    for v in values:
        if v not in ct.columns:
            ct[v] = 0
    ct = ct[values]
    ct["total"] = ct.sum(axis=1)
    for v in values:
        ct[f"{v}_frac"] = ct[v] / ct["total"].where(ct["total"] > 0, 1)
    return ct.reset_index()


def by_model(df: pd.DataFrame) -> pd.DataFrame:
    return _crosstab_with_rates(df, "seeker", "question_type", QUESTION_TYPES)


def by_model_turn(df: pd.DataFrame, max_turn: int = 30) -> pd.DataFrame:
    """Long-format table: one row per (seeker, turn, question_type) with count and fraction.

    Easy to plot with seaborn/matplotlib: ``x=turn, y=frac, hue=question_type, col=seeker``.
    """
    capped = df[df["turn"] <= max_turn].copy()
    grp = capped.groupby(["seeker", "turn", "question_type"]).size().rename("count").reset_index()
    totals = capped.groupby(["seeker", "turn"]).size().rename("total").reset_index()
    merged = grp.merge(totals, on=["seeker", "turn"], how="left")
    merged["frac"] = merged["count"] / merged["total"]
    return merged


def redundancy_by_model(df: pd.DataFrame) -> pd.DataFrame:
    return _crosstab_with_rates(df, "seeker", "redundancy", REDUNDANCY_TYPES)


def subclass_by_model(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """Top-N subclass tags per seeker, long format."""
    sc = df[df["subclass"].notna() & (df["subclass"].astype(str) != "")].copy()
    if sc.empty:
        return pd.DataFrame(columns=["seeker", "subclass", "count", "rank"])
    counts = sc.groupby(["seeker", "subclass"]).size().rename("count").reset_index()
    counts["rank"] = counts.groupby("seeker")["count"].rank(ascending=False, method="dense").astype(int)
    return counts[counts["rank"] <= top_n].sort_values(["seeker", "rank"]).reset_index(drop=True)


def by_model_summary(df: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    """Headline table: one row per seeker model."""
    rows: list[dict] = []
    for seeker, g in df.groupby("seeker"):
        total = len(g)
        if total == 0:
            continue
        qt = g["question_type"].value_counts()
        red = g["redundancy"].value_counts()
        sc = g["subclass"].value_counts() if "subclass" in g else pd.Series(dtype=int)

        redundant_n = sum(red.get(k, 0) for k in ("exact_duplicate", "semantic_equivalent", "strictly_implied"))
        row = {
            "seeker": seeker,
            "n_turns": total,
            "n_games": g[["experiment", "target", "run_index"]].drop_duplicates().shape[0],
            "malformed_rate": qt.get("malformed", 0) / total,
            "redundancy_rate": redundant_n / total,
            "semantic_rate": qt.get("semantic", 0) / total,
            "lexical_rate": qt.get("lexical", 0) / total,
            "direct_guess_rate": qt.get("direct_guess", 0) / total,
            "hierarchical_rate": sc.get("hierarchical_category", 0) / total if not sc.empty else 0.0,
            "fine_grained_rate": sc.get("fine_grained_category", 0) / total if not sc.empty else 0.0,
        }
        rows.append(row)
    summary = pd.DataFrame(rows)

    if not runs.empty:
        # Folder slugs replace "/" with "-"; runs.csv keeps the original
        # HuggingFace-style "org/name". Normalise before the join.
        runs = runs.copy()
        runs["seeker_norm"] = runs["seeker_model"].astype(str).str.replace("/", "-", regex=False)
        summary["seeker_norm"] = summary["seeker"].astype(str).str.replace("/", "-", regex=False)
        runs_by_seeker = (
            runs.groupby("seeker_norm")
            .agg(
                n_games_runs=("win", "size"),
                win_rate=("win", "mean"),
                mean_turns=("turns", "mean"),
                mean_info_gain=("total_info_gain", "mean"),
                mean_ig_per_turn=("avg_info_gain_per_turn", "mean"),
                mean_compliance=("compliance_rate", "mean"),
            )
            .reset_index()
        )
        summary = summary.merge(runs_by_seeker, on="seeker_norm", how="left").drop(columns=["seeker_norm"])

    # Stable column order
    preferred = [
        "seeker",
        "n_turns",
        "n_games",
        "n_games_runs",
        "win_rate",
        "mean_turns",
        "mean_info_gain",
        "mean_ig_per_turn",
        "mean_compliance",
        "malformed_rate",
        "redundancy_rate",
        "hierarchical_rate",
        "fine_grained_rate",
        "direct_guess_rate",
        "semantic_rate",
        "lexical_rate",
    ]
    return summary[[c for c in preferred if c in summary.columns]].sort_values("seeker").reset_index(drop=True)


def joined_with_runs(df: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    if runs.empty:
        return df
    # Join on (experiment_name, target_folder [= target_base + _runNN]).
    left = df.copy()
    left["target_folder"] = left.apply(
        lambda r: f"{r['target']}_run{int(r['run_index']):02d}" if pd.notna(r["run_index"]) else r["target"],
        axis=1,
    )
    right = runs[
        [
            "experiment_name",
            "target_folder",
            "win",
            "turns",
            "total_info_gain",
            "avg_info_gain_per_turn",
            "compliance_rate",
            "observability",
        ]
    ].rename(columns={"experiment_name": "experiment"})
    return left.merge(right, on=["experiment", "target_folder"], how="left")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--flat-csv",
        type=Path,
        default=Path("outputs/question_classifications.csv"),
        help="Flat CSV produced by flatten_question_classifications.py.",
    )
    p.add_argument(
        "--outputs-root",
        type=Path,
        default=Path("outputs"),
        help="Root containing models/**/runs.csv for the join step.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/question_classifications_analysis"),
    )
    p.add_argument("--subclass-top-n", type=int, default=15)
    p.add_argument("--max-turn", type=int, default=30)
    args = p.parse_args()

    if not args.flat_csv.exists():
        print(
            f"ERROR: {args.flat_csv} not found. Run flatten_question_classifications.py first.",
            file=sys.stderr,
        )
        return 1

    print(f"Loading {args.flat_csv}...")
    df = load_flat(args.flat_csv)
    print(f"  {len(df)} turns × {df['seeker'].nunique()} seeker model(s)")

    print(f"Loading runs.csv files under {args.outputs_root}/models...")
    runs = load_runs(args.outputs_root)
    if not runs.empty:
        print(f"  {len(runs)} rows across {runs['seeker_model'].nunique()} seeker model(s)")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    tables: dict[str, pd.DataFrame] = {
        "by_model.csv": by_model(df),
        "by_model_turn.csv": by_model_turn(df, max_turn=args.max_turn),
        "redundancy_by_model.csv": redundancy_by_model(df),
        "subclass_by_model.csv": subclass_by_model(df, top_n=args.subclass_top_n),
        "by_model_summary.csv": by_model_summary(df, runs),
        "joined_with_runs.csv": joined_with_runs(df, runs),
    }

    for name, tbl in tables.items():
        out = args.out_dir / name
        tbl.to_csv(out, index=False)
        print(f"  wrote {out}  ({len(tbl)} rows)")

    # Print the headline summary inline — easiest to eyeball.
    summary = tables["by_model_summary.csv"]
    if not summary.empty:
        print("\n=== by_model_summary (headline) ===")
        with pd.option_context("display.max_rows", None, "display.width", 200, "display.max_columns", None):
            print(summary.round(3).to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
