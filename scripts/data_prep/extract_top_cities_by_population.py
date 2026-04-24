#!/usr/bin/env python3
"""Extrai as N cidades mais populosas a partir de top_160_pop_cities.csv."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Gera top_N_pop_cities.csv ordenando por population_2025 (desc)."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=root / "data/geo/top_160_pop_cities.csv",
        help="CSV fonte (default: data/geo/top_160_pop_cities.csv)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Arquivo de saída (default: data/geo/top_N_pop_cities.csv com N de -n)",
    )
    parser.add_argument(
        "-n",
        type=int,
        default=20,
        help="Quantidade de cidades (default: 20)",
    )
    args = parser.parse_args()

    src = args.source
    if not src.is_file():
        raise SystemExit(f"Fonte não encontrada: {src}")

    out = args.out
    if out is None:
        out = root / "data/geo" / f"top_{args.n}_pop_cities.csv"

    with src.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if not fieldnames or "population_2025" not in fieldnames:
            raise SystemExit("CSV precisa da coluna population_2025")
        rows = list(reader)

    rows.sort(key=lambda r: int(r["population_2025"]), reverse=True)
    top = rows[: args.n]

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(top)

    print(f"Escrito {out} ({len(top)} linhas)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
