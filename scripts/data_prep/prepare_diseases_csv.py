#!/usr/bin/env python3
"""Preprocess Final_Augmented_dataset_Diseases_and_Symptoms.csv into derived CSVs.

Reads the original binary symptom matrix, groups by disease, computes union of
symptoms per disease, and outputs:
- data/diseases/diseases_full.csv (all 773 diseases)
- data/diseases/diseases_test.csv (~40 diseases for quick tests)

Usage:
    python scripts/prepare_diseases_csv.py
"""

import csv
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent.parent
    src_path = project_root / "data/diseases/Final_Augmented_dataset_Diseases_and_Symptoms.csv"
    out_dir = project_root / "data/diseases"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not src_path.exists():
        print(f"Source file not found: {src_path}")
        return

    print(f"Reading {src_path}...")
    import pandas as pd
    df = pd.read_csv(src_path)

    disease_col = "diseases"
    symptom_cols = [c for c in df.columns if c != disease_col]

    # Group by disease, union of symptoms
    disease_to_symptoms: dict[str, set[str]] = {}
    for _, row in df.iterrows():
        disease = str(row[disease_col]).strip()
        if not disease:
            continue
        if disease not in disease_to_symptoms:
            disease_to_symptoms[disease] = set()
        for col in symptom_cols:
            if row[col] == 1:
                disease_to_symptoms[disease].add(col)

    # Build rows: disease, symptoms (semicolon-separated), aliases (empty)
    rows_full = []
    for disease in sorted(disease_to_symptoms.keys()):
        symptoms = sorted(disease_to_symptoms[disease])
        symptoms_str = ";".join(symptoms)
        rows_full.append({"disease": disease, "symptoms": symptoms_str, "aliases": ""})

    # Write full
    full_path = out_dir / "diseases_full.csv"
    with full_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["disease", "symptoms", "aliases"])
        w.writeheader()
        w.writerows(rows_full)
    print(f"Wrote {full_path} ({len(rows_full)} diseases)")

    # Write test (first ~40 diseases, diverse)
    test_count = 40
    rows_test = rows_full[:test_count]
    test_path = out_dir / "diseases_test.csv"
    with test_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["disease", "symptoms", "aliases"])
        w.writeheader()
        w.writerows(rows_test)
    print(f"Wrote {test_path} ({len(rows_test)} diseases)")


if __name__ == "__main__":
    main()
