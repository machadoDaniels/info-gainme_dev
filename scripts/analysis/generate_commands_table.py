"""Gera tabela com progresso por grupo (seeker × CONFIGS_TARGET) e comando sbatch.

Usage:
    python scripts/analysis/generate_commands_table.py [configs_progress_run01.csv]
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Mapeamento seeker → (MODEL1_HF, gpu, partition, mode, extra_export)
# ---------------------------------------------------------------------------
BASE_CMD = "dgx/run_full_benchmark.sh"

MODEL_MAP: dict[str, dict] = {
    # Qwen3 ----------------------------------------------------------------
    "Qwen3-8B": dict(
        hf="Qwen/Qwen3-8B", gpu=1, partition="h100n2", mode="single",
    ),
    "Qwen3-0.6B": dict(
        hf="Qwen/Qwen3-0.6B", gpu=2, partition="h100n2", mode="dual",
    ),
    "Qwen3-4B-Thinking-2507": dict(
        hf="Qwen/Qwen3-4B-Thinking-2507", gpu=2, partition="h100n2", mode="dual",
    ),
    "Qwen3-4B-Instruct-2507": dict(
        hf="Qwen/Qwen3-4B-Instruct-2507", gpu=2, partition="h100n2", mode="dual",
    ),
    "Qwen3-30B-A3B-Thinking-2507": dict(
        hf="Qwen/Qwen3-30B-A3B-Thinking-2507", gpu=2, partition="h100n2", mode="dual",
        extra="VLLM_ENGINE_READY_TIMEOUT_S=3600",
    ),
    "Qwen3-30B-A3B-Instruct-2507": dict(
        hf="Qwen/Qwen3-30B-A3B-Instruct-2507", gpu=2, partition="h100n2", mode="dual",
        extra="VLLM_ENGINE_READY_TIMEOUT_S=3600",
    ),
    "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8": dict(
        hf="Qwen/Qwen3-235B-A22B-Instruct-2507-FP8", gpu=4, partition="b200n1", mode="dual",
        extra="VLLM_ENGINE_READY_TIMEOUT_S=3600",
        note="⚠️ precisa --tensor-parallel-size manual",
    ),
    # Llama ----------------------------------------------------------------
    "Llama-3.1-8B-Instruct": dict(
        hf="meta-llama/Llama-3.1-8B-Instruct", gpu=2, partition="h100n2", mode="dual",
    ),
    "paprika_Meta-Llama-3.1-8B-Instruct": dict(
        hf="ftajwar/paprika_Meta-Llama-3.1-8B-Instruct", gpu=2, partition="h100n2", mode="dual",
    ),
    # Nemotron -------------------------------------------------------------
    "Nemotron-Cascade-8B": dict(
        hf="nvidia/Nemotron-Cascade-8B", gpu=2, partition="h100n2", mode="dual",
    ),
    # Olmo -----------------------------------------------------------------
    "Olmo-3-7B-Think": dict(
        hf="allenai/Olmo-3-7B-Think", gpu=2, partition="h100n2", mode="dual",
    ),
    "Olmo-3-7B-Instruct": dict(
        hf="allenai/Olmo-3-7B-Instruct", gpu=2, partition="h100n2", mode="dual",
    ),
    "Olmo-3.1-32B-Think": dict(
        hf="allenai/Olmo-3.1-32B-Think", gpu=2, partition="b200n1", mode="dual",
        extra="VLLM_ENGINE_READY_TIMEOUT_S=3600",
    ),
    "Olmo-3.1-32B-Instruct": dict(
        hf="allenai/Olmo-3.1-32B-Instruct", gpu=2, partition="b200n1", mode="dual",
        extra="VLLM_ENGINE_READY_TIMEOUT_S=3600",
    ),
    # Phi ------------------------------------------------------------------
    "phi-4": dict(
        hf="microsoft/phi-4", gpu=2, partition="h100n2", mode="dual",
    ),
    "Phi-4-reasoning": dict(
        hf="microsoft/Phi-4-reasoning", gpu=2, partition="h100n2", mode="dual",
        extra="VLLM_ENGINE_READY_TIMEOUT_S=3600",
    ),
    "Phi-4-mini-instruct": dict(
        hf="microsoft/Phi-4-mini-instruct", gpu=2, partition="h100n2", mode="dual",
    ),
    "Phi-4-mini-reasoning": dict(
        hf="microsoft/Phi-4-mini-reasoning", gpu=2, partition="h100n2", mode="dual",
        extra="VLLM_ENGINE_READY_TIMEOUT_S=3600",
    ),
    # Gemma ----------------------------------------------------------------
    "google/gemma-4-31B-it": dict(
        hf="google/gemma-4-31B-it", gpu=2, partition="b200n1", mode="dual",
        extra="VLLM_ENGINE_READY_TIMEOUT_S=3600",
    ),
}

# Jobs com estado conhecido: (seeker, subfolder) → (state, job_id)
KNOWN_JOBS: dict[tuple[str, str], tuple[str, str]] = {
    ("Phi-4-reasoning",           "phi4/cot"):              ("RUNNING",  "19053"),
    ("Phi-4-mini-reasoning",      "phi4-mini/cot"):         ("RUNNING",  "19062"),
    ("Nemotron-Cascade-8B",       "nemotron-8b/fo"):        ("PENDING b200", "18942"),
    ("Qwen3-8B",                  "8b"):                    ("PENDING b200", "18946"),
    ("Qwen3-8B",                  "8b/with_prior"):         ("PENDING b200", "18947"),
    ("Qwen3-30B-A3B-Thinking-2507", "30b/with_prior/cot"): ("PENDING b200", "18948"),
    ("Olmo-3-7B-Think",           "olmo3-7b/cot"):          ("PENDING b200", "19000"),
    ("Olmo-3-7B-Instruct",        "olmo3-7b/no_cot"):       ("PENDING b200", "19001"),
    ("Llama-3.1-8B-Instruct",     "llama-3.1-8b/no_cot"):  ("PENDING b200", "19070"),
    ("Olmo-3.1-32B-Instruct",     "olmo3-32b/no_cot"):      ("PENDING b200", "18938"),
}


def build_command(seeker: str, configs_target: str) -> str:
    m = MODEL_MAP.get(seeker)
    if not m:
        return "⚠️ MODEL_MAP entry missing"

    parts = [
        f"MODEL1={m['hf']}",
        f"MODEL1_NAME={seeker}",
    ]
    if m["mode"] == "dual":
        parts += ["MODEL2=Qwen/Qwen3-8B", "MODEL2_NAME=Qwen3-8B"]
    parts.append(f"MODE={m['mode']}")
    if "extra" in m:
        parts.append(m["extra"])
    parts.append(f"CONFIGS_TARGET=configs/full/{configs_target}/")

    export_str = ",".join(["ALL"] + parts)
    return (
        f"sbatch --partition={m['partition']} --gres=gpu:{m['gpu']} "
        f"--export={export_str} {BASE_CMD}"
    )


def main() -> int:
    repo_root = Path(__file__).parent.parent.parent
    default_csv = repo_root / "outputs" / "configs_progress_run01.csv"
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_csv

    if not csv_path.exists():
        print(f"❌ CSV não encontrado: {csv_path}")
        return 1

    rows = list(csv.DictReader(open(csv_path)))

    # Agrupa por (seeker, subfolder)
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        if r["status"] == "DONE":
            continue
        parts = r["config"].split("/")
        subfolder = "/".join(parts[:-1]) if len(parts) > 1 else parts[0]
        groups[(r["seeker"], subfolder)].append(r)

    out_rows = []
    for (seeker, subfolder), configs in sorted(groups.items()):
        actual = sum(int(r["actual"]) for r in configs)
        expected = sum(int(r["expected"]) for r in configs if r["expected"].isdigit())
        pct = round(actual / expected, 3) if expected else 0.0
        n_miss = sum(1 for r in configs if r["status"] == "MISSING")
        n_part = sum(1 for r in configs if r["status"] == "PARTIAL")
        n_configs = len(configs)

        state, job_id = KNOWN_JOBS.get((seeker, subfolder), ("NEW", ""))
        cmd = build_command(seeker, subfolder)
        note = MODEL_MAP.get(seeker, {}).get("note", "")

        out_rows.append({
            "seeker":        seeker,
            "subfolder":     subfolder,
            "n_configs":     n_configs,
            "missing":       n_miss,
            "partial":       n_part,
            "actual":        actual,
            "expected":      expected,
            "pct":           f"{pct:.1%}",
            "state":         state,
            "job_id":        job_id,
            "note":          note,
            "command":       cmd,
        })

    out_path = repo_root / "outputs" / "configs_commands.csv"
    headers = ["seeker", "subfolder", "n_configs", "missing", "partial",
               "actual", "expected", "pct", "state", "job_id", "note", "command"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"✅ {len(out_rows)} grupos → {out_path}")
    print()

    # Preview
    col_w = {"seeker": 40, "subfolder": 30, "pct": 7, "state": 14, "job_id": 8}
    hdr = (f"{'seeker':<{col_w['seeker']}}  {'subfolder':<{col_w['subfolder']}}"
           f"  {'actual/exp':>12}  {'pct':>{col_w['pct']}}  "
           f"{'state':<{col_w['state']}}  {'job':>{col_w['job_id']}}")
    print(hdr)
    print("-" * len(hdr))
    for r in out_rows:
        prog = f"{r['actual']}/{r['expected']}"
        print(
            f"{r['seeker']:<{col_w['seeker']}}  {r['subfolder']:<{col_w['subfolder']}}"
            f"  {prog:>12}  {r['pct']:>{col_w['pct']}}  "
            f"{r['state']:<{col_w['state']}}  {r['job_id']:>{col_w['job_id']}}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
