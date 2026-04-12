"""Gera um CSV unificado com métricas globais de todos os experimentos.

Usage:
    # Varre ./outputs e salva em ./outputs/unified_experiments.csv
    python scripts/generate_unified_csv.py

    # Informar diretório base de outputs e caminho de saída
    python scripts/generate_unified_csv.py [base_outputs_dir] [output_csv_path]

Colunas geradas:
    Experimento, Seeker Model, Oracle Model, Pruner Model, Observabilidade, Total Runs, Win Rate,
    Mean Turns, Mean Info Gain/Turn, Mean Info Gain, Mean Initial Entropy,
    Mean Seeker Tokens, Mean Seeker Reasoning Tokens, Mean Seeker Final Tokens,
    SE Win Rate, SE Mean Turns, SE Mean Info Gain/Turn, SE Mean Info Gain, SE Mean Initial Entropy,
    SE Mean Seeker Tokens, SE Mean Seeker Reasoning Tokens, SE Mean Seeker Final Tokens, id

id: `s_<seeker>__o_<oracle>__p_<pruner>__<observabilidade>__<experimento>` (mesma convenção das pastas em outputs/models/).
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

# Garantir imports do projeto
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.loader import load_experiment_results


def _compose_experiment_id(row: dict) -> str:
    """Id único alinhado a `outputs/models/s_*__o_*__p_*`/ — inclui os três papéis explicitamente."""
    s = (row.get("Seeker Model") or "").strip()
    o = (row.get("Oracle Model") or "").strip()
    p = (row.get("Pruner Model") or "").strip()
    obs = (row.get("Observabilidade") or "").strip()
    exp = (row.get("Experimento") or "").strip()
    return f"s_{s}__o_{o}__p_{p}__{obs}__{exp}"


HEADERS = [
    "Experimento",
    "Seeker Model",
    "Oracle Model",
    "Pruner Model",
    "Observabilidade",
    "Total Runs",
    "Win Rate",
    "Mean Turns",
    "Mean Info Gain/Turn",
    "Mean Info Gain",
    "Mean Initial Entropy",
    "Mean Seeker Tokens",
    "Mean Seeker Reasoning Tokens",
    "Mean Seeker Final Tokens",
    "SE Win Rate",
    "SE Mean Turns",
    "SE Mean Info Gain/Turn",
    "SE Mean Info Gain",
    "SE Mean Initial Entropy",
    "SE Mean Seeker Tokens",
    "SE Mean Seeker Reasoning Tokens",
    "SE Mean Seeker Final Tokens",
    "id",
]


def _extract_from_summary(summary_path: Path) -> dict | None:
    """Extrai métricas do summary.json se existir."""
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        global_metrics = data.get("global_metrics", {}) or {}
        models = data.get("models", {}) or {}
        return {
            "Experimento": data.get("experiment_name"),
            "Seeker Model": models.get("seeker"),
            "Oracle Model": models.get("oracle"),
            "Pruner Model": models.get("pruner"),
            "Observabilidade": (data.get("config", {}) or {}).get("observability"),
            "Total Runs": global_metrics.get("total_runs"),
            "Win Rate": global_metrics.get("win_rate"),
            "Mean Turns": global_metrics.get("mean_turns"),
            "Mean Info Gain/Turn": global_metrics.get("mean_avg_info_gain_per_turn"),
            "Mean Info Gain": global_metrics.get("mean_info_gain"),
            "Mean Initial Entropy": global_metrics.get("mean_h_start"),
            "Mean Seeker Tokens": global_metrics.get("mean_seeker_tokens"),
            "Mean Seeker Reasoning Tokens": global_metrics.get("mean_seeker_reasoning_tokens"),
            "Mean Seeker Final Tokens": global_metrics.get("mean_seeker_final_tokens"),
            "SE Win Rate": global_metrics.get("se_win_rate"),
            "SE Mean Turns": global_metrics.get("se_mean_turns"),
            "SE Mean Info Gain/Turn": global_metrics.get("se_mean_avg_info_gain_per_turn"),
            "SE Mean Info Gain": global_metrics.get("se_mean_info_gain"),
            "SE Mean Initial Entropy": global_metrics.get("se_mean_h_start"),
            "SE Mean Seeker Tokens": global_metrics.get("se_mean_seeker_tokens"),
            "SE Mean Seeker Reasoning Tokens": global_metrics.get("se_mean_seeker_reasoning_tokens"),
            "SE Mean Seeker Final Tokens": global_metrics.get("se_mean_seeker_final_tokens"),
        }
    except Exception:
        return None


def _extract_from_runs_csv(runs_csv: Path) -> dict | None:
    """Calcula métricas carregando o runs.csv se summary.json não existir."""
    try:
        results = load_experiment_results(runs_csv)
        return {
            "Experimento": results.experiment_name,
            "Seeker Model": results.seeker_model,
            "Oracle Model": results.oracle_model,
            "Pruner Model": results.pruner_model,
            "Observabilidade": results.observability,
            "Total Runs": results.total_runs,
            "Win Rate": round(results.global_win_rate, 4),
            "Mean Turns": round(results.mean_turns, 2),
            "Mean Info Gain/Turn": round(results.mean_avg_info_gain_per_turn, 4),
            "Mean Info Gain": round(results.mean_info_gain, 4),
            "Mean Initial Entropy": round(results.mean_h_start, 4),
            "Mean Seeker Tokens": round(results.mean_seeker_tokens, 0),
            "Mean Seeker Reasoning Tokens": round(results.mean_seeker_reasoning_tokens, 0) if results.mean_seeker_reasoning_tokens is not None else None,
            "Mean Seeker Final Tokens": round(results.mean_seeker_final_tokens, 0),
            "SE Win Rate": round(results.se_win_rate, 4),
            "SE Mean Turns": round(results.se_mean_turns, 2),
            "SE Mean Info Gain/Turn": round(results.se_mean_avg_info_gain_per_turn, 4),
            "SE Mean Info Gain": round(results.se_mean_info_gain, 4),
            "SE Mean Initial Entropy": round(results.se_mean_h_start, 4),
            "SE Mean Seeker Tokens": round(results.se_mean_seeker_tokens, 0),
            "SE Mean Seeker Reasoning Tokens": round(results.se_mean_seeker_reasoning_tokens, 0) if results.se_mean_seeker_reasoning_tokens is not None else None,
            "SE Mean Seeker Final Tokens": round(results.se_mean_seeker_final_tokens, 0),
        }
    except Exception:
        return None


def _iter_experiments(base_outputs_dir: Path) -> list[dict]:
    """Percorre o diretório base e coleta linhas para o CSV unificado."""
    rows: list[dict] = []
    # Preferir summary.json quando disponível
    for summary_path in sorted(base_outputs_dir.rglob("summary.json")):
        row = _extract_from_summary(summary_path)
        if row:
            row["id"] = _compose_experiment_id(row)
            rows.append(row)

    # Para pastas sem summary.json, tentar via runs.csv
    seen_ids = {r["id"] for r in rows}
    for runs_csv in sorted(base_outputs_dir.rglob("runs.csv")):
        # Se existe summary no mesmo dir, já capturado
        if (runs_csv.parent / "summary.json").exists():
            continue
        row = _extract_from_runs_csv(runs_csv)
        if not row:
            continue
        row["id"] = _compose_experiment_id(row)
        if row["id"] in seen_ids:
            continue
        rows.append(row)
        seen_ids.add(row["id"])

    return rows


def main() -> int:
    """Ponto de entrada para geração do CSV unificado."""
    repo_root = Path(__file__).parent.parent
    default_base = repo_root / "outputs"
    default_out = default_base / "unified_experiments.csv"

    # Argumentos opcionais
    if len(sys.argv) >= 2:
        base_outputs_dir = Path(sys.argv[1])
    else:
        base_outputs_dir = default_base

    if len(sys.argv) >= 3:
        output_csv = Path(sys.argv[2])
    else:
        output_csv = default_out

    if not base_outputs_dir.exists():
        print(f"❌ Diretório de outputs não encontrado: {base_outputs_dir}")
        print(f"Usage: python {Path(__file__).name} [base_outputs_dir] [output_csv]")
        return 1

    print(f"🔎 Lendo experimentos em: {base_outputs_dir}")
    rows = _iter_experiments(base_outputs_dir)

    if not rows:
        print("❌ Nenhum experimento encontrado (summary.json ou runs.csv)")
        return 1

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HEADERS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"✅ CSV unificado salvo em: {output_csv}")
    print(f"📦 Total de experimentos: {len(rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


