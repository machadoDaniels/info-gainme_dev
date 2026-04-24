#!/usr/bin/env python3
"""Recalcula Standard Errors (SE) para question_evaluations_summary.json existentes.

Este script lê todos os question_evaluations_summary.json existentes e recalcula
os SEs usando a abordagem hierárquica (por target/cidade), adicionando-os ao JSON.

Usage:
    python scripts/recalculate_question_evaluation_se.py [base_outputs_dir]
"""

import json
import math
import statistics
import sys
from pathlib import Path
from typing import Dict, Any, Optional


def calculate_se_from_targets(by_target: Dict[str, Dict[str, Any]]) -> Dict[str, Optional[float]]:
    """Calcula stds e SEs globais usando abordagem hierárquica (por target).
    
    Args:
        by_target: Dicionário com estatísticas por target/cidade.
        
    Returns:
        Dicionário com stds e SEs calculados.
    """
    num_targets = len(by_target)
    if num_targets <= 1:
        return {
            "std_optimal_choice_rate": 0.0,
            "se_optimal_choice_rate": 0.0,
            "std_chosen_info_gain": None,
            "se_chosen_info_gain": None,
            "std_optimal_info_gain": None,
            "se_optimal_info_gain": None,
            "std_questions_considered_per_turn": None,
            "se_questions_considered_per_turn": None,
        }
    
    # Coletar médias por target
    target_rates = [target["avg_optimal_choice_rate"] for target in by_target.values()]
    target_chosen_igs = [
        target["avg_chosen_info_gain"] 
        for target in by_target.values() 
        if target.get("avg_chosen_info_gain") is not None
    ]
    target_optimal_igs = [
        target["avg_optimal_info_gain"] 
        for target in by_target.values() 
        if target.get("avg_optimal_info_gain") is not None
    ]
    target_questions = [
        target["avg_questions_considered_per_turn"] 
        for target in by_target.values() 
        if target.get("avg_questions_considered_per_turn") is not None
    ]
    
    # Calcular stds (amostral para inferência estatística)
    std_rate = statistics.stdev(target_rates) if len(target_rates) > 1 else 0.0
    std_chosen_ig = statistics.stdev(target_chosen_igs) if len(target_chosen_igs) > 1 else None
    std_optimal_ig = statistics.stdev(target_optimal_igs) if len(target_optimal_igs) > 1 else None
    std_questions = statistics.stdev(target_questions) if len(target_questions) > 1 else None
    
    # Calcular SEs (hierarchical: std / sqrt(n))
    se_rate = std_rate / math.sqrt(num_targets) if num_targets > 1 else 0.0
    se_chosen_ig = std_chosen_ig / math.sqrt(len(target_chosen_igs)) if target_chosen_igs and len(target_chosen_igs) > 1 else None
    se_optimal_ig = std_optimal_ig / math.sqrt(len(target_optimal_igs)) if target_optimal_igs and len(target_optimal_igs) > 1 else None
    se_questions = std_questions / math.sqrt(len(target_questions)) if target_questions and len(target_questions) > 1 else None
    
    return {
        "std_optimal_choice_rate": std_rate,
        "se_optimal_choice_rate": se_rate,
        "std_chosen_info_gain": std_chosen_ig,
        "se_chosen_info_gain": se_chosen_ig,
        "std_optimal_info_gain": std_optimal_ig,
        "se_optimal_info_gain": se_optimal_ig,
        "std_questions_considered_per_turn": std_questions,
        "se_questions_considered_per_turn": se_questions,
    }


def recalculate_se_for_summary(summary_path: Path) -> bool:
    """Recalcula e adiciona SEs a um question_evaluations_summary.json.
    
    Args:
        summary_path: Caminho para o arquivo JSON.
        
    Returns:
        True se atualizado com sucesso, False caso contrário.
    """
    try:
        # Ler JSON existente
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        
        # Verificar se já tem by_target
        by_target = data.get("by_target", {})
        if not by_target:
            print(f"⚠️  {summary_path}: Sem dados 'by_target', pulando...")
            return False
        
        # Calcular stds e SEs
        stats = calculate_se_from_targets(by_target)
        
        # Atualizar aggregate_statistics
        if "aggregate_statistics" not in data:
            data["aggregate_statistics"] = {}
        
        # Atualizar stds e SEs (sobrescrever se já existirem)
        data["aggregate_statistics"].update({
            "std_optimal_choice_rate": round(stats["std_optimal_choice_rate"], 4) if stats["std_optimal_choice_rate"] is not None else None,
            "se_optimal_choice_rate": round(stats["se_optimal_choice_rate"], 4) if stats["se_optimal_choice_rate"] is not None else None,
            "std_chosen_info_gain": round(stats["std_chosen_info_gain"], 4) if stats["std_chosen_info_gain"] is not None else None,
            "se_chosen_info_gain": round(stats["se_chosen_info_gain"], 4) if stats["se_chosen_info_gain"] is not None else None,
            "std_optimal_info_gain": round(stats["std_optimal_info_gain"], 4) if stats["std_optimal_info_gain"] is not None else None,
            "se_optimal_info_gain": round(stats["se_optimal_info_gain"], 4) if stats["se_optimal_info_gain"] is not None else None,
            "std_questions_considered_per_turn": round(stats["std_questions_considered_per_turn"], 4) if stats["std_questions_considered_per_turn"] is not None else None,
            "se_questions_considered_per_turn": round(stats["se_questions_considered_per_turn"], 4) if stats["se_questions_considered_per_turn"] is not None else None,
        })
        
        # Salvar JSON atualizado
        summary_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return True
        
    except Exception as e:
        print(f"❌ Erro ao processar {summary_path}: {e}")
        return False


def main() -> int:
    """Ponto de entrada para recalcular SEs."""
    repo_root = Path(__file__).parent.parent.parent
    default_base = repo_root / "outputs"
    
    # Argumentos opcionais
    if len(sys.argv) >= 2:
        base_outputs_dir = Path(sys.argv[1])
    else:
        base_outputs_dir = default_base
    
    if not base_outputs_dir.exists():
        print(f"❌ Diretório não encontrado: {base_outputs_dir}")
        print(f"Usage: python {Path(__file__).name} [base_outputs_dir]")
        return 1
    
    print(f"🔎 Procurando question_evaluations_summary.json em: {base_outputs_dir}")
    
    # Encontrar todos os arquivos
    summary_files = sorted(base_outputs_dir.rglob("question_evaluations_summary.json"))
    
    if not summary_files:
        print("❌ Nenhum question_evaluations_summary.json encontrado")
        return 1
    
    print(f"📦 Encontrados {len(summary_files)} arquivos para processar\n")
    
    updated = 0
    skipped = 0
    errors = 0
    
    for summary_path in summary_files:
        print(f"🔄 Processando: {summary_path.relative_to(base_outputs_dir)}")
        if recalculate_se_for_summary(summary_path):
            updated += 1
            print(f"   ✅ SEs atualizados")
        else:
            skipped += 1
            print(f"   ⏭️  Pulado")
    
    print(f"\n{'='*70}")
    print(f"📊 Resumo:")
    print(f"   ✅ Atualizados: {updated}")
    print(f"   ⏭️  Pulados: {skipped}")
    print(f"   ❌ Erros: {errors}")
    print(f"{'='*70}\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

