#!/usr/bin/env python3
"""Analisa todos os reasoning_traces.json gerados e extrai insights agregados.

Lê todos os seeker_traces.json, agrega:
- Distribuição de turnos
- Opções de perguntas mais consideradas
- Tipos de decisões/padrões de raciocínio
- Comparação entre experimentos
"""

import json
import sys
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Any
import statistics

def load_all_traces(output_dir: Path = Path("outputs")) -> Dict[str, List[Dict]]:
    """Carrega todos os seeker_traces.json agrupados por experimento."""
    traces_by_exp = defaultdict(list)
    traces_dir = output_dir / "models"

    trace_files = list(traces_dir.glob("**/conversations/*/seeker_traces.json"))
    print(f"📂 Carregando {len(trace_files)} traces...")

    for trace_file in trace_files:
        try:
            # Extrair nome do experimento
            parts = trace_file.parts
            # outputs/models/[config]/[dataset]/conversations/[target]/seeker_traces.json
            exp_name = parts[-4] if len(parts) >= 4 else "unknown"

            # Ignorar experimentos sem CoT (modelos sem raciocínio explícito)
            if not (exp_name.endswith("_cot") or "_cot_" in exp_name):
                continue

            data = json.load(open(trace_file))
            traces_by_exp[exp_name].append(data)

        except Exception as e:
            print(f"  ⚠️  Erro em {trace_file}: {e}")

    return traces_by_exp

def aggregate_traces(traces_by_exp: Dict[str, List[Dict]]) -> Dict[str, Any]:
    """Agrega estatísticas dos traces."""

    all_options = Counter()
    all_summaries = []
    all_decisions = Counter()

    exp_stats = {}

    for exp_name, exp_traces in traces_by_exp.items():
        turn_counts = []
        options_in_exp = Counter()
        summaries_in_exp = []
        decisions_in_exp = Counter()

        for trace in exp_traces:
            history = trace.get("history", [])
            turn_counts.append(len(history))

            for turn in history:
                rt = turn.get("reasoning_trace", {})
                if not rt:
                    continue

                # Opções consideradas
                opts = rt.get("options_considered", [])
                for opt in opts:
                    if isinstance(opt, str):
                        all_options[opt] += 1
                        options_in_exp[opt] += 1

                # Sumários (padrões de raciocínio)
                summary = rt.get("summary", "")
                if summary:
                    all_summaries.append(summary)
                    summaries_in_exp.append(summary)

                # Decisões/rationale
                decision = rt.get("decision_rationale", "")
                if decision:
                    # Pegar primeiras 60 chars como fingerprint
                    fp = decision[:60].strip()
                    all_decisions[fp] += 1
                    decisions_in_exp[fp] += 1

        if turn_counts:
            exp_stats[exp_name] = {
                "num_conversations": len(exp_traces),
                "total_turns_sampled": sum(turn_counts),
                "avg_turns": statistics.mean(turn_counts),
                "median_turns": statistics.median(turn_counts),
                "min_turns": min(turn_counts),
                "max_turns": max(turn_counts),
                "num_unique_options": len(options_in_exp),
                "top_options": options_in_exp.most_common(5),
                "num_summaries": len(summaries_in_exp),
            }

    return {
        "total_traces": sum(len(traces) for traces in traces_by_exp.values()),
        "total_conversations": sum(exp_stats[e]["num_conversations"] for e in exp_stats),
        "experiments": exp_stats,
        "global_top_options": all_options.most_common(20),
        "global_top_decisions": all_decisions.most_common(15),
        "global_num_summaries": len(all_summaries),
        "global_num_unique_options": len(all_options),
    }

def print_report(agg: Dict[str, Any]) -> None:
    """Imprime relatório formatado."""

    print("\n" + "="*80)
    print("📊 ANÁLISE DE REASONING TRACES")
    print("="*80)

    print(f"\n✅ TOTAL: {agg['total_traces']} traces de {agg['total_conversations']} conversas")
    print(f"📝 Opções únicas consideradas: {agg['global_num_unique_options']}")
    print(f"💭 Sumários de raciocínio: {agg['global_num_summaries']}")

    # Estatísticas por experimento
    print("\n" + "-"*80)
    print("📈 ESTATÍSTICAS POR EXPERIMENTO:")
    print("-"*80)

    for exp, stats in sorted(agg['experiments'].items()):
        print(f"\n  📁 {exp}")
        print(f"      Conversas: {stats['num_conversations']}")
        print(f"      Turnos: avg={stats['avg_turns']:.1f}, min={stats['min_turns']}, max={stats['max_turns']}")
        print(f"      Opções únicas: {stats['num_unique_options']}")
        if stats['top_options']:
            print(f"      Top 3 opções:")
            for opt, count in stats['top_options'][:3]:
                pct = (count / stats['num_summaries'] * 100) if stats['num_summaries'] > 0 else 0
                print(f"        • {count}x ({pct:.1f}%) - {opt[:70]}")

    # Top opções globais
    print("\n" + "-"*80)
    print("🎯 TOP 20 OPÇÕES CONSIDERADAS GLOBALMENTE:")
    print("-"*80)
    for i, (opt, count) in enumerate(agg['global_top_options'], 1):
        pct = (count / agg['global_num_summaries'] * 100) if agg['global_num_summaries'] > 0 else 0
        print(f"  {i:2d}. {count:5d}x ({pct:5.1f}%) - {opt[:75]}")

    # Top decisões/rationales
    print("\n" + "-"*80)
    print("💡 TOP 15 PADRÕES DE DECISÃO (rationale):")
    print("-"*80)
    for i, (decision, count) in enumerate(agg['global_top_decisions'], 1):
        print(f"  {i:2d}. {count:5d}x - {decision[:75]}")

    print("\n" + "="*80)

def save_json_report(agg: Dict[str, Any], output_file: Path) -> None:
    """Salva relatório em JSON."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(agg, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n💾 Relatório salvo em: {output_file}")

if __name__ == "__main__":
    output_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("outputs")

    traces_by_exp = load_all_traces(output_dir)
    agg = aggregate_traces(traces_by_exp)

    print_report(agg)

    # Salvar JSON
    report_file = output_dir / "reasoning_traces_analysis.json"
    save_json_report(agg, report_file)
