"""
Analisa resultados de um experimento e gera summary.json e variance.json.

Usage:
    python scripts/analyze_results.py
    # ou especificar CSV:
    python scripts/analyze_results.py path/to/runs.csv
"""

import sys
from pathlib import Path

# Garantir imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.loader import load_experiment_results
from src.analysis.writer import save_summary, save_city_variance


def main():
    """Main entry point for results analysis."""
    # Default: analisar último experimento conhecido
    default_csv = Path("/Users/daniel2/Documents/AKCIT-RL/clary_quest/outputs/models/s_qwen3-8b__o_gpt-4o-mini__p_gpt-4o-mini/top40_po/runs.csv")
    
    # Permitir especificar CSV via argumento
    if len(sys.argv) > 1:
        csv_path = Path(sys.argv[1])
    else:
        csv_path = default_csv
    
    if not csv_path.exists():
        print(f"❌ CSV não encontrado: {csv_path}")
        print(f"\nUsage: python {Path(__file__).name} [path/to/runs.csv]")
        return 1
    
    print(f"\n{'='*70}")
    print(f"📊 ANÁLISE DE RESULTADOS - CLARY QUEST")
    print(f"{'='*70}")
    print(f"📁 CSV: {csv_path}")
    print(f"{'='*70}\n")
    
    # Carregar resultados
    try:
        results = load_experiment_results(csv_path)
    except Exception as e:
        print(f"❌ Erro ao carregar CSV: {e}")
        return 1
    
    # Exibir resumo no terminal
    print(f"🎯 Experimento: {results.experiment_name}")
    print(f"🤖 Models: {results.seeker_model} (S) / {results.oracle_model} (O) / {results.pruner_model} (P)")
    print(f"👁️  Observability: {results.observability}")
    print(f"🔢 Max Turns: {results.max_turns}")
    print(f"\n{'='*70}")
    print(f"📊 MÉTRICAS GLOBAIS")
    print(f"{'='*70}")
    print(f"📁 Total de runs: {results.total_runs}")
    print(f"🏙️  Total de cidades: {len(results.cities)}")
    print(f"📈 Ganho de informação médio: {results.mean_info_gain:.4f}")
    print(f"📊 GI médio por turno: {results.mean_avg_info_gain_per_turn:.4f}")
    print(f"🏆 Win rate global: {results.global_win_rate:.2%}")
    print(f"🔄 Média de turnos: {results.mean_turns:.2f}")
    print(f"✅ Compliance médio: {results.mean_compliance:.2%}")
    
    print(f"\n{'='*70}")
    print(f"📊 RESULTADOS POR CIDADE")
    print(f"{'='*70}")
    
    for city_id, city in sorted(results.cities.items(), key=lambda x: x[1].mean_info_gain, reverse=True):
        print(f"\n🏙️  {city.city_label} ({city_id})")
        print(f"   📊 Runs: {city.num_runs}")
        print(f"   📈 Ganho médio: {city.mean_info_gain:.4f} ± {city.std_info_gain:.4f}")
        print(f"   📉 Variância: {city.var_info_gain:.4f}")
        print(f"   📊 GI médio/turno: {city.mean_avg_info_gain_per_turn:.4f} ± {city.std_avg_info_gain_per_turn:.4f}")
        print(f"   🏆 Win rate: {city.win_rate:.2%}")
        print(f"   🔄 Turnos médios: {city.mean_turns:.2f} ± {city.std_turns:.2f}")
    
    # Salvar JSONs
    print(f"\n{'='*70}")
    print(f"💾 SALVANDO RESULTADOS")
    print(f"{'='*70}\n")
    
    output_dir = csv_path.parent
    save_summary(results, output_dir / "summary.json")
    save_city_variance(results, output_dir / "variance.json")
    
    print(f"\n{'='*70}")
    print(f"✅ ANÁLISE COMPLETA!")
    print(f"{'='*70}")
    print(f"📁 Resultados salvos em: {output_dir}")
    print(f"   - summary.json (métricas globais + por cidade)")
    print(f"   - variance.json (foco em variância)")
    print(f"{'='*70}\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

