"""
Analisa resultados de um experimento e gera summary.json e variance.json.

Usage:
    # analisar um CSV específico
    python scripts/analyze_results.py path/to/runs.csv

    # analisar todos os runs.csv sob outputs (default: ./outputs)
    python scripts/analyze_results.py --all [base_outputs_dir]

    # sem argumentos: usa um CSV padrão legado
    python scripts/analyze_results.py
"""

import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

# Garantir imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.analysis.loader import load_experiment_results
from src.analysis.writer import save_summary, save_city_variance


def _analyze_single_csv(csv_path_str: str, verbose: bool = True, force: bool = False) -> tuple[str, int]:
    """Executa a análise para um único arquivo runs.csv.

    Args:
        csv_path_str: Caminho (string) para o arquivo runs.csv
        verbose: Se True, imprime informações detalhadas
        force: Se True, re-analisa mesmo que summary.json já exista e esteja atualizado

    Returns:
        Tupla (csv_path_str, exit_code) onde exit_code é 0 para sucesso, 1 para erro
    """
    # Garantir sys.path para processos separados (imports dentro da função)
    import sys
    from pathlib import Path

    # Calcular repo_root baseado no caminho do script
    script_dir = Path(__file__).parent if '__file__' in globals() else Path.cwd()
    repo_root = script_dir.parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    # Importar aqui para garantir que sys.path está configurado
    from src.analysis.loader import load_experiment_results
    from src.analysis.writer import save_summary, save_city_variance

    csv_path = Path(csv_path_str)
    if not csv_path.exists():
        if verbose:
            print(f"❌ CSV não encontrado: {csv_path}")
        return (csv_path_str, 1)

    # Pular se summary.json já existe e é mais recente que runs.csv
    if not force:
        summary_path = csv_path.parent / "summary.json"
        if summary_path.exists() and summary_path.stat().st_mtime >= csv_path.stat().st_mtime:
            if verbose:
                print(f"⏭️  Pulando (summary.json atualizado): {csv_path}")
            return (csv_path_str, 0)

    if verbose:
        print(f"\n{'='*70}")
        print(f"📊 ANÁLISE DE RESULTADOS - CLARY QUEST")
        print(f"{'='*70}")
        print(f"📁 CSV: {csv_path}")
        print(f"{'='*70}\n")

    try:
        results = load_experiment_results(csv_path)
    except Exception as e:
        if verbose:
            print(f"❌ Erro ao carregar CSV: {e}")
        return (csv_path_str, 1)

    if verbose:
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
        print(f"🔤 Tokens médios (Seeker): {results.mean_seeker_tokens:.0f}")
        if results.mean_seeker_reasoning_tokens is not None:
            print(f"🧠 Tokens reasoning médios: {results.mean_seeker_reasoning_tokens:.0f}")
            print(f"💬 Tokens resposta final médios: {results.mean_seeker_final_tokens:.0f}")

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
            print(f"   🔤 Tokens médios: {city.mean_seeker_tokens:.0f}")
            if city.mean_seeker_reasoning_tokens is not None:
                print(f"   🧠 Tokens reasoning: {city.mean_seeker_reasoning_tokens:.0f}")
                print(f"   💬 Tokens resposta final: {city.mean_seeker_final_tokens:.0f}")

        print(f"\n{'='*70}")
        print(f"💾 SALVANDO RESULTADOS")
        print(f"{'='*70}\n")

    output_dir = csv_path.parent
    save_summary(results, output_dir / "summary.json")
    save_city_variance(results, output_dir / "variance.json")

    if verbose:
        print(f"\n{'='*70}")
        print(f"✅ ANÁLISE COMPLETA!")
        print(f"{'='*70}")
        print(f"📁 Resultados salvos em: {output_dir}")
        print(f"   - summary.json (métricas globais + por cidade)")
        print(f"   - variance.json (foco em variância)")
        print(f"{'='*70}\n")

    return (csv_path_str, 0)


def main():
    """Main entry point for results analysis."""
    # Base de outputs padrão relativa ao projeto
    repo_root = Path(__file__).parent.parent.parent
    default_outputs_dir = repo_root / "outputs"

    # Compatibilidade legado: CSV padrão específico (se nada for passado)
    default_csv = (
        repo_root
        / "outputs/models/s_Llama-3.1-8B-Instruct__o_Qwen3-8B__p_Qwen3-8B/top40_fo/runs.csv"
    )

    # Sem argumentos → tenta default_csv legado
    if len(sys.argv) == 1:
        _, exit_code = _analyze_single_csv(str(default_csv))
        return exit_code

    # Modo --all: varre todos os runs.csv sob base_outputs_dir
    if sys.argv[1] in {"--all", "all"}:
        base_dir = (
            Path(sys.argv[2])
            if len(sys.argv) > 2
            else default_outputs_dir
        )

        if not base_dir.exists():
            print(f"❌ Diretório base não encontrado: {base_dir}")
            print(
                f"\nUsage: python {Path(__file__).name} --all [base_outputs_dir]"
            )
            return 1

        runs_files = sorted(base_dir.rglob("runs.csv"))
        if not runs_files:
            print(f"❌ Nenhum runs.csv encontrado em: {base_dir}")
            return 1

        print(f"\n{'='*70}")
        print(f"🚀 PROCESSAMENTO PARALELO - {len(runs_files)} arquivos")
        print(f"{'='*70}")
        print(f"📁 Base: {base_dir}")
        print(f"🔢 Workers: {os.cpu_count() or 4}")
        print(f"{'='*70}\n")

        total_ok = 0
        total_fail = 0
        failed_files = []

        # Processar em paralelo
        with ProcessPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
            # Submeter todas as tarefas (converter Path para string para serialização)
            future_to_csv = {
                executor.submit(_analyze_single_csv, str(csv_path), verbose=False): csv_path
                for csv_path in runs_files
            }

            # Processar resultados conforme completam
            for future in as_completed(future_to_csv):
                csv_path = future_to_csv[future]
                try:
                    csv_path_str, exit_code = future.result()
                    if exit_code == 0:
                        total_ok += 1
                        print(f"✅ {csv_path_str}")
                    else:
                        total_fail += 1
                        failed_files.append(csv_path_str)
                        print(f"❌ {csv_path_str}")
                except Exception as e:
                    total_fail += 1
                    failed_files.append(str(csv_path))
                    print(f"❌ {csv_path} - Erro: {e}")

        print(f"\n{'='*70}")
        print(f"📦 Processados: {total_ok} com sucesso, {total_fail} com erro(s)")
        if failed_files:
            print(f"\n❌ Arquivos com erro:")
            for f in failed_files:
                print(f"   - {f}")
        print(f"{'='*70}\n")
        return 0 if total_fail == 0 else 1

    # Caso contrário: argumento é caminho para um CSV específico
    _, exit_code = _analyze_single_csv(sys.argv[1])
    return exit_code


if __name__ == "__main__":
    sys.exit(main())

