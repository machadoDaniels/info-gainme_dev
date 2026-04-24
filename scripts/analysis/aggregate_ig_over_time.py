"""Agrega Information Gain ao longo do tempo de todas as cidades.

Lê os arquivos JSONL de métricas por cidade e agrega o IG médio por turno
across todas as cidades.

Usage:
    python scripts/aggregate_ig_over_time.py [city_metrics_dir] [output_file]
    
    - city_metrics_dir: Diretório contendo arquivos JSONL por cidade
    - output_file: (opcional) Arquivo de saída JSONL. Padrão: "aggregated_ig_over_time.jsonl"
                   no mesmo diretório de entrada

Cada linha do JSONL contém:
    - turn_index: número do turno
    - mean_info_gain: média do ganho de informação nesse turno
    - variance_info_gain: variância do ganho de informação nesse turno
    - num_cities: número de cidades que chegaram nesse turno
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any

# Garantir imports do projeto
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def load_city_metrics(jsonl_path: Path) -> List[Dict[str, Any]]:
    """Carrega métricas de uma cidade de um arquivo JSONL.
    
    Args:
        jsonl_path: Caminho para o arquivo JSONL da cidade
        
    Returns:
        Lista de dicionários com dados de cada turno
    """
    turns = []
    if not jsonl_path.exists():
        return turns
    
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                turn_data = json.loads(line)
                turns.append(turn_data)
            except json.JSONDecodeError as e:
                print(f"⚠️  Erro ao parsear linha em {jsonl_path}: {e}")
                continue
    
    return turns


def aggregate_ig_over_time(city_metrics_dir: Path, output_file: Path) -> None:
    """Agrega IG ao longo do tempo de todas as cidades.
    
    Args:
        city_metrics_dir: Diretório contendo arquivos JSONL por cidade
        output_file: Arquivo de saída JSONL
    """
    # Encontrar todos os arquivos JSONL
    jsonl_files = list(city_metrics_dir.glob("*.jsonl"))
    
    if not jsonl_files:
        print(f"❌ Nenhum arquivo JSONL encontrado em {city_metrics_dir}")
        return
    
    print(f"📖 Processando {len(jsonl_files)} arquivos de cidades...")
    
    # Estrutura: turn_index -> lista de mean_info_gain de todas as cidades
    turn_data: Dict[int, List[float]] = defaultdict(list)
    
    # Carregar dados de todas as cidades
    cities_processed = 0
    for jsonl_path in jsonl_files:
        city_turns = load_city_metrics(jsonl_path)
        
        if not city_turns:
            continue
        
        cities_processed += 1
        
        # Agregar mean_info_gain por turno
        for turn in city_turns:
            turn_index = turn.get("turn_index")
            mean_info_gain = turn.get("mean_info_gain")
            
            if turn_index is not None and mean_info_gain is not None:
                turn_data[turn_index].append(mean_info_gain)
    
    print(f"✅ Processadas {cities_processed} cidades")
    
    # Calcular estatísticas agregadas por turno
    print(f"📊 Calculando estatísticas agregadas por turno...")
    
    aggregated_turns = []
    for turn_index in sorted(turn_data.keys()):
        info_gains = turn_data[turn_index]
        
        if not info_gains:
            continue
        
        # Calcular estatísticas
        mean_ig = statistics.mean(info_gains)
        
        # Variância
        if len(info_gains) > 1:
            variance_ig = statistics.variance(info_gains)
        else:
            variance_ig = 0.0
        
        aggregated_turns.append({
            "turn_index": turn_index,
            "mean_info_gain": round(mean_ig, 6),
            "variance_info_gain": round(variance_ig, 6),
            "num_cities": len(info_gains)
        })
    
    # Salvar arquivo JSONL
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        for turn in aggregated_turns:
            f.write(json.dumps(turn, ensure_ascii=False) + "\n")
    
    print(f"✅ Agregação completa! {len(aggregated_turns)} turnos agregados")
    print(f"📁 Arquivo salvo em: {output_file}")
    
    # Mostrar resumo
    if aggregated_turns:
        total_ig = sum(t["mean_info_gain"] for t in aggregated_turns)
        avg_ig_per_turn = total_ig / len(aggregated_turns)
        print(f"\n📊 Resumo:")
        print(f"   - Turnos agregados: {len(aggregated_turns)}")
        print(f"   - IG médio por turno: {avg_ig_per_turn:.4f}")
        print(f"   - IG total acumulado: {total_ig:.4f}")


def main() -> None:
    """Função principal."""
    if len(sys.argv) < 2:
        print("Usage: python scripts/aggregate_ig_over_time.py [city_metrics_dir] [output_file]")
        print("\nExemplo:")
        print("  python scripts/aggregate_ig_over_time.py outputs/models/.../city_metrics_by_turn")
        print("  python scripts/aggregate_ig_over_time.py outputs/models/.../city_metrics_by_turn aggregated.jsonl")
        sys.exit(1)
    
    city_metrics_dir = Path(sys.argv[1])
    
    # Output padrão: mesmo diretório com nome padrão
    if len(sys.argv) > 2:
        output_file = Path(sys.argv[2])
    else:
        output_file = city_metrics_dir / "../aggregated_ig_over_time.jsonl"
    
    if not city_metrics_dir.exists():
        print(f"❌ Diretório não encontrado: {city_metrics_dir}")
        sys.exit(1)
    
    if not city_metrics_dir.is_dir():
        print(f"❌ Caminho não é um diretório: {city_metrics_dir}")
        sys.exit(1)
    
    aggregate_ig_over_time(city_metrics_dir, output_file)


if __name__ == "__main__":
    main()
