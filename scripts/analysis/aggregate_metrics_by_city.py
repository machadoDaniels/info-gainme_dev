"""Agrega métricas por turno para cada cidade.

Cria uma pasta com arquivos JSONL, um para cada cidade, contendo métricas agregadas
por turno de todas as execuções dessa cidade.

Usage:
    python scripts/aggregate_metrics_by_city.py [runs_csv_path] [output_dir]
    
    - runs_csv_path: Caminho para o arquivo runs.csv
    - output_dir: (opcional) Diretório de saída. Padrão: subpasta "city_metrics_by_turn" 
                  no mesmo diretório do runs.csv
    - O diretório base "outputs" é inferido automaticamente a partir do caminho

Cada linha do JSONL contém:
    - turn_index: número do turno
    - mean_info_gain: média do ganho de informação nesse turno
    - variance_info_gain: variância do ganho de informação nesse turno
    - num_runs: número de execuções que chegaram nesse turno
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any

# Garantir imports do projeto
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def load_turns_from_jsonl(jsonl_path: Path) -> List[Dict[str, Any]]:
    """Carrega todos os turnos de um arquivo turns.jsonl.
    
    Args:
        jsonl_path: Caminho para o arquivo turns.jsonl
        
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


def aggregate_metrics_by_city(runs_csv_path: Path, output_dir: Path, base_outputs_dir: Path) -> None:
    """Agrega métricas por turno para cada cidade.
    
    Args:
        runs_csv_path: Caminho para o arquivo runs.csv
        output_dir: Diretório onde salvar os arquivos JSONL agregados
        base_outputs_dir: Diretório base dos outputs (para construir caminhos relativos)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Estrutura: city_id -> turn_index -> lista de info_gains
    city_turns_data: Dict[str, Dict[int, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    
    # Ler runs.csv e processar cada conversa
    print(f"📖 Lendo {runs_csv_path}...")
    with runs_csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            target_id = row["target_id"]
            target_label = row["target_label"]
            conversation_path = row.get("conversation_path", "")
            
            if not conversation_path:
                continue
            
            # Construir caminho completo para turns.jsonl
            turns_jsonl_path = base_outputs_dir / conversation_path / "turns.jsonl"
            
            if not turns_jsonl_path.exists():
                print(f"⚠️  Arquivo não encontrado: {turns_jsonl_path}")
                continue
            
            # Carregar turnos dessa execução
            turns = load_turns_from_jsonl(turns_jsonl_path)
            
            # Agregar dados por turno
            for turn in turns:
                turn_index = turn.get("turn_index")
                info_gain = turn.get("info_gain")
                
                if turn_index is None or info_gain is None:
                    continue
                
                city_turns_data[target_id][turn_index].append({
                    "info_gain": info_gain
                })
    
    # Calcular estatísticas e salvar arquivos JSONL por cidade
    print(f"📊 Agregando métricas para {len(city_turns_data)} cidades...")
    
    for city_id, turns_data in city_turns_data.items():
        # Obter label da cidade (pegar do primeiro run)
        city_label = None
        with runs_csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["target_id"] == city_id:
                    city_label = row["target_label"]
                    break
        
        # Criar nome do arquivo seguro
        safe_label = city_label.replace(" ", "_").replace("/", "-") if city_label else city_id
        output_file = output_dir / f"{safe_label}_{city_id}.jsonl"
        
        # Calcular estatísticas por turno
        aggregated_turns = []
        for turn_index in sorted(turns_data.keys()):
            turn_data_list = turns_data[turn_index]
            
            # Extrair info_gains
            info_gains = [d["info_gain"] for d in turn_data_list]
            
            # Calcular média e variância
            mean_info_gain = statistics.mean(info_gains) if info_gains else 0.0
            
            # Variância amostral (usando n-1 no denominador)
            if len(info_gains) > 1:
                variance_info_gain = statistics.variance(info_gains)
            elif len(info_gains) == 1:
                variance_info_gain = 0.0
            else:
                variance_info_gain = 0.0
            
            aggregated_turns.append({
                "turn_index": turn_index,
                "mean_info_gain": round(mean_info_gain, 6),
                "variance_info_gain": round(variance_info_gain, 6),
                "num_runs": len(turn_data_list)
            })
        
        # Salvar arquivo JSONL
        with output_file.open("w", encoding="utf-8") as f:
            for turn in aggregated_turns:
                f.write(json.dumps(turn, ensure_ascii=False) + "\n")
        
        print(f"✅ {city_label} ({city_id}): {len(aggregated_turns)} turnos agregados")
    
    print(f"\n🎉 Agregação completa! Arquivos salvos em: {output_dir}")


def find_outputs_base_dir(runs_csv_path: Path) -> Path:
    """Encontra o diretório base 'outputs' a partir do caminho do runs.csv.
    
    Args:
        runs_csv_path: Caminho para o arquivo runs.csv
        
    Returns:
        Diretório base 'outputs'
    """
    # Percorrer o caminho procurando por "outputs"
    current = runs_csv_path.resolve()
    while current != current.parent:
        if current.name == "outputs":
            return current
        current = current.parent
    
    # Se não encontrar, assumir que "outputs" está no diretório raiz do projeto
    # (subindo 3 níveis típicos: experiment_name/ -> models/ -> outputs/)
    return runs_csv_path.parent.parent.parent


def main() -> None:
    """Função principal."""
    if len(sys.argv) < 2:
        print("Usage: python scripts/aggregate_metrics_by_city.py [runs_csv_path] [output_dir]")
        print("\nExemplo:")
        print("  python scripts/aggregate_metrics_by_city.py outputs/models/.../runs.csv")
        print("  python scripts/aggregate_metrics_by_city.py outputs/models/.../runs.csv custom_output_dir")
        sys.exit(1)
    
    runs_csv_path = Path(sys.argv[1])
    # Output padrão: subpasta no mesmo diretório do runs.csv
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else runs_csv_path.parent / "city_metrics_by_turn"
    # Inferir base_outputs_dir automaticamente
    base_outputs_dir = find_outputs_base_dir(runs_csv_path)
    
    if not runs_csv_path.exists():
        print(f"❌ Arquivo não encontrado: {runs_csv_path}")
        sys.exit(1)
    
    print(f"📁 Diretório base inferido: {base_outputs_dir}")
    aggregate_metrics_by_city(runs_csv_path, output_dir, base_outputs_dir)


if __name__ == "__main__":
    main()
