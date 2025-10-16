"""Loader for experiment results from CSV files."""

import csv
from pathlib import Path
from collections import defaultdict

from .data_types import GameRun, CityStats, ExperimentResults


def load_experiment_results(csv_path: Path) -> ExperimentResults:
    """
    Carrega um arquivo runs.csv e retorna ExperimentResults estruturado.
    
    Args:
        csv_path: Caminho para runs.csv
        
    Returns:
        ExperimentResults com todas as métricas calculadas
        
    Raises:
        ValueError: Se o CSV estiver vazio ou malformado
    """
    csv_path = Path(csv_path)
    
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV não encontrado: {csv_path}")
    
    with csv_path.open("r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    if not rows:
        raise ValueError(f"CSV vazio: {csv_path}")
    
    # Extrair metadata do experimento (da primeira linha)
    first = rows[0]
    experiment_name = first["experiment_name"]
    seeker_model = first["seeker_model"]
    oracle_model = first["oracle_model"]
    pruner_model = first["pruner_model"]
    observability = first["observability"]
    max_turns = int(first["max_turns"])
    
    # Agrupar runs por cidade
    cities_data = defaultdict(list)
    for row in rows:
        game_run = GameRun(
            target_id=row["target_id"],
            target_label=row["target_label"],
            run_index=int(row["run_index"]),
            turns=int(row["turns"]),
            h_start=float(row["h_start"]),
            h_end=float(row["h_end"]),
            total_info_gain=float(row["total_info_gain"]),
            avg_info_gain_per_turn=float(row.get("avg_info_gain_per_turn", 0.0)),
            win=bool(int(row["win"])),
            compliance_rate=float(row["compliance_rate"]),
            conversation_path=row.get("conversation_path") or None,
        )
        cities_data[game_run.target_id].append(game_run)
    
    # Criar CityStats para cada cidade
    cities = {}
    for city_id, runs in cities_data.items():
        city_label = runs[0].target_label  # Todas runs têm mesmo label
        cities[city_id] = CityStats(
            city_id=city_id,
            city_label=city_label,
            runs=runs,
        )
    
    return ExperimentResults(
        experiment_name=experiment_name,
        seeker_model=seeker_model,
        oracle_model=oracle_model,
        pruner_model=pruner_model,
        observability=observability,
        max_turns=max_turns,
        cities=cities,
    )

