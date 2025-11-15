#!/usr/bin/env python3
"""Remove duplicatas do arquivo runs.csv mantendo a última ocorrência de cada cidade+run_index."""

import csv
from pathlib import Path
from collections import OrderedDict

def remove_duplicates(input_path: Path, output_path: Path = None):
    """Remove duplicatas do CSV mantendo a última ocorrência."""
    if output_path is None:
        output_path = input_path
    
    # Ler todas as linhas
    with input_path.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    
    # Usar OrderedDict para manter ordem, mas sobrescrever com última ocorrência
    seen = OrderedDict()
    for row in rows:
        key = (row['target_id'], row['run_index'])
        seen[key] = row  # Sempre sobrescreve com a última ocorrência
    
    # Escrever resultado
    with output_path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(seen.values())
    
    removed = len(rows) - len(seen)
    print(f"Removidas {removed} duplicatas de {len(rows)} linhas.")
    print(f"Resultado: {len(seen)} linhas únicas.")
    print(f"Arquivo salvo em: {output_path}")

if __name__ == '__main__':
    import sys
    
    csv_path = Path('outputs/models/s_Llama-3.1-8B-Instruct__o_Qwen3-8B__p_Qwen3-8B/top40_fo/runs.csv')
    
    if len(sys.argv) > 1:
        csv_path = Path(sys.argv[1])
    
    if not csv_path.exists():
        print(f"Erro: Arquivo não encontrado: {csv_path}")
        sys.exit(1)
    
    # Criar backup
    backup_path = csv_path.with_suffix('.csv.backup')
    import shutil
    shutil.copy2(csv_path, backup_path)
    print(f"Backup criado em: {backup_path}")
    
    # Remover duplicatas
    remove_duplicates(csv_path)



