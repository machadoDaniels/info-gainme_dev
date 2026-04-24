"""Plot aggregated Information Gain over time for all experiments.

Finds all aggregated_ig_over_time.jsonl files in outputs/models and creates
comparative plots showing mean IG over turns.

Usage:
    python scripts/plot_aggregated_ig.py [output_dir]
    
    - output_dir: (optional) Directory to save plots. Default: "outputs/plots"
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# Garantir imports do projeto
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Configurar matplotlib para usar backend não-interativo
matplotlib.use('Agg')


def load_aggregated_data(jsonl_path: Path) -> List[Dict[str, Any]]:
    """Carrega dados agregados de um arquivo JSONL.
    
    Args:
        jsonl_path: Caminho para o arquivo aggregated_ig_over_time.jsonl
        
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


def extract_model_and_experiment(file_path: Path) -> Tuple[str, str, str]:
    """Extrai modelo e nome do experimento a partir do caminho do arquivo.
    
    Args:
        file_path: Caminho completo para aggregated_ig_over_time.jsonl
        
    Returns:
        Tupla (modelo_categoria, modelo_nome, experimento_nome)
        modelo_categoria: "qwen8b", "qwen30b", ou "llama_paprika"
    """
    # Caminho típico: outputs/models/s_MODEL__o_MODEL__p_MODEL/EXPERIMENT_NAME/aggregated_ig_over_time.jsonl
    parts = file_path.parts
    
    # Encontrar índice de "models"
    try:
        models_idx = parts.index("models")
        if models_idx + 2 < len(parts):
            model_name = parts[models_idx + 1]  # s_MODEL__o_MODEL__p_MODEL
            exp_name = parts[models_idx + 2]    # EXPERIMENT_NAME
            
            # Extrair nome do modelo seeker
            seeker_model = model_name.replace("s_", "").split("__")[0]
            
            # Categorizar modelo
            seeker_lower = seeker_model.lower()
            
            if "qwen3-8b" in seeker_lower or ("qwen" in seeker_lower and "8b" in seeker_lower and "30" not in seeker_lower):
                model_category = "qwen8b"
                model_display = "Qwen 8B"
            elif "qwen3-30b" in seeker_lower or ("qwen" in seeker_lower and "30b" in seeker_lower):
                model_category = "qwen30b"
                model_display = "Qwen 30B"
            elif "llama" in seeker_lower or "paprika" in seeker_lower:
                model_category = "llama_paprika"
                if "paprika" in seeker_lower:
                    model_display = "Paprika Llama 3.1 8B"
                else:
                    model_display = "Llama 3.1 8B"
            else:
                model_category = "other"
                model_display = seeker_model
            
            return model_category, model_display, exp_name
    except (ValueError, IndexError):
        pass
    
    # Fallback: usar nome do diretório pai
    return "other", file_path.parent.name, file_path.parent.name


def extract_experiment_name(file_path: Path) -> str:
    """Extrai nome do experimento formatado (compatibilidade).
    
    Args:
        file_path: Caminho completo para aggregated_ig_over_time.jsonl
        
    Returns:
        Nome do experimento formatado
    """
    _, model_display, exp_name = extract_model_and_experiment(file_path)
    return f"{model_display} | {exp_name}"


def plot_all_experiments(
    output_dir: Path,
    base_dir: Path = None
) -> None:
    """Plota gráficos de IG agregado para todos os experimentos.
    
    Args:
        output_dir: Diretório para salvar os gráficos
        base_dir: Diretório base para buscar arquivos. Padrão: outputs/models
    """
    if base_dir is None:
        base_dir = Path("outputs/models")
    
    # Encontrar todos os arquivos aggregated_ig_over_time.jsonl
    jsonl_files = list(base_dir.rglob("aggregated_ig_over_time.jsonl"))
    
    if not jsonl_files:
        print(f"❌ No aggregated_ig_over_time.jsonl files found in {base_dir}")
        return
    
    print(f"📊 Found {len(jsonl_files)} experiments to plot...")
    
    # Load data from all experiments (incluindo caminho do arquivo)
    experiments_data: List[Tuple[str, str, str, List[Dict[str, Any]], Path]] = []
    
    for jsonl_path in sorted(jsonl_files):
        model_category, model_display, exp_name = extract_model_and_experiment(jsonl_path)
        data = load_aggregated_data(jsonl_path)
        
        if data:
            experiments_data.append((model_category, model_display, exp_name, data, jsonl_path))
            print(f"  ✅ {model_display} | {exp_name}: {len(data)} turns")
    
    if not experiments_data:
        print("❌ No valid data found")
        return
    
    # Criar gráfico
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Criar visualização agregada em grade 2x3
    plot_aggregated_grid(experiments_data, output_dir)
    
    # Criar gráficos separados por modelo
    plot_by_model(experiments_data, output_dir)


def plot_aggregated_grid(
    experiments_data: List[Tuple[str, str, str, List[Dict[str, Any]], Path]],
    output_dir: Path
) -> None:
    """Cria uma visualização agregada em grade 2x3 (FO/PO x Modelos).
    
    Args:
        experiments_data: Lista de tuplas (modelo_categoria, modelo_display, exp_name, dados)
        output_dir: Diretório para salvar o gráfico
    """
    # Agrupar por modelo e observabilidade
    qwen8b_fo = []
    qwen8b_po = []
    qwen30b_fo = []
    qwen30b_po = []
    llama_paprika_fo = []
    llama_paprika_po = []
    
    for model_category, model_display, exp_name, data, file_path in experiments_data:
        is_fo = is_fo_experiment(exp_name)
        
        if model_category == "qwen8b":
            if is_fo:
                qwen8b_fo.append((model_display, exp_name, data, file_path))
            else:
                qwen8b_po.append((model_display, exp_name, data, file_path))
        elif model_category == "qwen30b":
            if is_fo:
                qwen30b_fo.append((model_display, exp_name, data, file_path))
            else:
                qwen30b_po.append((model_display, exp_name, data, file_path))
        elif model_category == "llama_paprika":
            if is_fo:
                llama_paprika_fo.append((model_display, exp_name, data, file_path))
            else:
                llama_paprika_po.append((model_display, exp_name, data, file_path))
    
    # Criar figura com subplots 2x3
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    # fig.suptitle("Aggregated Information Gain Over Time", fontsize=16, fontweight='bold', y=0.995)
    
    # Função auxiliar para plotar em um subplot
    def plot_subplot(ax, experiments_list, title, ylabel=False):
        if not experiments_list:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes, fontsize=16)
            ax.set_title(title, fontsize=16, fontweight='bold')
            return
        
        for model_display, exp_name, data, file_path in experiments_list:
            turn_indices = [d["turn_index"] for d in data]
            mean_ig = [d["mean_info_gain"] for d in data]
            variance_ig = [d["variance_info_gain"] for d in data]
            num_cities = [d["num_cities"] for d in data]
            
            # Calcular erro padrão (standard error): SE = sqrt(variance) / sqrt(n)
            std_error = [math.sqrt(v) / math.sqrt(n) if n > 0 and v > 0 else 0.0 
                        for v, n in zip(variance_ig, num_cities)]
            
            # Obter cor fixa baseada na configuração do modelo
            config_id = get_model_config_id(model_display, exp_name, file_path)
            color = get_color_for_config(config_id)
            
            # Usar nome formatado do modelo para o label
            label = get_formatted_model_name(config_id)
            
            # Plotar linha
            ax.plot(
                turn_indices,
                mean_ig,
                label=label,
                color=color,
                linewidth=2,
                marker='o',
                markersize=3,
                alpha=0.8
            )
            
            # Adicionar sombra com erro padrão
            upper_bound = [m + se for m, se in zip(mean_ig, std_error)]
            lower_bound = [m - se for m, se in zip(mean_ig, std_error)]
            
            ax.fill_between(
                turn_indices,
                lower_bound,
                upper_bound,
                color=color,
                alpha=0.2,
                linewidth=0
            )
        
        ax.set_xlabel("Turn", fontsize=14, fontweight='bold')
        if ylabel:
            ax.set_ylabel("Average Information Gain", fontsize=14, fontweight='bold')
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=13)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
    
    # Row 1: FO (Fully Observable)
    plot_subplot(axes[0, 0], qwen8b_fo, "Qwen 8B", ylabel=True)
    plot_subplot(axes[0, 1], qwen30b_fo, "Qwen 30B")
    plot_subplot(axes[0, 2], llama_paprika_fo, "Llama 3.1 8B & Paprika")
    
    # Row 2: PO (Partially Observable)
    plot_subplot(axes[1, 0], qwen8b_po, "Qwen 8B", ylabel=True)
    plot_subplot(axes[1, 1], qwen30b_po, "Qwen 30B")
    plot_subplot(axes[1, 2], llama_paprika_po, "Llama 3.1 8B & Paprika")
    
    # Add side labels (more visible)
    # Centralizar verticalmente: linha superior ~0.76, linha inferior ~0.33 (considerando rect bottom=0.12)
    fig.text(0.015, 0.76, "Fully Observable\n(FO)", rotation=90, fontsize=17, fontweight='bold', 
             ha='center', va='center', color='black')
    fig.text(0.015, 0.33, "Partially Observable\n(PO)", rotation=90, fontsize=17, fontweight='bold', 
             ha='center', va='center', color='black')
    
    # Create unified legend at the bottom
    # Collect all unique model configs and assign fixed colors and formatted names
    all_configs_dict = {}  # {config_id: (color, formatted_name)}
    
    for experiments_list in [qwen8b_fo, qwen8b_po, qwen30b_fo, qwen30b_po, llama_paprika_fo, llama_paprika_po]:
        for model_display, exp_name, data, file_path in experiments_list:
            config_id = get_model_config_id(model_display, exp_name, file_path)
            if config_id not in all_configs_dict:
                color = get_color_for_config(config_id)
                formatted_name = get_formatted_model_name(config_id)
                all_configs_dict[config_id] = (color, formatted_name)
    
    # Create handles and labels for legend (ordenado por config_id)
    handles = []
    labels_legend = []
    for config_id in sorted(all_configs_dict.keys()):
        color, formatted_name = all_configs_dict[config_id]
        handles.append(plt.Line2D([0], [0], color=color, 
                                  linewidth=2, marker='o', markersize=5))
        labels_legend.append(formatted_name)
    
    # Add legend at the bottom centered
    plt.tight_layout(rect=[0.03, 0.12, 1, 0.98])
    
    fig.legend(handles, labels_legend, loc='lower center', ncol=min(6, len(labels_legend)), 
              fontsize=13, bbox_to_anchor=(0.5, 0.08), frameon=True, fancybox=True, shadow=True)
    
    # Save plot
    output_path = output_dir / "aggregated_ig_grid.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
    print(f"✅ Aggregated grid plot saved to: {output_path}")
    plt.close()


def is_fo_experiment(exp_name: str) -> bool:
    """Verifica se o experimento é Fully Observable.
    
    Args:
        exp_name: Nome do experimento
        
    Returns:
        True se for FO, False se for PO
    """
    exp_lower = exp_name.lower()
    return "_fo" in exp_lower or "fully_observable" in exp_lower


def is_cot_experiment(exp_name: str) -> bool:
    """Verifica se o experimento usa Chain of Thought (CoT).
    
    Args:
        exp_name: Nome do experimento
        
    Returns:
        True se usar CoT, False se for no_cot
    """
    exp_lower = exp_name.lower()
    return "_no_cot" not in exp_lower and "no_cot" not in exp_lower


def get_model_config_id(model_display: str, exp_name: str, file_path: Path = None) -> str:
    """Identifica a configuração específica do modelo para atribuir cor fixa.
    
    Args:
        model_display: Nome de exibição do modelo
        exp_name: Nome do experimento
        file_path: Caminho do arquivo (opcional, para extrair mais info)
        
    Returns:
        ID da configuração do modelo (1-6)
    """
    # Extrair nome completo do modelo do caminho se disponível
    model_full_name = model_display
    if file_path:
        parts = file_path.parts
        try:
            models_idx = parts.index("models")
            if models_idx + 1 < len(parts):
                model_name = parts[models_idx + 1]  # s_MODEL__o_MODEL__p_MODEL
                seeker_model = model_name.replace("s_", "").split("__")[0]
                model_full_name = seeker_model
        except (ValueError, IndexError):
            pass
    
    model_lower = model_full_name.lower()
    exp_lower = exp_name.lower()
    
    # 5: paprika_Meta-Llama-3.1-8B-Instruct (verificar primeiro para evitar conflito)
    if "paprika" in model_lower:
        return "5"
    
    # 6: Llama-3.1-8B-Instruct (sem paprika)
    if "llama" in model_lower and "paprika" not in model_lower:
        return "6"
    
    # 4: Qwen3-30B Thinking (A3B-Thinking-2507) - verificar antes de Instruct
    if ("qwen3-30b" in model_lower or "30b" in model_lower) and "thinking" in model_lower:
        return "4"
    
    # 3: Qwen3-30B Instruct (A3B-Instruct-2507)
    if ("qwen3-30b" in model_lower or "30b" in model_lower) and "instruct" in model_lower:
        return "3"
    
    # 1 e 2: Qwen3-8B (verificar COT vs No CoT)
    if "qwen3-8b" in model_lower or ("qwen" in model_lower and "8b" in model_lower and "30" not in model_lower):
        # 2: Qwen3-8B No CoT
        if "_no_cot" in exp_lower or "no_cot" in exp_lower:
            return "2"
        # 1: Qwen3-8B COT
        else:
            return "1"
    
    # Fallback
    return "unknown"


def get_color_for_config(config_id: str) -> str:
    """Retorna a cor fixa para uma configuração específica.
    
    Args:
        config_id: ID da configuração (1-6)
        
    Returns:
        Cor hexadecimal
    """
    color_map = {
        "1": "#1f77b4",  # Azul - Qwen3-8B COT
        "2": "#aec7e8",  # Azul claro - Qwen3-8B No CoT
        "3": "#ff7f0e",  # Laranja - Qwen3-30B Instruct
        "4": "#2ca02c",  # Verde - Qwen3-30B Thinking
        "5": "#9467bd",  # Roxo - paprika Llama
        "6": "#8c564b",  # Marrom - Llama
    }
    return color_map.get(config_id, "#000000")  # Preto como fallback


def get_formatted_model_name(config_id: str) -> str:
    """Retorna o nome formatado do modelo para a legenda.
    
    Args:
        config_id: ID da configuração (1-6)
        
    Returns:
        Nome formatado do modelo
    """
    name_map = {
        "1": "Qwen3-8B (CoT)",
        "2": "Qwen3-8B (No CoT)",
        "3": "Qwen3-30B-Instruct",
        "4": "Qwen3-30B-Thinking",
        "5": "paprika_Meta-Llama-3.1-8B-Instruct",
        "6": "Llama-3.1-8B-Instruct",
    }
    return name_map.get(config_id, "Unknown")


def plot_by_model(
    experiments_data: List[Tuple[str, str, str, List[Dict[str, Any]], Path]],
    output_dir: Path
) -> None:
    """Cria gráficos separados agrupados por modelo e observabilidade.
    
    Args:
        experiments_data: Lista de tuplas (modelo_categoria, modelo_display, exp_name, dados)
        output_dir: Diretório para salvar os gráficos
    """
    # Agrupar por modelo e observabilidade
    qwen8b_fo = []
    qwen8b_po = []
    qwen30b_fo = []
    qwen30b_po = []
    llama_paprika_fo = []
    llama_paprika_po = []
    
    for model_category, model_display, exp_name, data, file_path in experiments_data:
        is_fo = is_fo_experiment(exp_name)
        
        if model_category == "qwen8b":
            if is_fo:
                qwen8b_fo.append((model_display, exp_name, data, file_path))
            else:
                qwen8b_po.append((model_display, exp_name, data, file_path))
        elif model_category == "qwen30b":
            if is_fo:
                qwen30b_fo.append((model_display, exp_name, data, file_path))
            else:
                qwen30b_po.append((model_display, exp_name, data, file_path))
        elif model_category == "llama_paprika":
            if is_fo:
                llama_paprika_fo.append((model_display, exp_name, data, file_path))
            else:
                llama_paprika_po.append((model_display, exp_name, data, file_path))
    
    # Função auxiliar para plotar um grupo
    def plot_group(experiments_list, title, filename):
        if not experiments_list:
            return
        
        fig, ax = plt.subplots(figsize=(12, 7))
        colors = plt.cm.tab10(np.linspace(0, 1, len(experiments_list)))
        
        for model_display, exp_name, data, file_path in experiments_list:
            turn_indices = [d["turn_index"] for d in data]
            mean_ig = [d["mean_info_gain"] for d in data]
            variance_ig = [d["variance_info_gain"] for d in data]
            num_cities = [d["num_cities"] for d in data]
            
            # Calcular erro padrão (standard error): SE = sqrt(variance) / sqrt(n)
            std_error = [math.sqrt(v) / math.sqrt(n) if n > 0 and v > 0 else 0.0 
                        for v, n in zip(variance_ig, num_cities)]
            
            # Obter cor fixa baseada na configuração do modelo
            config_id = get_model_config_id(model_display, exp_name, file_path)
            color = get_color_for_config(config_id)
            
            # Plotar linha
            ax.plot(
                turn_indices,
                mean_ig,
                label=exp_name,
                color=color,
                linewidth=2,
                marker='o',
                markersize=4,
                alpha=0.8
            )
            
            # Adicionar sombra com erro padrão
            upper_bound = [m + se for m, se in zip(mean_ig, std_error)]
            lower_bound = [m - se for m, se in zip(mean_ig, std_error)]
            
            ax.fill_between(
                turn_indices,
                lower_bound,
                upper_bound,
                color=color,
                alpha=0.2,
                linewidth=0
            )
        
        ax.set_xlabel("Turn", fontsize=12, fontweight='bold')
        ax.set_ylabel("Average Information Gain", fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        plt.tight_layout()
        
        output_path = output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Plot saved to: {output_path}")
        plt.close()
    
    # Plotar Qwen 8B - FO
    plot_group(qwen8b_fo, "Information Gain - Qwen 8B (FO)", 
               "aggregated_ig_qwen8b_fo.png")
    
    # Plotar Qwen 8B - PO
    plot_group(qwen8b_po, "Information Gain - Qwen 8B (PO)", 
               "aggregated_ig_qwen8b_po.png")
    
    # Plotar Qwen 30B - FO
    plot_group(qwen30b_fo, "Information Gain - Qwen 30B (FO)", 
               "aggregated_ig_qwen30b_fo.png")
    
    # Plotar Qwen 30B - PO
    plot_group(qwen30b_po, "Information Gain - Qwen 30B (PO)", 
               "aggregated_ig_qwen30b_po.png")
    
    # Plotar Llama/Paprika - FO (com label especial)
    if llama_paprika_fo:
        fig, ax = plt.subplots(figsize=(12, 7))
        
        for model_display, exp_name, data, file_path in llama_paprika_fo:
            turn_indices = [d["turn_index"] for d in data]
            mean_ig = [d["mean_info_gain"] for d in data]
            variance_ig = [d["variance_info_gain"] for d in data]
            num_cities = [d["num_cities"] for d in data]
            
            # Calcular erro padrão (standard error): SE = sqrt(variance) / sqrt(n)
            std_error = [math.sqrt(v) / math.sqrt(n) if n > 0 and v > 0 else 0.0 
                        for v, n in zip(variance_ig, num_cities)]
            
            # Obter cor fixa baseada na configuração do modelo
            config_id = get_model_config_id(model_display, exp_name, file_path)
            color = get_color_for_config(config_id)
            
            label = f"{model_display} | {exp_name}"
            
            # Plotar linha
            ax.plot(
                turn_indices,
                mean_ig,
                label=label,
                color=color,
                linewidth=2,
                marker='o',
                markersize=4,
                alpha=0.8
            )
            
            # Adicionar sombra com erro padrão
            upper_bound = [m + se for m, se in zip(mean_ig, std_error)]
            lower_bound = [m - se for m, se in zip(mean_ig, std_error)]
            
            ax.fill_between(
                turn_indices,
                lower_bound,
                upper_bound,
                color=color,
                alpha=0.2,
                linewidth=0
            )
        
        ax.set_xlabel("Turn", fontsize=12, fontweight='bold')
        ax.set_ylabel("Average Information Gain", fontsize=12, fontweight='bold')
        ax.set_title("Information Gain - Llama 3.1 8B & Paprika (FO)", fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        plt.tight_layout()
        
        output_path = output_dir / "aggregated_ig_llama_paprika_fo.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Plot saved to: {output_path}")
        plt.close()
    
    # Plotar Llama/Paprika - PO (com label especial)
    if llama_paprika_po:
        fig, ax = plt.subplots(figsize=(12, 7))
        
        for model_display, exp_name, data, file_path in llama_paprika_po:
            turn_indices = [d["turn_index"] for d in data]
            mean_ig = [d["mean_info_gain"] for d in data]
            variance_ig = [d["variance_info_gain"] for d in data]
            num_cities = [d["num_cities"] for d in data]
            
            # Calcular erro padrão (standard error): SE = sqrt(variance) / sqrt(n)
            std_error = [math.sqrt(v) / math.sqrt(n) if n > 0 and v > 0 else 0.0 
                        for v, n in zip(variance_ig, num_cities)]
            
            # Obter cor fixa baseada na configuração do modelo
            config_id = get_model_config_id(model_display, exp_name, file_path)
            color = get_color_for_config(config_id)
            
            label = f"{model_display} | {exp_name}"
            
            # Plotar linha
            ax.plot(
                turn_indices,
                mean_ig,
                label=label,
                color=color,
                linewidth=2,
                marker='o',
                markersize=4,
                alpha=0.8
            )
            
            # Adicionar sombra com erro padrão
            upper_bound = [m + se for m, se in zip(mean_ig, std_error)]
            lower_bound = [m - se for m, se in zip(mean_ig, std_error)]
            
            ax.fill_between(
                turn_indices,
                lower_bound,
                upper_bound,
                color=color,
                alpha=0.2,
                linewidth=0
            )
        
        ax.set_xlabel("Turn", fontsize=12, fontweight='bold')
        ax.set_ylabel("Average Information Gain", fontsize=12, fontweight='bold')
        ax.set_title("Information Gain - Llama 3.1 8B & Paprika (PO)", fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        plt.tight_layout()
        
        output_path = output_dir / "aggregated_ig_llama_paprika_po.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Plot saved to: {output_path}")
        plt.close()


def main() -> None:
    """Função principal."""
    if len(sys.argv) > 1:
        output_dir = Path(sys.argv[1])
    else:
        output_dir = Path("outputs/plots")
    
    print("📊 Plotting aggregated Information Gain plots...")
    print(f"📁 Output directory: {output_dir}")
    
    plot_all_experiments(output_dir)
    
    print("\n🎉 Plotting complete!")


if __name__ == "__main__":
    main()
