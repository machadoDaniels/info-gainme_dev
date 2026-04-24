#!/bin/bash
# Analisa todos os runs.csv sob outputs/ e gera summary.json e variance.json.
# Uso:
#   ./dgx/run_analyze_results.sh              # analisa todos os runs.csv sob outputs/
#   ./dgx/run_analyze_results.sh path/to.csv  # analisa um CSV específico

umask 002

CSV_PATH="${1:-}"
SHARED_GROUP="sd22"

PROJECT_DIR="/raid/user_danielpedrozo/projects/info-gainme_dev"
SINGULARITY_IMAGE="/raid/user_danielpedrozo/images/vllm_openai_latest.sif"

echo "=========================================="
echo "Info Gainme - Analyze Results - $(date)"
if [ -n "${CSV_PATH}" ]; then
    echo "CSV: ${CSV_PATH}"
else
    echo "Modo: --all (todos os runs.csv sob outputs/)"
fi
echo "=========================================="

if [ -n "${CSV_PATH}" ]; then
    ANALYSIS_CMD="python3 scripts/analysis/analyze_results.py '${CSV_PATH}'"
else
    ANALYSIS_CMD="python3 scripts/analysis/analyze_results.py --all"
fi

sg "${SHARED_GROUP}" -c "
    singularity exec \
        --bind /raid/user_danielpedrozo:/workspace \
        --pwd /workspace/projects/info-gainme_dev \
        '${SINGULARITY_IMAGE}' \
        bash -c \"
            pip install --user -r requirements.txt 2>/dev/null
            ${ANALYSIS_CMD}
            python3 scripts/analysis/generate_unified_csv.py
        \"
"

echo "=========================================="
echo "Análise de resultados finalizada - $(date)"
echo "=========================================="
