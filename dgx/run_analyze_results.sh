#!/bin/bash
# Analisa todos os runs.csv sob outputs/ e gera summary.json e variance.json.
# Uso:
#   ./dgx/run_analyze_results.sh                     # analisa todos os runs.csv
#   ./dgx/run_analyze_results.sh path/to.csv         # analisa um CSV específico
#   ./dgx/run_analyze_results.sh --only-run 1        # filtra run_index=1 (gera summary_run01.json)
#   ./dgx/run_analyze_results.sh path/to.csv --only-run 1

umask 002

CSV_PATH=""
ONLY_RUN_ARG=""
SHARED_GROUP="sd22"

PROJECT_DIR="/raid/user_danielpedrozo/projects/info-gainme_dev"
SINGULARITY_IMAGE="/raid/user_danielpedrozo/images/vllm_openai_latest.sif"

# Parsear argumentos
while [[ $# -gt 0 ]]; do
    case "$1" in
        --only-run)
            ONLY_RUN_ARG="--only-run $2"
            shift 2
            ;;
        *)
            CSV_PATH="$1"
            shift
            ;;
    esac
done

echo "=========================================="
echo "Info Gainme - Analyze Results - $(date)"
if [ -n "${CSV_PATH}" ]; then
    echo "CSV: ${CSV_PATH}"
else
    echo "Modo: --all (todos os runs.csv sob outputs/)"
fi
[ -n "${ONLY_RUN_ARG}" ] && echo "Filtro: ${ONLY_RUN_ARG}"
echo "=========================================="

if [ -n "${CSV_PATH}" ]; then
    ANALYSIS_CMD="python3 scripts/analysis/analyze_results.py '${CSV_PATH}' ${ONLY_RUN_ARG}"
else
    ANALYSIS_CMD="python3 scripts/analysis/analyze_results.py --all ${ONLY_RUN_ARG}"
fi

sg "${SHARED_GROUP}" -c "
    singularity exec \
        --bind /raid/user_danielpedrozo:/workspace \
        --pwd /workspace/projects/info-gainme_dev \
        '${SINGULARITY_IMAGE}' \
        bash -c \"
            pip install --user -r requirements.txt 2>/dev/null
            ${ANALYSIS_CMD}
            python3 scripts/analysis/generate_unified_csv.py ${ONLY_RUN_ARG}
        \"
"

echo "=========================================="
echo "Análise de resultados finalizada - $(date)"
echo "=========================================="
