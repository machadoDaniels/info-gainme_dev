#!/bin/bash
# Analisa todos os seeker_traces.json e extrai insights agregados.
# Uso:
#   ./dgx/run_analyze_traces.sh              # usa outputs/ padrão
#   ./dgx/run_analyze_traces.sh path/outputs # usa diretório customizado

umask 002

OUTPUTS_DIR="${1:-outputs}"
SHARED_GROUP="sd22"

PROJECT_DIR="/raid/user_danielpedrozo/projects/info-gainme_dev"
SINGULARITY_IMAGE="/raid/user_danielpedrozo/images/vllm_openai_latest.sif"

echo "=========================================="
echo "Info Gainme - Analyze Traces - $(date)"
echo "Outputs dir: ${OUTPUTS_DIR}"
echo "=========================================="

sg "${SHARED_GROUP}" -c "
    singularity exec \
        --bind /raid/user_danielpedrozo:/workspace \
        --pwd /workspace/projects/info-gainme_dev \
        '${SINGULARITY_IMAGE}' \
        bash -c \"
            pip install --user -r requirements.txt 2>/dev/null
            python3 scripts/analyze_reasoning_traces.py '${OUTPUTS_DIR}'
        \"
"

echo "=========================================="
echo "Análise de traces finalizada - $(date)"
echo "=========================================="
