#!/bin/bash
# Valida se o Oracle respeitou o formato Yes/No definido em
# src/prompts/oracle_system.md. Escreve oracle_validation.csv +
# oracle_validation.json no diretório alvo.
#
# Uso:
#   ./dgx/run_validate_oracle.sh              # varre outputs/
#   ./dgx/run_validate_oracle.sh path/dir     # varre o diretório dado
#   ./dgx/run_validate_oracle.sh outputs --errors-only   # args extras passados ao script

umask 002

SHARED_GROUP="sd22"
PROJECT_DIR="/raid/user_danielpedrozo/projects/info-gainme_dev"
SINGULARITY_IMAGE="/raid/user_danielpedrozo/images/vllm_openai_latest.sif"

BASE_DIR="${1:-outputs}"
shift 2>/dev/null || true
EXTRA_ARGS="$*"

echo "=========================================="
echo "Info Gainme - Validate Oracle Outputs - $(date)"
echo "Base dir: ${BASE_DIR}"
if [ -n "${EXTRA_ARGS}" ]; then
    echo "Extra args: ${EXTRA_ARGS}"
fi
echo "=========================================="

sg "${SHARED_GROUP}" -c "
    singularity exec \
        --bind /raid/user_danielpedrozo:/workspace \
        --pwd /workspace/projects/info-gainme_dev \
        '${SINGULARITY_IMAGE}' \
        bash -c \"
            python3 scripts/maintenance/validate_oracle_answers.py '${BASE_DIR}' ${EXTRA_ARGS}
        \"
"

echo "=========================================="
echo "Validação do Oracle finalizada - $(date)"
echo "=========================================="
