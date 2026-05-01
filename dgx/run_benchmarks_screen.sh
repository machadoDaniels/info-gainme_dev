#!/bin/bash
# Roda benchmarks sequencialmente via screen (sem SLURM)
#
# Uso:
#   bash dgx/run_benchmarks_screen.sh configs/full/8b/
#   bash dgx/run_benchmarks_screen.sh configs/full/8b/geo_160_8b_fo_cot.yaml

PROJECT_DIR="/raid/user_danielpedrozo/projects/info-gainme_dev"
SINGULARITY_IMAGE="/raid/user_danielpedrozo/images/vllm_openai_latest.sif"
SHARED_GROUP="sd22"
TARGET="${1:-configs/full/8b}"

# Se não estiver dentro de um screen, relança dentro de um automaticamente
if [[ -z "${STY}" ]]; then
    SESSION_NAME="benchmarks-$(basename "${TARGET}" .yaml)"
    LOG_ALL="${PROJECT_DIR}/logs/screen-$(basename "${TARGET}" .yaml)-all.log"
    mkdir -p "${PROJECT_DIR}/logs"
    echo "Iniciando screen '${SESSION_NAME}'..."
    echo "Acompanhe com: screen -r ${SESSION_NAME}"
    echo "Ou: tail -f ${LOG_ALL}"
    screen -dmS "${SESSION_NAME}" bash -c \
        "bash '$(realpath "$0")' '${TARGET}' 2>&1 | tee '${LOG_ALL}'; exec bash"
    exit 0
fi

[[ "${TARGET}" != /* ]] && TARGET="${PROJECT_DIR}/${TARGET}"

if [[ -f "${TARGET}" ]]; then
    CONFIGS=("${TARGET}")
elif [[ -d "${TARGET}" ]]; then
    mapfile -t CONFIGS < <(find "${TARGET}" -maxdepth 1 -name "*.yaml" | sort)
else
    echo "ERRO: '${TARGET}' não é um arquivo nem pasta válida."
    exit 1
fi

mkdir -p "${PROJECT_DIR}/logs"
mkdir -p "${PROJECT_DIR}/outputs/logs"
chmod o+rwx "${PROJECT_DIR}/outputs" "${PROJECT_DIR}/outputs/logs" 2>/dev/null
umask 002

echo "=========================================="
echo "Rodando ${#CONFIGS[@]} benchmark(s) via screen"
echo "Target: ${TARGET}"
echo "=========================================="

for CONFIG in "${CONFIGS[@]}"; do
    BENCHMARK_CONFIG="${CONFIG#${PROJECT_DIR}/}"
    NAME=$(basename "${CONFIG}" .yaml)
    LOG="${PROJECT_DIR}/logs/screen-${NAME}.log"

    echo ""
    echo ">>> ${NAME}"
    echo "    Log: ${LOG}"
    echo "    Início: $(date)"

    sg "${SHARED_GROUP}" -c "
        singularity exec \
            --bind /raid/user_danielpedrozo:/workspace \
            --pwd /workspace/projects/info-gainme_dev \
            '${SINGULARITY_IMAGE}' \
            bash -c \"
                pip install --quiet --user -r requirements.txt
                python3 benchmark_runner.py --config '${BENCHMARK_CONFIG}'
            \"
    " 2>&1 | tee "${LOG}"

    echo "    Fim: $(date)"
done

echo ""
echo "=========================================="
echo "Todos os benchmarks finalizados - $(date)"
echo "=========================================="
