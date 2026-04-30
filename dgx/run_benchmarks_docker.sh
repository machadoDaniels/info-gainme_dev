#!/bin/bash
# Roda benchmarks sequencialmente via Docker (sem SLURM, sem Singularity, sem vLLM local).
# Usa endpoints já disponíveis em configs/servers.yaml (ex: Qwen3-8B em h100n3).
# Equivalente ao run_benchmarks_screen.sh para nós com Docker.
#
# Uso:
#   bash dgx/run_benchmarks_docker.sh configs/full/8b/
#   bash dgx/run_benchmarks_docker.sh configs/full/8b/geo_160_8b_fo_cot.yaml

VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:latest}"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TARGET="${1:-configs/full/8b}"

# Se não estiver dentro de um screen, relança dentro de um automaticamente
if [[ -z "${STY}" ]]; then
    SESSION_NAME="benchmarks-$(basename "${TARGET}" .yaml)"
    LOG_ALL="${PROJECT_DIR}/logs/docker-$(basename "${TARGET}" .yaml)-all.log"
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
    mapfile -t CONFIGS < <(find "${TARGET}" -name "*.yaml" -type f | sort)
else
    echo "ERRO: '${TARGET}' não é um arquivo nem pasta válida."
    exit 1
fi

if [ ! -f "${PROJECT_DIR}/.env" ]; then
    echo "ERROR: .env não encontrado em ${PROJECT_DIR}/.env"
    exit 1
fi
source "${PROJECT_DIR}/.env"
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
export HF_TOKEN="${HF_TOKEN:-}"

mkdir -p "${PROJECT_DIR}/logs" "${PROJECT_DIR}/outputs"

echo "=========================================="
echo "Rodando ${#CONFIGS[@]} benchmark(s) via Docker"
echo "Target: ${TARGET}"
echo "=========================================="

for CONFIG in "${CONFIGS[@]}"; do
    REL="${CONFIG#${PROJECT_DIR}/}"
    NAME=$(basename "${CONFIG}" .yaml)
    LOG="${PROJECT_DIR}/logs/docker-${NAME}.log"

    echo ""
    echo ">>> ${NAME}"
    echo "    Log: ${LOG}"
    echo "    Início: $(date)"

    docker run --rm \
        --network=host \
        --entrypoint bash \
        -v "${PROJECT_DIR}:/workspace" \
        -v "${PROJECT_DIR}/.pip-cache:/root/.cache/pip" \
        -w /workspace \
        -e OPENAI_API_KEY="${OPENAI_API_KEY}" \
        -e HF_TOKEN="${HF_TOKEN}" \
        "${VLLM_IMAGE}" \
        -c "pip install --quiet -r requirements.txt 2>/dev/null && python3 benchmark_runner.py --config '${REL}'" \
        2>&1 | tee "${LOG}"

    echo "    Fim: $(date)"
done

echo ""
echo "=========================================="
echo "Todos os benchmarks finalizados - $(date)"
echo "=========================================="
