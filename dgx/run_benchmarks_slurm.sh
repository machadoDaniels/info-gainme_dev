#!/bin/bash
# Executa benchmarks via screen
#
# Uso:
#   ./dgx/run_benchmarks_slurm.sh configs/8b/diseases_test_po_cot.yaml   # um yaml
#   ./dgx/run_benchmarks_slurm.sh configs/30b/cot/                       # pasta inteira

PROJECT_DIR="/raid/user_danielpedrozo/projects/info-gainme_dev"
TARGET="${1:-configs/8b}"
SCREEN_PREFIX="${SCREEN_PREFIX:-info-gainme-benchmark}"

# Resolve TARGET relative to PROJECT_DIR if not absolute
[[ "${TARGET}" != /* ]] && TARGET="${PROJECT_DIR}/${TARGET}"

# Parse args
shift || true
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dep)
            echo "AVISO: --dep é ignorado quando executando via screen."
            shift 2
            ;;
        *)     echo "Arg desconhecido: $1"; exit 1 ;;
    esac
done

# Resolve lista de configs
if [[ -f "${TARGET}" ]]; then
    CONFIGS=("${TARGET}")
elif [[ -d "${TARGET}" ]]; then
    mapfile -t CONFIGS < <(find "${TARGET}" -maxdepth 1 -name "*.yaml" | sort)
else
    echo "ERRO: '${TARGET}' não é um arquivo nem pasta válida."
    exit 1
fi

mkdir -p "${PROJECT_DIR}/logs/screen"

echo "=========================================="
echo "Iniciando ${#CONFIGS[@]} benchmark(s) via screen"
echo "Target: ${TARGET}"
echo "Sessões screen serão criadas com prefixo: ${SCREEN_PREFIX}"
echo "=========================================="

for CONFIG in "${CONFIGS[@]}"; do
    BASE_NAME="$(basename "${CONFIG}")"
    BASE_NAME="${BASE_NAME%.yaml}"
    SESSION_NAME="${SCREEN_PREFIX}-${BASE_NAME}"
    LOG_FILE="${PROJECT_DIR}/logs/screen/${SESSION_NAME}.log"

    screen -dmS "${SESSION_NAME}" bash -lc "cd '${PROJECT_DIR}' && bash dgx/run_benchmark.sh '${CONFIG}' >'${LOG_FILE}' 2>&1"
    echo "  ✓ $(basename ${CONFIG}) → screen ${SESSION_NAME}"
done

echo "=========================================="
echo "Todos iniciados. Acompanhe com: screen -ls"
echo "=========================================="
