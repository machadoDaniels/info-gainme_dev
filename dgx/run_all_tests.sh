#!/bin/bash
# Submete benchmarks via SLURM
#
# Uso:
#   ./dgx/run_all_tests.sh configs/8b/diseases_test_po_cot.yaml   # um yaml
#   ./dgx/run_all_tests.sh configs/30b/cot/                       # pasta inteira
#   ./dgx/run_all_tests.sh configs/30b/cot/ --dep 16130           # com dependency

PROJECT_DIR="/raid/user_danielpedrozo/projects/info-gainme_dev"
DEPENDENCY=""
TARGET="${1:-configs/8b}"

# Parse args
shift || true
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dep) DEPENDENCY="$2"; shift 2 ;;
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

SBATCH_ARGS=""
[ -n "${DEPENDENCY}" ] && SBATCH_ARGS="--dependency=after:${DEPENDENCY}"

echo "=========================================="
echo "Submetendo ${#CONFIGS[@]} benchmark(s)"
echo "Target: ${TARGET}"
[ -n "${DEPENDENCY}" ] && echo "Dependency: after:${DEPENDENCY}"
echo "=========================================="

for CONFIG in "${CONFIGS[@]}"; do
    JOB_ID=$(sbatch ${SBATCH_ARGS} "${PROJECT_DIR}/dgx/run_benchmark.sh" "${CONFIG}" | awk '{print $4}')
    echo "  ✓ $(basename ${CONFIG}) → job ${JOB_ID}"
done

echo "=========================================="
echo "Todos submetidos. Acompanhe com: squeue -u \$USER"
echo "=========================================="
