#!/bin/bash
# Remove runs sinalizados por oracle_validation.csv para que possam ser
# re-executados. Por padrão roda em --dry-run.
#
# Uso:
#   ./dgx/run_delete_affected_runs.sh                 # dry-run (não altera nada)
#   ./dgx/run_delete_affected_runs.sh --apply         # aplica de fato
#   ./dgx/run_delete_affected_runs.sh --apply --errors-only

umask 002

SHARED_GROUP="sd22"
PROJECT_DIR="/raid/user_danielpedrozo/projects/info-gainme_dev"
SINGULARITY_IMAGE="/raid/user_danielpedrozo/images/vllm_openai_latest.sif"

APPLY=false
EXTRA_ARGS=""
for arg in "$@"; do
    case "$arg" in
        --apply) APPLY=true ;;
        *) EXTRA_ARGS="${EXTRA_ARGS} ${arg}" ;;
    esac
done

if ${APPLY}; then
    SCRIPT_ARGS="${EXTRA_ARGS}"
    echo "⚠  MODO APPLY — arquivos SERÃO deletados"
else
    SCRIPT_ARGS="--dry-run ${EXTRA_ARGS}"
    echo "ℹ  MODO DRY-RUN — nada será alterado. Use --apply para executar."
fi

echo "=========================================="
echo "Info Gainme - Delete Affected Runs - $(date)"
echo "Args: ${SCRIPT_ARGS}"
echo "=========================================="

sg "${SHARED_GROUP}" -c "
    singularity exec \
        --bind /raid/user_danielpedrozo:/workspace \
        --pwd /workspace/projects/info-gainme_dev \
        '${SINGULARITY_IMAGE}' \
        bash -c \"
            python3 scripts/maintenance/delete_affected_runs.py ${SCRIPT_ARGS}
        \"
"

echo "=========================================="
echo "Finalizado - $(date)"
echo "=========================================="
