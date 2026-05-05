#!/bin/bash
# Wrapper conveniente pro scripts/hf/zip_experiments.py.
#
# Cria conversations.zip por experiment (sob outputs/models/<triple>/<exp>/),
# rebuildando incrementalmente quando algum arquivo dentro de conversations/
# tem mtime maior que o zip. Idempotente — re-rodar é seguro.
#
# Uso:
#   bash dgx/run_zip_experiments.sh                    # default: 8 workers, screen
#   bash dgx/run_zip_experiments.sh --dry-run          # só mostra o que faria
#   FORCE=1 bash dgx/run_zip_experiments.sh            # rebuild todos os zips
#   WORKERS=16 bash dgx/run_zip_experiments.sh         # mais paralelismo
#   FOREGROUND=1 bash dgx/run_zip_experiments.sh       # sem screen
#
# Acompanhar:
#   screen -r zip-exp
#   tail -f logs/zip-experiments-latest.log

umask 002
set -uo pipefail

PROJECT_DIR="${PROJECT_DIR:-/raid/user_danielpedrozo/projects/info-gainme_dev}"
WORKERS="${WORKERS:-8}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d-%H%M%S)}"

# Auto-screen
if [ -z "${STY:-}" ] && [ "${FOREGROUND:-0}" != "1" ]; then
    mkdir -p "${PROJECT_DIR}/logs"
    echo "Iniciando screen 'zip-exp' (run=${RUN_TS})..."
    screen -dmS zip-exp bash -c "RUN_TS='${RUN_TS}' \
        WORKERS='${WORKERS}' FORCE='${FORCE:-}' \
        FOREGROUND=1 bash '${BASH_SOURCE[0]}' $*; exec bash"
    echo "  screen -r zip-exp"
    echo "  tail -f ${PROJECT_DIR}/logs/zip-experiments-latest.log"
    exit 0
fi

# Build flags
FLAGS="--workers ${WORKERS}"
[ "${FORCE:-}" = "1" ] && FLAGS+=" --force"
# Repassa todos os args pro python (--dry-run, --outputs-dir, etc)
FLAGS+=" $*"

# Log com timestamp + symlink "latest"
LOG_FILE="${LOG_FILE:-${PROJECT_DIR}/logs/zip-experiments-${RUN_TS}.log}"
ln -sfn "${LOG_FILE}" "${PROJECT_DIR}/logs/zip-experiments-latest.log"

if [ -z "${__LOG_REDIRECTED__:-}" ]; then
    export __LOG_REDIRECTED__=1
    exec > >(tee -a "${LOG_FILE}") 2>&1
fi

echo "=========================================="
echo "Zip Experiments — RUN ${RUN_TS}"
echo "Project:    ${PROJECT_DIR}"
echo "Workers:    ${WORKERS}"
echo "Force:      ${FORCE:-0}"
echo "Args:       $*"
echo "Log:        ${LOG_FILE}"
echo "Started:    $(date)"
echo "=========================================="

cd "${PROJECT_DIR}"
python3 scripts/hf/zip_experiments.py ${FLAGS}
status=$?

echo "=========================================="
echo "Zip RUN ${RUN_TS} finalizado — $(date) — exit=${status}"
echo "=========================================="

exit ${status}
