#!/bin/bash
# Roda síntese + análise de traces de raciocínio localmente, dentro de um screen.
#
# Uso:
#   bash local/synthesize_traces.sh                       # tudo (--all, retomável)
#   bash local/synthesize_traces.sh --runs path/to/runs.csv
#   screen -r traces                                      # acompanhar
#
# OBS: só funciona em conversas CoT que tenham seeker.json local. As pastas
# CoT podem estar vazias se você ainda não sincronizou da DGX:
#   rsync -av <node>:.../outputs/models/<triple>/<exp>/ outputs/models/<triple>/<exp>/

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PROJECT_DIR}/.venv/bin/python3"
RUN_TS="${RUN_TS:-$(date +%Y%m%d-%H%M%S)}"
BACKEND="${BACKEND:-external}"

BASE_URL="${BASE_URL:-http://200.137.197.131:60002/v1}"
API_KEY="${API_KEY:-NINGUEM-TA-PURO-2K26}"
MODEL="${MODEL:-kimi-k26}"
WORKERS="${WORKERS:-8}"
TURN_WORKERS="${TURN_WORKERS:-4}"

# Default: --all. Se passar argumentos, eles substituem.
ARGS="${@:---all}"

if [ -n "${STY:-}" ]; then
    mkdir -p "${PROJECT_DIR}/logs"
    LOG_FILE="${LOG_FILE:-${PROJECT_DIR}/logs/traces-${BACKEND}-${RUN_TS}.out}"
    ln -sfn "${LOG_FILE}" "${PROJECT_DIR}/logs/traces-latest.out"
    if [ -z "${__LOG_REDIRECTED__:-}" ]; then
        export __LOG_REDIRECTED__=1
        exec > >(tee -a "${LOG_FILE}") 2>&1
    fi
    echo "=== Síntese de traces — RUN ${RUN_TS} ==="
    echo "Backend:  ${BACKEND}"
    echo "Endpoint: ${BASE_URL}"
    echo "Modelo:   ${MODEL}"
    echo "Workers:  ${WORKERS} conv x ${TURN_WORKERS} turns"
    echo "Args:     ${ARGS}"
    echo "Log:      ${LOG_FILE}"
    echo "Started:  $(date)"
    echo "===================================="

    cd "${PROJECT_DIR}"

    "${PYTHON}" scripts/reasoning_traces/synthesize_traces.py \
        ${ARGS} \
        --base-url "${BASE_URL}" \
        --api-key  "${API_KEY}" \
        --model    "${MODEL}" \
        --workers  "${WORKERS}" \
        --turn-workers "${TURN_WORKERS}"

    echo ""
    echo "--- Agregando ---"
    "${PYTHON}" scripts/reasoning_traces/analyze_traces.py

    echo ""
    echo "=== Pronto — $(date) ==="
    echo "JSONL:    ${PROJECT_DIR}/outputs/seeker_traces.jsonl"
    echo "Resumo:   ${PROJECT_DIR}/outputs/reasoning_traces_analysis.json"
else
    mkdir -p "${PROJECT_DIR}/logs"
    echo "Iniciando screen 'traces' (backend=${BACKEND}, run=${RUN_TS})..."
    screen -dmS traces bash -c "RUN_TS='${RUN_TS}' BACKEND='${BACKEND}' bash '${BASH_SOURCE[0]}' ${ARGS}; exec bash"
    echo "  screen -r traces"
    echo "  tail -f ${PROJECT_DIR}/logs/traces-latest.out"
fi
