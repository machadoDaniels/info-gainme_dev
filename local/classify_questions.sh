#!/bin/bash
# Roda a classificação de perguntas + flatten localmente, dentro de um screen.
#
# Backends suportados (env BACKEND):
#   qwen235b (default) : Qwen3-235B na B200-2 — RECOMENDADO (próprio, rápido)
#   external           : Kimi-K2.6 em 200.137.197.131:60002 (sem tunnel)
#   gptoss             : gpt-oss-120b na H100-02 (compartilhado c/ aluno_daniel)
#   gemma              : gemma-4-31B-it na H100-02 (compartilhado)
#   minimax            : MiniMax-M2.7 na H100-01
#   qwen8b             : Qwen3-8B na B200-1 (pequeno mas livre)
#
# Uso:
#   bash local/classify_questions.sh                       # qwen235b, sweep completo
#   bash local/classify_questions.sh --per-stratum 30      # amostra rápida
#   BACKEND=external bash local/classify_questions.sh      # outro endpoint
#   screen -r classify                                     # acompanhar

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PROJECT_DIR}/.venv/bin/python3"
BACKEND="${BACKEND:-qwen235b}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d-%H%M%S)}"

case "$BACKEND" in
    qwen235b)
        BASE_URL="http://localhost:18026/v1"
        API_KEY="${API_KEY:-EMPTY}"
        MODEL="${MODEL:-Qwen/Qwen3-235B-A22B-Instruct-2507-FP8}"
        TUNNEL="user_danielpedrozo@10.100.0.122:18026:localhost:8026"
        ;;
    external)
        BASE_URL="${BASE_URL:-http://200.137.197.131:60002/v1}"
        API_KEY="${API_KEY:-NINGUEM-TA-PURO-2K26}"
        MODEL="${MODEL:-kimi-k26}"
        TUNNEL=""
        ;;
    gptoss)
        BASE_URL="http://localhost:18836/v1"
        API_KEY="${API_KEY:-vllm_ceia_100}"
        MODEL="${MODEL:-openai/gpt-oss-120b}"
        TUNNEL="dgx-H100-02:18836:localhost:8836"
        ;;
    gemma)
        BASE_URL="http://localhost:18226/v1"
        API_KEY="${API_KEY:-vllm_ceia_100}"
        MODEL="${MODEL:-google/gemma-4-31B-it}"
        TUNNEL="dgx-H100-02:18226:localhost:8226"
        ;;
    minimax)
        BASE_URL="http://localhost:18060/v1"
        API_KEY="${API_KEY:-EMPTY}"
        MODEL="${MODEL:-MiniMaxAI/MiniMax-M2.7}"
        TUNNEL="user_danielpedrozo@10.100.0.111:18060:localhost:8060"
        ;;
    qwen8b)
        BASE_URL="http://localhost:12181/v1"
        API_KEY="${API_KEY:-EMPTY}"
        MODEL="${MODEL:-Qwen3-8B}"
        TUNNEL="dgx-B200-1:12181:localhost:12181"
        ;;
    *)
        echo "BACKEND desconhecido: $BACKEND" >&2; exit 1 ;;
esac

MAX_CONCURRENCY="${MAX_CONCURRENCY:-16}"
EXTRA_FLAGS="${*:-}"

if [ -n "${STY:-}" ]; then
    cd "${PROJECT_DIR}"
    mkdir -p logs
    LOG_FILE="${LOG_FILE:-${PROJECT_DIR}/logs/classify-${BACKEND}-${RUN_TS}.out}"
    ln -sfn "${LOG_FILE}" "${PROJECT_DIR}/logs/classify-latest.out"
    if [ -z "${__LOG_REDIRECTED__:-}" ]; then
        export __LOG_REDIRECTED__=1
        exec > >(tee -a "${LOG_FILE}") 2>&1
    fi
    echo "=== Classificação de perguntas — RUN ${RUN_TS} ==="
    echo "Backend:     ${BACKEND}"
    echo "Endpoint:    ${BASE_URL}"
    echo "Modelo:      ${MODEL}"
    echo "Concurrency: ${MAX_CONCURRENCY}"
    echo "Flags:       ${EXTRA_FLAGS:-(nenhuma)}"
    echo "Log:         ${LOG_FILE}"
    echo "Started:     $(date)"

    if [ -n "$TUNNEL" ]; then
        IFS=':' read -r host lport rhost rport <<< "$TUNNEL"
        echo "Tunnel:      $host  ($lport -> $rhost:$rport)"
        pkill -f "ssh -fN -L ${lport}:" 2>/dev/null || true
        sleep 1
        ssh -fN -L "${lport}:${rhost}:${rport}" "$host"
        trap "pkill -f 'ssh -fN -L ${lport}:' 2>/dev/null || true" EXIT
        sleep 2
    fi
    echo "============================================"

    "${PYTHON}" scripts/question_classification/classify_questions.py \
        --base-url  "${BASE_URL}" \
        --api-key   "${API_KEY}" \
        --model     "${MODEL}" \
        --max-concurrency "${MAX_CONCURRENCY}" \
        ${EXTRA_FLAGS}

    echo ""
    echo "--- Gerando CSV ---"
    "${PYTHON}" scripts/question_classification/flatten_question_classifications.py

    echo ""
    echo "=== Pronto — $(date) ==="
    echo "CSV: ${PROJECT_DIR}/outputs/question_classifications.csv"
else
    mkdir -p "${PROJECT_DIR}/logs"
    echo "Iniciando screen 'classify' (backend=${BACKEND}, run=${RUN_TS})..."
    # script faz seu próprio log timestamped (não precisa de tee aqui)
    screen -dmS classify bash -c "RUN_TS='${RUN_TS}' BACKEND='${BACKEND}' bash '${BASH_SOURCE[0]}' ${EXTRA_FLAGS}; exec bash"
    echo "  screen -r classify"
    echo "  tail -f ${PROJECT_DIR}/logs/classify-latest.out"
fi
