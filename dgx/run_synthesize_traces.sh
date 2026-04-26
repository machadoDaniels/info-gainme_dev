#!/bin/bash
# Roda síntese de reasoning traces direto na DGX (sem SLURM).
#
# Cada run gera log timestamped em logs/traces-<backend>-<YYYYMMDD-HHMMSS>.out
# + atualiza o symlink logs/traces-latest.out.
#
# Uso:
#   bash dgx/run_synthesize_traces.sh                   # todos os runs.csv (--all)
#   bash dgx/run_synthesize_traces.sh path/to/runs.csv  # CSV específico
#   BACKEND=gptoss bash dgx/run_synthesize_traces.sh    # outro modelo
#
# Acompanhar:
#   screen -r traces
#   tail -f logs/traces-latest.out

umask 002

# Tag única deste run.
RUN_TS="${RUN_TS:-$(date +%Y%m%d-%H%M%S)}"

PROJECT_DIR="/raid/user_danielpedrozo/projects/info-gainme_dev"
SINGULARITY_IMAGE="/raid/user_danielpedrozo/images/vllm_openai_latest.sif"

RUNS_PATH="${1:-}"
WORKERS="${WORKERS:-8}"
TURN_WORKERS="${TURN_WORKERS:-4}"

# Default: Qwen3-235B no B200-2 (acesso direto na rede DGX). Override via BACKEND ou
# BASE_URL/MODEL/API_KEY direto.
BACKEND="${BACKEND:-qwen235b}"
case "$BACKEND" in
    qwen235b)
        BASE_URL="${BASE_URL:-http://10.100.0.122:8026/v1}"
        API_KEY="${API_KEY:-EMPTY}"
        MODEL="${MODEL:-Qwen/Qwen3-235B-A22B-Instruct-2507-FP8}"
        ;;
    external)
        BASE_URL="${BASE_URL:-http://200.137.197.131:60002/v1}"
        API_KEY="${API_KEY:-NINGUEM-TA-PURO-2K26}"
        MODEL="${MODEL:-kimi-k26}"
        ;;
    gptoss)
        BASE_URL="${BASE_URL:-http://10.100.0.112:8836/v1}"
        API_KEY="${API_KEY:-vllm_ceia_100}"
        MODEL="${MODEL:-openai/gpt-oss-120b}"
        ;;
    openai|gpt-4o-mini)
        BASE_URL="${BASE_URL:-}"
        API_KEY="${API_KEY:-${OPENAI_API_KEY:-}}"
        MODEL="${MODEL:-gpt-4o-mini}"
        ;;
    *) echo "BACKEND desconhecido: $BACKEND" >&2; exit 1 ;;
esac

MODEL_ARGS="--model '${MODEL}' --workers ${WORKERS} --turn-workers ${TURN_WORKERS}"
[ -n "${BASE_URL}" ] && MODEL_ARGS="${MODEL_ARGS} --base-url '${BASE_URL}'"
[ -n "${API_KEY}" ]  && MODEL_ARGS="${MODEL_ARGS} --api-key '${API_KEY}'"

# Log timestamped + symlink "latest". Override via LOG_FILE.
mkdir -p "${PROJECT_DIR}/logs"
LOG_FILE="${LOG_FILE:-${PROJECT_DIR}/logs/traces-${BACKEND}-${RUN_TS}.out}"
ln -sfn "${LOG_FILE}" "${PROJECT_DIR}/logs/traces-latest.out"
if [ -z "${__LOG_REDIRECTED__:-}" ]; then
    export __LOG_REDIRECTED__=1
    exec > >(tee -a "${LOG_FILE}") 2>&1
fi

echo "=========================================="
echo "Reasoning Traces Synthesis — RUN ${RUN_TS}"
echo "Backend:  ${BACKEND}"
echo "Modelo:   ${MODEL}"
echo "Endpoint: ${BASE_URL:-(default)}"
echo "Workers:  ${WORKERS} conversas × ${TURN_WORKERS} turns = $(( WORKERS * TURN_WORKERS )) LLM calls max"
echo "Log:      ${LOG_FILE}"
echo "Started:  $(date)"
if [ -n "${RUNS_PATH}" ]; then
    echo "CSV:      ${RUNS_PATH}"
    SYNTHESIS_CMD="python3 scripts/reasoning_traces/synthesize_traces.py --runs '${RUNS_PATH}' ${MODEL_ARGS}"
else
    echo "Modo:     --all (todos os runs.csv sob outputs/)"
    SYNTHESIS_CMD="python3 scripts/reasoning_traces/synthesize_traces.py --all ${MODEL_ARGS}"
fi
echo "=========================================="

sg sd22 -c "
    singularity exec \
        --bind /raid/user_danielpedrozo:/workspace \
        --pwd /workspace/projects/info-gainme_dev \
        '${SINGULARITY_IMAGE}' \
        bash -c \"${SYNTHESIS_CMD}\"
"

echo "=========================================="
echo "Síntese RUN ${RUN_TS} finalizada — $(date)"
echo "=========================================="
