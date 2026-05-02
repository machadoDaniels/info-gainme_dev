#!/bin/bash
# Roda scripts/judge_eval/evaluate.py contra um endpoint vLLM já aberto
# (sem alocar GPU/SLURM próprio). Análogo a run_evaluate_choices_screen.sh.
#
# Uso típico (cria screen automaticamente):
#   bash dgx/run_judge_eval_screen.sh
#
# Forçar foreground (sem screen):
#   FOREGROUND=1 bash dgx/run_judge_eval_screen.sh
#
# Acompanhar:
#   screen -r judge-eval
#   tail -f logs/judge-eval-latest.out
#
# Variáveis de ambiente (com defaults):
#   BACKEND        gpt_oss_h2 (default — h2:8836, gpt-oss-120b, max_len=12k)
#                  | qwen3_8b_h3 (h3:8800)
#   BASE_URL/API_KEY/MODEL   override manual do backend selecionado
#   WHAT           both (default) | oracle | pruner
#   RUN_INDEX      filtro de run_index    (default: 1, só run01)
#   SAMPLE_INDICES amostras espaçadas     (default: 10,20,30,40,50,60,70,80,90)
#   WORKERS        conversations em paralelo (default: 8)
#   TURN_WORKERS   turns em paralelo por conv (default: 4)
#   FORCE          "1" sobrescreve resultados existentes (default: vazio)
#
# Argumento posicional opcional:
#   $1   path para runs.csv específico OU pasta de conversation. Se omitido,
#        processa todos via --all.

umask 002

RUN_TS="${RUN_TS:-$(date +%Y%m%d-%H%M%S)}"

PROJECT_DIR="/raid/user_danielpedrozo/projects/info-gainme_dev"
SINGULARITY_IMAGE="/raid/user_danielpedrozo/images/vllm_openai_latest.sif"

# Auto-screen
if [ -z "${STY:-}" ] && [ "${FOREGROUND:-0}" != "1" ]; then
    mkdir -p "${PROJECT_DIR}/logs"
    echo "Iniciando screen 'judge-eval' (run=${RUN_TS})..."
    screen -dmS judge-eval bash -c "RUN_TS='${RUN_TS}' \
        BACKEND='${BACKEND:-}' BASE_URL='${BASE_URL:-}' API_KEY='${API_KEY:-}' MODEL='${MODEL:-}' \
        WHAT='${WHAT:-}' RUN_INDEX='${RUN_INDEX:-}' SAMPLE_INDICES='${SAMPLE_INDICES:-}' \
        WORKERS='${WORKERS:-}' TURN_WORKERS='${TURN_WORKERS:-}' FORCE='${FORCE:-}' \
        bash '${BASH_SOURCE[0]}' ${1:-}; exec bash"
    echo "  screen -r judge-eval"
    echo "  tail -f ${PROJECT_DIR}/logs/judge-eval-latest.out"
    exit 0
fi

TARGET_PATH="${1:-}"
WHAT="${WHAT:-both}"
WORKERS="${WORKERS:-8}"
TURN_WORKERS="${TURN_WORKERS:-4}"
RUN_INDEX="${RUN_INDEX-1}"
SAMPLE_INDICES="${SAMPLE_INDICES-10,20,30,40,50,60,70,80,90}"

# Default: gpt-oss-120b em h2:8836 (aluno_daniel/cemig_grpo, max_len=12000).
BACKEND="${BACKEND:-gpt_oss_h2}"
case "$BACKEND" in
    gpt_oss_h2)
        BASE_URL="${BASE_URL:-http://10.100.0.112:8836/v1}"
        API_KEY="${API_KEY:-vllm_ceia_100}"
        MODEL="${MODEL:-openai/gpt-oss-120b}"
        ;;
    qwen3_8b_h3)
        BASE_URL="${BASE_URL:-http://10.100.0.113:8800/v1}"
        API_KEY="${API_KEY:-EMPTY}"
        MODEL="${MODEL:-Qwen3-8B}"
        ;;
    *) echo "BACKEND desconhecido: $BACKEND" >&2; exit 1 ;;
esac

case "$WHAT" in
    oracle|pruner) TARGETS=("$WHAT") ;;
    both)          TARGETS=(oracle pruner) ;;
    *) echo "WHAT inválido: $WHAT (use oracle|pruner|both)" >&2; exit 1 ;;
esac

# Decidir flag para o evaluate.py: --runs / --conversation / --all
if [ -z "${TARGET_PATH}" ]; then
    TARGET_FLAG="--all"
elif [ -d "${TARGET_PATH}" ]; then
    TARGET_FLAG="--conversation '${TARGET_PATH}'"
elif [ -f "${TARGET_PATH}" ]; then
    TARGET_FLAG="--runs '${TARGET_PATH}'"
else
    echo "ERROR: TARGET_PATH '${TARGET_PATH}' não é arquivo nem pasta válida" >&2
    exit 2
fi

EXTRA_FLAGS=""
[[ -n "${RUN_INDEX}" ]]      && EXTRA_FLAGS+=" --run-index ${RUN_INDEX}"
[[ -n "${SAMPLE_INDICES}" ]] && EXTRA_FLAGS+=" --sample-indices ${SAMPLE_INDICES}"
[[ "${FORCE}" == "1" ]]      && EXTRA_FLAGS+=" --force"

mkdir -p "${PROJECT_DIR}/logs"

LOG_FILE="${LOG_FILE:-${PROJECT_DIR}/logs/judge-eval-${BACKEND}-${RUN_TS}.out}"
ln -sfn "${LOG_FILE}" "${PROJECT_DIR}/logs/judge-eval-latest.out"

if [ -z "${__LOG_REDIRECTED__:-}" ]; then
    export __LOG_REDIRECTED__=1
    exec > >(tee -a "${LOG_FILE}") 2>&1
fi

echo "=========================================="
echo "Judge Evaluation — RUN ${RUN_TS}"
echo "Backend:       ${BACKEND}"
echo "Endpoint:      ${BASE_URL}"
echo "Judge model:   ${MODEL}"
echo "What:          ${WHAT} (${TARGETS[*]})"
echo "Workers:       ${WORKERS} × turn-workers ${TURN_WORKERS}"
echo "Filters:       run_index=${RUN_INDEX:-all} sample_indices=${SAMPLE_INDICES:-all}"
echo "Target:        ${TARGET_PATH:-(all CoT runs.csv)}"
echo "Log:           ${LOG_FILE}"
echo "Started:       $(date)"
echo "=========================================="

EVAL_CMDS=""
for t in "${TARGETS[@]}"; do
    EVAL_CMDS+="echo ''; echo \"[\$(date '+%H:%M:%S')] --target ${t}\"; "
    EVAL_CMDS+="python3 scripts/judge_eval/evaluate.py --target ${t} ${TARGET_FLAG} \
        --judge-model '${MODEL}' \
        --base-url '${BASE_URL}' \
        --api-key '${API_KEY}' \
        --workers ${WORKERS} --turn-workers ${TURN_WORKERS}${EXTRA_FLAGS}; "
done
EVAL_CMDS+="python3 scripts/judge_eval/aggregate_judge_results.py"

sg sd22 -c "
    singularity exec \
        --bind /raid/user_danielpedrozo:/workspace \
        --pwd /workspace/projects/info-gainme_dev \
        '${SINGULARITY_IMAGE}' \
        bash -c \"
            pip install --quiet --user -r requirements.txt
            ${EVAL_CMDS}
        \"
"

echo "=========================================="
echo "Judge eval RUN ${RUN_TS} finalizado — $(date)"
echo "=========================================="
