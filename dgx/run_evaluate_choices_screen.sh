#!/bin/bash
# Roda evaluate_all_seeker_choices.py — re-simula Oracle+Pruner em cada pergunta
# considerada pelo Seeker (extraída dos seeker_traces.json) para calcular o IG
# contrafactual de cada candidata.
#
# Diferente de run_judge_eval.sh: aquele usa um Judge LLM externo (gpt-oss-120b)
# pra dar nota de qualidade; este re-roda Oracle+Pruner reais (Qwen3-8B por
# default) e mede ganho de informação direto.
#
# Lançamento típico (script gera log timestamped automaticamente):
#   screen -dmS eval-choices bash -c 'bash dgx/run_evaluate_choices_screen.sh; exec bash'
#
# Cada run grava em logs/eval-choices-<backend>-<YYYYMMDD-HHMMSS>.out + atualiza
# logs/eval-choices-latest.out (symlink) pra facilitar tail -f.
#
# Acompanhar:
#   screen -r eval-choices
#   tail -f logs/eval-choices-latest.out
#
# Variáveis de ambiente (com defaults):
#   BACKEND       qwen3_8b_h2 (default — H100-02 :8461) | qwen3_8b_h3 (H100-03 :9200, raro)
#   BASE_URL/API_KEY/MODEL   override manual do backend selecionado
#   MAX_WORKERS   conversations em paralelo            (default: 8)
#   FORCE         "1" re-avalia mesmo com question_evaluation.json válido (default: vazio)
#   DRY_RUN       "1" só lista o que seria processado  (default: vazio)
#   TEMPERATURE   override de temperatura              (default: vazio = API decide)
#
# Argumento posicional opcional:
#   $1   path para runs.csv específico. Se omitido, processa todos via --all.

umask 002

# Tag única deste run — vai pro nome do log e pro header.
RUN_TS="${RUN_TS:-$(date +%Y%m%d-%H%M%S)}"

PROJECT_DIR="/raid/user_danielpedrozo/projects/info-gainme_dev"
SINGULARITY_IMAGE="/raid/user_danielpedrozo/images/vllm_openai_latest.sif"

RUNS_PATH="${1:-}"
MAX_WORKERS="${MAX_WORKERS:-8}"

# Default: Qwen3-8B no H100-02 :8461 (sem auth, served-model-name "Qwen3-8B").
# Alternativa: BACKEND=qwen3_8b_h3 (H100-03 :9200, atualmente off mas mantido por
# compat com configs/servers.yaml caso volte).
BACKEND="${BACKEND:-qwen3_8b_h2}"
case "$BACKEND" in
    qwen3_8b_h2)
        BASE_URL="${BASE_URL:-http://10.100.0.112:8461/v1}"
        API_KEY="${API_KEY:-EMPTY}"
        MODEL="${MODEL:-Qwen3-8B}"
        ;;
    qwen3_8b_h3)
        BASE_URL="${BASE_URL:-http://10.100.0.113:9200/v1}"
        API_KEY="${API_KEY:-EMPTY}"
        MODEL="${MODEL:-Qwen3-8B}"
        ;;
    *) echo "BACKEND desconhecido: $BACKEND" >&2; exit 1 ;;
esac

EXTRA_FLAGS=""
[[ "${FORCE}" == "1" ]]        && EXTRA_FLAGS+=" --force"
[[ "${DRY_RUN}" == "1" ]]      && EXTRA_FLAGS+=" --dry-run"
[[ -n "${TEMPERATURE:-}" ]]    && EXTRA_FLAGS+=" --temperature ${TEMPERATURE}"

mkdir -p "${PROJECT_DIR}/logs"

# Log timestamped + symlink "latest". Override via LOG_FILE=/path/to/log.
LOG_FILE="${LOG_FILE:-${PROJECT_DIR}/logs/eval-choices-${BACKEND}-${RUN_TS}.out}"
ln -sfn "${LOG_FILE}" "${PROJECT_DIR}/logs/eval-choices-latest.out"

# Re-exec redirecionando stdout+stderr pro log (e ainda mostra na tela via tee).
# Marcador __LOG_REDIRECTED__ evita loop se o re-exec rodar de novo.
if [ -z "${__LOG_REDIRECTED__:-}" ]; then
    export __LOG_REDIRECTED__=1
    exec > >(tee -a "${LOG_FILE}") 2>&1
fi

echo "=========================================="
echo "Question Choice Evaluation — RUN ${RUN_TS}"
echo "Backend:       ${BACKEND}"
echo "Endpoint:      ${BASE_URL}"
echo "Oracle/Pruner: ${MODEL}"
echo "Max workers:   ${MAX_WORKERS}"
echo "Flags:         ${EXTRA_FLAGS:-(default: resume on)}"
echo "Log:           ${LOG_FILE}"
echo "Started:       $(date)"
if [ -n "${RUNS_PATH}" ]; then
    echo "CSV:           ${RUNS_PATH}"
    EVAL_TARGET="'${RUNS_PATH}'"
else
    echo "Mode:          --all (todos os CoT runs.csv sob outputs/)"
    EVAL_TARGET="--all"
fi
echo "=========================================="

EVAL_CMD="python3 scripts/reasoning_traces/evaluate_all_seeker_choices.py \
    ${EVAL_TARGET} \
    --oracle-model '${MODEL}' \
    --pruner-model '${MODEL}' \
    --base-url '${BASE_URL}' \
    --api-key '${API_KEY}' \
    --max-workers ${MAX_WORKERS}${EXTRA_FLAGS}"

sg sd22 -c "
    singularity exec \
        --bind /raid/user_danielpedrozo:/workspace \
        --pwd /workspace/projects/info-gainme_dev \
        '${SINGULARITY_IMAGE}' \
        bash -c \"
            pip install --quiet --user -r requirements.txt
            ${EVAL_CMD}
        \"
"

echo "=========================================="
echo "Avaliação RUN ${RUN_TS} finalizada — $(date)"
echo "=========================================="
