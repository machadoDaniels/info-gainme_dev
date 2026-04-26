#!/bin/bash
# Roda o classificador de perguntas (scripts/question_classification/classify_questions.py) na DGX via Singularity.
# Sem SLURM — pensado para ser executado dentro de um screen.
#
# Lançamento típico (script gera log timestamped automaticamente):
#   screen -dmS classify bash -c 'bash dgx/run_classify_questions_screen.sh; exec bash'
#
# Cada run grava em logs/classify-<backend>-<YYYYMMDD-HHMMSS>.out + atualiza
# logs/classify-latest.out (symlink) pra facilitar tail -f.
#
# Acompanhar:
#   screen -r classify
#   tail -f logs/classify-latest.out
#   watch -n 10 'wc -l outputs/question_classifications.jsonl'
#
# Variáveis de ambiente (com defaults):
#   BACKEND          qwen235b (default) | external | gptoss | minimax
#   BASE_URL/API_KEY/MODEL   override manual do backend selecionado
#   PER_STRATUM      conversas por stratum              (default: 99999 — todas)
#   MAX_CONCURRENCY  chamadas LLM simultâneas           (default: 32)
#   NO_THINKING      "1" desativa modo de reasoning     (default: vazio = thinking ligado)
#   FORCE            "1" re-classifica arquivos prontos (default: vazio)
#   SEED             seed da amostragem estratificada   (default: 42)

umask 002

# Tag única deste run — vai pro nome do log e pro header.
RUN_TS="${RUN_TS:-$(date +%Y%m%d-%H%M%S)}"

PROJECT_DIR="/raid/user_danielpedrozo/projects/info-gainme_dev"
SINGULARITY_IMAGE="/raid/user_danielpedrozo/images/vllm_openai_latest.sif"

# Default: Qwen3-235B-Instruct rodando no B200-2 (acesso direto na rede interna).
# Alternativas: BACKEND=external (Kimi-K2.6 público) ou BACKEND=gptoss (gpt-oss-120b H100-02).
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
    minimax)
        BASE_URL="${BASE_URL:-http://10.100.0.111:8060/v1}"
        API_KEY="${API_KEY:-EMPTY}"
        MODEL="${MODEL:-MiniMaxAI/MiniMax-M2.7}"
        ;;
    *) echo "BACKEND desconhecido: $BACKEND" >&2; exit 1 ;;
esac
PER_STRATUM="${PER_STRATUM:-99999}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-32}"
SEED="${SEED:-42}"

EXTRA_FLAGS=""
[[ "${NO_THINKING}" == "1" ]] && EXTRA_FLAGS+=" --no-thinking"
[[ "${FORCE}" == "1" ]]       && EXTRA_FLAGS+=" --force"

mkdir -p "${PROJECT_DIR}/logs"
mkdir -p "${PROJECT_DIR}/outputs/question_classification"

# Cada run tem seu próprio log timestamped + symlink "latest" pra fácil tail.
# Override via LOG_FILE=/path/to/log se quiser.
LOG_FILE="${LOG_FILE:-${PROJECT_DIR}/logs/classify-${BACKEND}-${RUN_TS}.out}"
ln -sfn "${LOG_FILE}" "${PROJECT_DIR}/logs/classify-latest.out"

# Re-exec redirecionando stdout+stderr pro log (e ainda mostra na tela via tee).
# Marcador __LOG_REDIRECTED__ evita loop se o re-exec rodar de novo.
if [ -z "${__LOG_REDIRECTED__:-}" ]; then
    export __LOG_REDIRECTED__=1
    exec > >(tee -a "${LOG_FILE}") 2>&1
fi

echo "=========================================="
echo "Question Classification — RUN ${RUN_TS}"
echo "Backend:         ${BACKEND}"
echo "Endpoint:        ${BASE_URL}"
echo "Model:           ${MODEL}"
echo "Per-stratum:     ${PER_STRATUM}"
echo "Max concurrency: ${MAX_CONCURRENCY}"
echo "Seed:            ${SEED}"
echo "Flags:           ${EXTRA_FLAGS:-(default: thinking on, resume on)}"
echo "Log:             ${LOG_FILE}"
echo "Started:         $(date)"
echo "=========================================="

CLASSIFY_CMD="python3 scripts/question_classification/classify_questions.py \
    --base-url '${BASE_URL}' \
    --api-key  '${API_KEY}' \
    --model    '${MODEL}' \
    --per-stratum ${PER_STRATUM} \
    --max-concurrency ${MAX_CONCURRENCY} \
    --seed ${SEED}${EXTRA_FLAGS}"

sg sd22 -c "
    singularity exec \
        --bind /raid/user_danielpedrozo:/workspace \
        --pwd /workspace/projects/info-gainme_dev \
        '${SINGULARITY_IMAGE}' \
        bash -c \"
            pip install --quiet --user -r requirements.txt
            ${CLASSIFY_CMD}
        \"
"

echo "=========================================="
echo "Classificação RUN ${RUN_TS} finalizada — $(date)"
echo "Resultado em: ${PROJECT_DIR}/outputs/question_classification/summary.json"
echo "=========================================="
