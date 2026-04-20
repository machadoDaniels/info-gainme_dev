#!/bin/bash
# Roda o classificador de perguntas (scripts/question_classification/classify_questions.py) na DGX via Singularity.
# Sem SLURM — pensado para ser executado dentro de um screen.
#
# Lançamento típico:
#   screen -dmS classify bash -c \
#     'bash dgx/run_classify_questions_screen.sh 2>&1 | tee logs/classify-all.out; exec bash'
#
# Acompanhar:
#   screen -r classify
#   tail -f logs/classify-all.out
#   watch -n 10 'find outputs/question_classification -name classification.json | wc -l'
#
# Variáveis de ambiente (com defaults):
#   BASE_URL         endpoint vLLM/OpenAI-compatible   (default: http://200.137.197.131:60002/v1)
#   API_KEY          bearer token                       (default: NINGUEM-TA-PURO-2K26)
#   MODEL            nome do modelo no endpoint         (default: nvidia/Kimi-K2.5-NVFP4)
#   PER_STRATUM      conversas por stratum              (default: 99999 — todas)
#   MAX_CONCURRENCY  chamadas LLM simultâneas           (default: 32)
#   NO_THINKING      "1" desativa modo de reasoning     (default: vazio = thinking ligado)
#   FORCE            "1" re-classifica arquivos prontos (default: vazio)
#   SEED             seed da amostragem estratificada   (default: 42)

umask 002

PROJECT_DIR="/raid/user_danielpedrozo/projects/info-gainme_dev"
SINGULARITY_IMAGE="/raid/user_danielpedrozo/images/vllm_openai_latest.sif"

BASE_URL="${BASE_URL:-http://200.137.197.131:60002/v1}"
API_KEY="${API_KEY:-NINGUEM-TA-PURO-2K26}"
MODEL="${MODEL:-nvidia/Kimi-K2.5-NVFP4}"
PER_STRATUM="${PER_STRATUM:-99999}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-32}"
SEED="${SEED:-42}"

EXTRA_FLAGS=""
[[ "${NO_THINKING}" == "1" ]] && EXTRA_FLAGS+=" --no-thinking"
[[ "${FORCE}" == "1" ]]       && EXTRA_FLAGS+=" --force"

mkdir -p "${PROJECT_DIR}/logs"
mkdir -p "${PROJECT_DIR}/outputs/question_classification"

echo "=========================================="
echo "Question Classification - $(date)"
echo "Project:         ${PROJECT_DIR}"
echo "Endpoint:        ${BASE_URL}"
echo "Model:           ${MODEL}"
echo "Per-stratum:     ${PER_STRATUM}"
echo "Max concurrency: ${MAX_CONCURRENCY}"
echo "Seed:            ${SEED}"
echo "Flags:           ${EXTRA_FLAGS:-(default: thinking on, resume on)}"
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
echo "Classificação finalizada - $(date)"
echo "Resultado em: ${PROJECT_DIR}/outputs/question_classification/summary.json"
echo "=========================================="
