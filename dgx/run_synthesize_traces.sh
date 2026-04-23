#!/bin/bash
# Roda síntese de reasoning traces direto na DGX (sem SLURM)
# Uso:
#   ./run_synthesize_traces.sh                          # processa todos os runs.csv
#   ./run_synthesize_traces.sh path/to/runs.csv         # processa um CSV específico
#   ./run_synthesize_traces.sh --model Qwen3-8B         # modelo específico

umask 002

PROJECT_DIR="/raid/user_danielpedrozo/projects/info-gainme_dev"
SINGULARITY_IMAGE="/raid/user_danielpedrozo/images/vllm_openai_latest.sif"

RUNS_PATH="${1:-}"
MODEL="${MODEL:-gpt-4o-mini}"
WORKERS="${WORKERS:-8}"
TURN_WORKERS="${TURN_WORKERS:-4}"

# Monta args opcionais
MODEL_ARGS="--model '${MODEL}' --workers ${WORKERS} --turn-workers ${TURN_WORKERS}"
if [ -n "${BASE_URL}" ]; then
    MODEL_ARGS="${MODEL_ARGS} --base-url '${BASE_URL}'"
fi

echo "=========================================="
echo "Reasoning Traces Synthesis - $(date)"
echo "Modelo: ${MODEL}"
echo "Workers: ${WORKERS} conversas × ${TURN_WORKERS} turns = $(( WORKERS * TURN_WORKERS )) LLM calls max"
if [ -n "${RUNS_PATH}" ]; then
    echo "CSV: ${RUNS_PATH}"
    SYNTHESIS_CMD="python3 scripts/reasoning_traces/synthesize_traces.py --runs '${RUNS_PATH}' ${MODEL_ARGS}"
else
    echo "Modo: --all (todos os runs.csv sob outputs/)"
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
echo "Síntese finalizada - $(date)"
echo "=========================================="
