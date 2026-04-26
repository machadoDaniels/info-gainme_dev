#!/bin/bash
#SBATCH --job-name=info-gainme-judge
#SBATCH --partition=b200n1
#SBATCH --gres=gpu:2
#SBATCH --mem=120G
#SBATCH --time=2-00:00:00
#SBATCH --output=/raid/user_danielpedrozo/projects/info-gainme_dev/logs/%x-%j.log

# ==================================================================
# Self-contained judge evaluation:
#   1) Spins up a vLLM server for the judge model (default gpt-oss-120b).
#   2) Runs scripts/judge_eval/evaluate_oracle.py and evaluate_pruner.py
#      against TARGET (a conversation dir, a runs.csv, or "all").
#   3) Kills the vLLM and cleans up the override file.
#
# Usage (overrides via --export=ALL,KEY=VAL,... — NEVER as positional args):
#   sbatch --partition=b200n1 --gres=gpu:2 \
#     --export=ALL,JUDGE_MODEL=openai/gpt-oss-120b,JUDGE_MODEL_NAME=gpt-oss-120b,\
#              TARGET=configs/full/8b/diseases_160_8b_po_cot.yaml \
#     dgx/run_judge_eval.sh
#
# TARGET can be:
#   - a runs.csv path (most common)            → passed as --runs
#   - a conversation directory                  → --conversation
#   - the literal string "all"                  → --all
# ==================================================================

umask 002
set -o pipefail

# ---------------- configuration ----------------
JUDGE_MODEL="${JUDGE_MODEL:-openai/gpt-oss-120b}"       # HF id / served model
JUDGE_MODEL_NAME="${JUDGE_MODEL_NAME:-gpt-oss-120b}"    # --served-model-name
JUDGE_GPU_MEM="${JUDGE_GPU_MEM:-0.90}"
JUDGE_MAX_LEN="${JUDGE_MAX_LEN:-65536}"
JUDGE_TP_SIZE="${JUDGE_TP_SIZE:-2}"                    # tensor-parallel-size
# GPT-OSS uses Harmony; vLLM 0.16+ registers the parser as "openai_gptoss".
# Older docs mention "gptoss" — name varies by release; check
# `vllm serve --help | grep reasoning-parser` if it errors KeyError.
JUDGE_REASONING_PARSER="${JUDGE_REASONING_PARSER:-openai_gptoss}"
JUDGE_EXTRA_ARGS="${JUDGE_EXTRA_ARGS:-}"               # any extra vllm flags

TARGET="${TARGET:-all}"                                 # runs.csv | conv dir | "all"
WHAT="${WHAT:-both}"                                    # oracle | pruner | both
WORKERS="${WORKERS:-8}"
TURN_WORKERS="${TURN_WORKERS:-4}"

# Sampling defaults: piloto = primeiro run por target, 9 alvos espaçados (10, 20, ..., 90).
# Override:
#   - via env exportada antes do sbatch: export SAMPLE_INDICES=0,10,20,...
#   - via --export inline: SAMPLE_INDICES=0_10_20_... (vírgulas quebram o parser do
#     SLURM --export; underscores são convertidos em vírgulas aqui).
#   - desligar amostragem: passar RUN_INDEX="" e SAMPLE_INDICES="".
RUN_INDEX="${RUN_INDEX-1}"
SAMPLE_INDICES="${SAMPLE_INDICES-10,20,30,40,50,60,70,80,90}"
SAMPLE_INDICES="${SAMPLE_INDICES//_/,}"

PROJECT_DIR="/raid/user_danielpedrozo/projects/info-gainme_dev"
SHARED_GROUP="sd22"
SINGULARITY_IMAGE="/raid/user_danielpedrozo/images/vllm_openai_latest.sif"

export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-32}"
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-3600}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"
export HF_HOME=/workspace/hf-cache
source "${PROJECT_DIR}/.env"
export HF_TOKEN="${HF_TOKEN:?HF_TOKEN não definido no .env}"

# gpt-oss-120b hangs in CUDA graph capture on vLLM 0.16 + MXFP4 (intermittent
# deadlock during shm_broadcast under concurrent traffic). Force eager always
# for the judge — output is short JSON (~500 tokens), CUDA graphs don't matter
# for throughput here. Override with VLLM_ENFORCE_EAGER=false if you've
# verified the bug is fixed in your vLLM version.
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-true}"

mkdir -p "${PROJECT_DIR}/logs" "${PROJECT_DIR}/hf-cache"
LOGS_DIR_HOST="${PROJECT_DIR}/logs"
cd "${PROJECT_DIR}"

# ---------------- port selection ----------------
BASE_PORT=$((8000 + (SLURM_JOB_ID % 500) * 10))
port_in_use() { ss -tln 2>/dev/null | awk '{print $4}' | grep -qE ":$1$"; }
JUDGE_PORT="${JUDGE_PORT:-$BASE_PORT}"
while port_in_use "${JUDGE_PORT}"; do
    JUDGE_PORT=$((JUDGE_PORT + 1))
done

# ---------------- GPU allocation ----------------
if [ -z "${CUDA_VISIBLE_DEVICES}" ]; then
    echo "ERROR: CUDA_VISIBLE_DEVICES not set by SLURM"
    exit 1
fi
IFS=',' read -ra GPU_ARRAY <<< "${CUDA_VISIBLE_DEVICES}"
TOTAL_GPUS=${#GPU_ARRAY[@]}
if [ "${TOTAL_GPUS}" -lt "${JUDGE_TP_SIZE}" ]; then
    echo "ERROR: JUDGE_TP_SIZE=${JUDGE_TP_SIZE} but only ${TOTAL_GPUS} GPUs allocated"
    exit 1
fi
JUDGE_GPUS=$(IFS=,; echo "${GPU_ARRAY[*]:0:${JUDGE_TP_SIZE}}")

echo "=========================================="
echo "Judge eval — $(date)"
echo "Judge model:  ${JUDGE_MODEL} (served as ${JUDGE_MODEL_NAME})"
echo "GPUs:         ${JUDGE_GPUS}   TP=${JUDGE_TP_SIZE}   port=${JUDGE_PORT}"
echo "Target:       ${TARGET}"
echo "Sample:       run_index=${RUN_INDEX:-all}   sample_indices=${SAMPLE_INDICES:-all}"
echo "What:         ${WHAT}"
echo "Workers:      ${WORKERS} × ${TURN_WORKERS} = $(( WORKERS * TURN_WORKERS )) concurrent LLM calls"
echo "Reasoning:    ${JUDGE_REASONING_PARSER}"
echo "=========================================="

# ---------------- start vLLM ----------------
VLLM_LOG="${LOGS_DIR_HOST}/info-gainme-judge-${SLURM_JOB_ID}-vllm-${JUDGE_MODEL_NAME}.log"

start_vllm() {
    local cmd="/usr/bin/python3 -m vllm.entrypoints.openai.api_server \
        --model ${JUDGE_MODEL} \
        --served-model-name ${JUDGE_MODEL_NAME} \
        --download-dir /workspace/hf-cache/hub \
        --port ${JUDGE_PORT} --host 0.0.0.0 \
        --tensor-parallel-size ${JUDGE_TP_SIZE} \
        --gpu-memory-utilization ${JUDGE_GPU_MEM} \
        --max-num-seqs ${VLLM_MAX_NUM_SEQS} \
        --max-model-len ${JUDGE_MAX_LEN}"
    [ "${VLLM_ENFORCE_EAGER}" = "true" ] && cmd="${cmd} --enforce-eager"
    [ -n "${JUDGE_REASONING_PARSER}" ] && cmd="${cmd} --reasoning-parser ${JUDGE_REASONING_PARSER}"
    [ -n "${JUDGE_EXTRA_ARGS}" ] && cmd="${cmd} ${JUDGE_EXTRA_ARGS}"

    singularity exec --nv \
        --bind /raid/user_danielpedrozo:/workspace \
        --bind "/usr/lib/x86_64-linux-gnu/libcuda.so.1:/usr/local/cuda/compat/lib/libcuda.so.1" \
        --bind /dev/shm:/dev/shm \
        --pwd /workspace \
        --env HF_TOKEN="${HF_TOKEN}" \
        --env VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL}" \
        --env VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S}" \
        --env HF_HOME="${HF_HOME}" \
        --env CUDA_VISIBLE_DEVICES="${JUDGE_GPUS}" \
        "${SINGULARITY_IMAGE}" \
        bash -c "${cmd}" >> "${VLLM_LOG}" 2>&1 &
    echo "$!"
}

wait_vllm_ready() {
    local pid=$1 port=$2 timeout=$3
    local elapsed=0
    echo "Waiting up to ${timeout}s for judge vLLM on port ${port} (pid=${pid})..."
    while ! curl -s "http://localhost:${port}/v1/models" > /dev/null 2>&1; do
        if ! kill -0 "${pid}" 2>/dev/null; then
            echo "ERROR: vLLM (${pid}) died before readiness — tailing log:"
            tail -n 80 "${VLLM_LOG}" 2>/dev/null || true
            exit 1
        fi
        if [ "${elapsed}" -ge "${timeout}" ]; then
            echo "ERROR: vLLM not ready after ${timeout}s — aborting"
            tail -n 80 "${VLLM_LOG}" 2>/dev/null || true
            kill "${pid}" 2>/dev/null || true
            exit 1
        fi
        sleep 5
        elapsed=$((elapsed + 5))
    done
    echo "✓ judge ready after ${elapsed}s"
}

VLLM_PID=$(start_vllm)
wait_vllm_ready "${VLLM_PID}" "${JUDGE_PORT}" "${VLLM_ENGINE_READY_TIMEOUT_S}"

# ---------------- servers override ----------------
NODE_IP=$(python3 -c "import socket; print(socket.gethostbyname('$(hostname)'))" 2>/dev/null || hostname)
SERVERS_OVERRIDE="${PROJECT_DIR}/.servers_override_${SLURM_JOB_ID}.yaml"
cat > "${SERVERS_OVERRIDE}" <<EOF
servers:
  ${JUDGE_MODEL_NAME}: http://${NODE_IP}:${JUDGE_PORT}/v1
EOF
echo "Wrote ${SERVERS_OVERRIDE}"
trap 'rm -f "${SERVERS_OVERRIDE}"; kill ${VLLM_PID} 2>/dev/null || true' EXIT

# ---------------- translate TARGET into script flag ----------------
target_flag() {
    local t=$1
    if [ "${t}" = "all" ] || [ "${t}" = "ALL" ]; then
        echo "--all"
    elif [ -d "${t}" ]; then
        echo "--conversation '${t}'"
    elif [ -f "${t}" ]; then
        echo "--runs '${t}'"
    else
        echo "ERROR: TARGET '${t}' is neither a file, dir, nor 'all'" >&2
        exit 2
    fi
}
TARGET_FLAG=$(target_flag "${TARGET}")

# ---------------- run evaluation(s) ----------------
# Install deps once, then run each requested target + aggregate in a single
# singularity session so pip doesn't run repeatedly.
case "${WHAT}" in
    oracle|pruner) TARGETS=("${WHAT}") ;;
    both)          TARGETS=(oracle pruner) ;;
    *)             echo "ERROR: WHAT must be oracle|pruner|both, got '${WHAT}'"; exit 2 ;;
esac

SAMPLE_FLAGS=""
[ -n "${RUN_INDEX}" ]       && SAMPLE_FLAGS+=" --run-index ${RUN_INDEX}"
[ -n "${SAMPLE_INDICES}" ]  && SAMPLE_FLAGS+=" --sample-indices ${SAMPLE_INDICES}"

EVAL_CMDS=""
for t in "${TARGETS[@]}"; do
    EVAL_CMDS+="echo ''; echo \"[$(date '+%%H:%%M:%%S')] --target ${t}\"; "
    EVAL_CMDS+="python3 scripts/judge_eval/evaluate.py --target ${t} ${TARGET_FLAG} \
        --judge-model '${JUDGE_MODEL_NAME}' \
        --servers-override '${SERVERS_OVERRIDE}' \
        --workers ${WORKERS} --turn-workers ${TURN_WORKERS}${SAMPLE_FLAGS}; "
done
EVAL_CMDS+="python3 scripts/judge_eval/aggregate_judge_results.py"

sg "${SHARED_GROUP}" -c "
    singularity exec \
        --bind /raid/user_danielpedrozo:/workspace \
        --pwd /workspace/projects/info-gainme_dev \
        '${SINGULARITY_IMAGE}' \
        bash -c 'pip install --user -r requirements.txt 2>/dev/null; ${EVAL_CMDS}'
"

echo ""
echo "=========================================="
echo "Done — $(date)"
echo "=========================================="
