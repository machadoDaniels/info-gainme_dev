#!/bin/bash
#SBATCH --job-name=info-gainme-full
#SBATCH --partition=h100n2
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --time=2-00:00:00
#SBATCH --output=/raid/user_danielpedrozo/projects/info-gainme_dev/logs/%x-%j.log

umask 002

# ============================================
# Configuration
# ============================================
CONFIGS_TARGET="${CONFIGS_TARGET:-configs/full/8b/}"

export MODE="${MODE:-single}"

# Dynamic ports based on SLURM_JOB_ID to avoid conflicts.
# Multiply by 10 so a job needing up to 2 ports (base, base+1) can never collide
# with neighboring JOB_IDs (previous bug: consecutive job IDs shared a port).
BASE_PORT=$((8000 + (SLURM_JOB_ID % 500) * 10))
DEFAULT_MODEL1_PORT=$BASE_PORT
DEFAULT_MODEL2_PORT=$((BASE_PORT + 1))

# If either port is already bound on this node, advance to the next free pair.
port_in_use() { ss -tln 2>/dev/null | awk '{print $4}' | grep -qE ":$1$"; }
while port_in_use $DEFAULT_MODEL1_PORT || port_in_use $DEFAULT_MODEL2_PORT; do
    DEFAULT_MODEL1_PORT=$((DEFAULT_MODEL1_PORT + 2))
    DEFAULT_MODEL2_PORT=$((DEFAULT_MODEL2_PORT + 2))
done

export MODEL1_PORT="${MODEL1_PORT:-$DEFAULT_MODEL1_PORT}"
export MODEL2_PORT="${MODEL2_PORT:-$DEFAULT_MODEL2_PORT}"

export MODEL1="${MODEL1:-Qwen/Qwen3-4B-Thinking-2507}"
export MODEL1_NAME="${MODEL1_NAME:-Qwen3-4B-Thinking-2507}"
export MODEL1_GPU_MEM="${MODEL1_GPU_MEM:-0.90}"
export MODEL1_MAX_LEN="${MODEL1_MAX_LEN:-32000}"
export MODEL1_REASONING_PARSER="${MODEL1_REASONING_PARSER:-}"

export MODEL2="${MODEL2:-Qwen/Qwen3-8B}"
export MODEL2_NAME="${MODEL2_NAME:-Qwen3-8B}"
export MODEL2_GPU_MEM="${MODEL2_GPU_MEM:-0.90}"
export MODEL2_MAX_LEN="${MODEL2_MAX_LEN:-32000}"
export MODEL2_REASONING_PARSER="${MODEL2_REASONING_PARSER:-}"

# MODE can be overridden via: sbatch --export=ALL,MODE=dual,CONFIGS_TARGET=configs/full/4b ...
# All MODEL1/MODEL2 vars can similarly be overridden via --export

PROJECT_DIR="/raid/user_danielpedrozo/projects/info-gainme_dev"
SHARED_GROUP="sd22"
SINGULARITY_IMAGE="/raid/user_danielpedrozo/images/vllm_openai_latest.sif"


# ============================================
# GPU Detection & Mode Selection
# ============================================
# SLURM provides CUDA_VISIBLE_DEVICES with actual GPU indices on the node
# e.g., CUDA_VISIBLE_DEVICES=0,1 or CUDA_VISIBLE_DEVICES=2,5
if [ -z "${CUDA_VISIBLE_DEVICES}" ]; then
    echo "ERROR: CUDA_VISIBLE_DEVICES not set by SLURM"
    exit 1
fi

# Parse GPU list
IFS=',' read -ra GPU_ARRAY <<< "${CUDA_VISIBLE_DEVICES}"
export TOTAL_GPUS=${#GPU_ARRAY[@]}

# Auto-detect mode if not manually set
if [ -z "${MODE}" ]; then
    if [ ${TOTAL_GPUS} -eq 1 ]; then
        MODE="single"
    elif [ ${TOTAL_GPUS} -ge 2 ]; then
        MODE="dual"
    fi
    echo "Mode: AUTO-DETECTED = ${MODE} (GPUs: ${TOTAL_GPUS})"
else
    echo "Mode: MANUAL = ${MODE}"
fi

# Assign GPUs based on mode
if [ "${MODE}" = "single" ]; then
    MODEL1_GPU=${GPU_ARRAY[0]}
    MODEL2_GPU=${GPU_ARRAY[0]}  # Same GPU
    echo "  → Single model: all agents on GPU ${MODEL1_GPU}"
elif [ "${MODE}" = "dual" ]; then
    if [ ${TOTAL_GPUS} -lt 2 ]; then
        echo "ERROR: MODE=dual requires 2+ GPUs, but only ${TOTAL_GPUS} allocated"
        exit 1
    fi
    MODEL1_GPU=${GPU_ARRAY[0]}
    MODEL2_GPU=${GPU_ARRAY[1]}
    echo "  → Dual model: seeker on GPU ${MODEL1_GPU}, oracle/pruner on GPU ${MODEL2_GPU}"
else
    echo "ERROR: MODE must be 'single' or 'dual', got: ${MODE}"
    exit 1
fi



export VLLM_LOGGING_LEVEL=DEBUG
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-1800}"
export HF_HOME=/workspace/hf-cache
source /raid/user_danielpedrozo/projects/info-gainme_dev/.env
export HF_TOKEN="${HF_TOKEN:?HF_TOKEN não definido no .env}"
# LOGS_DIR is the container path (used only by code that runs inside singularity);
# LOGS_DIR_HOST is the equivalent host path used for redirects and tail.
export LOGS_DIR="/workspace/projects/info-gainme_dev/logs"
export LOGS_DIR_HOST="${PROJECT_DIR}/logs"

mkdir -p "${PROJECT_DIR}/logs" "${PROJECT_DIR}/hf-cache" "${PROJECT_DIR}/outputs"
cd "${PROJECT_DIR}"

echo "=========================================="
echo "Info Gainme Full Benchmark - $(date)"
echo "GPUs allocated: ${TOTAL_GPUS} (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})"
echo "Seeker:  ${MODEL1_NAME} on GPU ${MODEL1_GPU} (port ${MODEL1_PORT})"
echo "Oracle:  ${MODEL2_NAME} on GPU ${MODEL2_GPU} (port ${MODEL2_PORT})"
echo "Configs: ${CONFIGS_TARGET}"
echo "=========================================="
echo ""

# Tuning dos vLLMs:
# - VLLM_MAX_NUM_SEQS: número de requests paralelos máximo (default 32).
# - VLLM_ENFORCE_EAGER: se "true" passa --enforce-eager (mais rápido pra subir,
#   ~20-30% mais lento em runtime). Auto-desativado em B200 (CUDA graphs valem
#   mais a pena em Blackwell); ativado em H100 por segurança. Override manual:
#   VLLM_ENFORCE_EAGER=true|false.
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-32}"
if [ -z "${VLLM_ENFORCE_EAGER:-}" ]; then
    if [[ "${SLURM_JOB_PARTITION:-}" == *b200* ]]; then
        VLLM_ENFORCE_EAGER="false"
    else
        VLLM_ENFORCE_EAGER="true"
    fi
fi
echo "  max_num_seqs=${VLLM_MAX_NUM_SEQS} | enforce_eager=${VLLM_ENFORCE_EAGER} | partition=${SLURM_JOB_PARTITION}"
echo ""

start_vllm_server() {
    local model=$1 name=$2 port=$3 gpu=$4 gpu_mem=$5 max_len=$6 log=$7 parser=${8:-""}
    echo "Starting ${name} (GPU ${gpu}:${port})..." >&2

    local cmd="/usr/bin/python3 -m vllm.entrypoints.openai.api_server --model ${model} --served-model-name ${name} --download-dir /workspace/hf-cache/hub --port ${port} --host 0.0.0.0 --gpu-memory-utilization ${gpu_mem} --max-num-seqs ${VLLM_MAX_NUM_SEQS} --max-model-len ${max_len}"
    [ "${VLLM_ENFORCE_EAGER}" = "true" ] && cmd="${cmd} --enforce-eager"
    [ -n "${parser}" ] && cmd="${cmd} --reasoning-parser ${parser}"

    # Redireciona stdout/stderr do singularity exec (host) para o MESMO arquivo
    # de log do vLLM, assim erros de startup do container não são perdidos.
    # Também evita nohup para reduzir interferência com cgroup do SLURM.
    mkdir -p "$(dirname ${log})" 2>/dev/null || true
    singularity exec --nv \
      --bind /raid/user_danielpedrozo:/workspace \
      --bind "/usr/lib/x86_64-linux-gnu/libcuda.so.1:/usr/local/cuda/compat/lib/libcuda.so.1" \
      --bind /dev/shm:/dev/shm \
      --pwd /workspace \
      --env HF_TOKEN=${HF_TOKEN} \
      --env VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL} \
      --env VLLM_ENGINE_READY_TIMEOUT_S=${VLLM_ENGINE_READY_TIMEOUT_S} \
      --env HF_HOME=${HF_HOME} \
      --env CUDA_VISIBLE_DEVICES=${gpu} \
      "${SINGULARITY_IMAGE}" \
      bash -c "${cmd}" >> "${log}" 2>&1 &
    echo "$!"
}
clear
# Wait for a vLLM server to become ready, with timeout and liveness check.
# Aborts the job (exit 1) if the readiness check times out or the vLLM process dies.
# Usage: wait_vllm_ready <pid> <port> <name> [timeout_seconds]
wait_vllm_ready() {
    local pid=$1 port=$2 name=$3 timeout=${4:-1800}
    local elapsed=0
    echo "Waiting up to ${timeout}s for ${name} on port ${port} (pid=${pid})..."
    while ! curl -s http://localhost:${port}/v1/models > /dev/null 2>&1; do
        if ! kill -0 ${pid} 2>/dev/null; then
            echo "ERROR: vLLM process ${pid} for ${name} died before readiness"
            tail -n 50 "${LOGS_DIR_HOST}/info-gainme-full-${SLURM_JOB_ID}-vllm-${name}.log" 2>/dev/null || true
            exit 1
        fi
        if [ ${elapsed} -ge ${timeout} ]; then
            echo "ERROR: ${name} not ready after ${timeout}s — aborting"
            tail -n 50 "${LOGS_DIR_HOST}/info-gainme-full-${SLURM_JOB_ID}-vllm-${name}.log" 2>/dev/null || true
            kill ${pid} 2>/dev/null || true
            exit 1
        fi
        sleep 5
        elapsed=$((elapsed + 5))
    done
    echo "✓ ${name} ready after ${elapsed}s"
}

PID1=$(start_vllm_server "${MODEL1}" "${MODEL1_NAME}" ${MODEL1_PORT} ${MODEL1_GPU} ${MODEL1_GPU_MEM} ${MODEL1_MAX_LEN} "${LOGS_DIR_HOST}/info-gainme-full-${SLURM_JOB_ID}-vllm-${MODEL1_NAME}.log" "${MODEL1_REASONING_PARSER}")
wait_vllm_ready ${PID1} ${MODEL1_PORT} "${MODEL1_NAME}" "${VLLM_READY_TIMEOUT:-1800}"
echo ""

# Start second model only in dual mode
PID2=""
if [ "${MODE}" = "dual" ]; then
    PID2=$(start_vllm_server "${MODEL2}" "${MODEL2_NAME}" ${MODEL2_PORT} ${MODEL2_GPU} ${MODEL2_GPU_MEM} ${MODEL2_MAX_LEN} "${LOGS_DIR_HOST}/info-gainme-full-${SLURM_JOB_ID}-vllm-${MODEL2_NAME}.log" "${MODEL2_REASONING_PARSER}")
    wait_vllm_ready ${PID2} ${MODEL2_PORT} "${MODEL2_NAME}" "${VLLM_READY_TIMEOUT:-1800}"
    echo ""
else
    # Single mode: use MODEL1 for all agents
    MODEL2_NAME="${MODEL1_NAME}"
    MODEL2_PORT=${MODEL1_PORT}
    echo "(Single mode: using ${MODEL1_NAME} for all agents)"
    echo ""
fi

echo "Creating servers override file..."
NODE_IP=$(python3 -c "import socket; print(socket.gethostbyname('$(hostname)'))" 2>/dev/null || echo "$(hostname)")

SERVERS_OVERRIDE="${PROJECT_DIR}/.servers_override_${SLURM_JOB_ID}.yaml"
if [ "${MODE}" = "dual" ]; then
    cat > "${SERVERS_OVERRIDE}" <<EOF
servers:
  ${MODEL1_NAME}: http://${NODE_IP}:${MODEL1_PORT}/v1
  ${MODEL2_NAME}: http://${NODE_IP}:${MODEL2_PORT}/v1
EOF
else
    # Single mode: only one server, write MODEL1_NAME only to avoid duplicate key
    cat > "${SERVERS_OVERRIDE}" <<EOF
servers:
  ${MODEL1_NAME}: http://${NODE_IP}:${MODEL1_PORT}/v1
EOF
fi

echo "  ✓ ${SERVERS_OVERRIDE} created"
echo "  ✓ ${MODEL1_NAME} → http://${NODE_IP}:${MODEL1_PORT}/v1"
if [ "${MODE}" = "dual" ]; then
    echo "  ✓ ${MODEL2_NAME} → http://${NODE_IP}:${MODEL2_PORT}/v1"
fi
echo ""

[[ "${CONFIGS_TARGET}" != /* ]] && CONFIGS_TARGET="${PROJECT_DIR}/${CONFIGS_TARGET}"
if [[ -f "${CONFIGS_TARGET}" ]]; then
    CONFIGS=("${CONFIGS_TARGET}")
elif [[ -d "${CONFIGS_TARGET}" ]]; then
    mapfile -t CONFIGS < <(find "${CONFIGS_TARGET}" -name "*.yaml" -type f | sort)
else
    echo "ERROR: '${CONFIGS_TARGET}' not found"; kill $PID1 $PID2 2>/dev/null; exit 1
fi

echo "=========================================="
echo "Running ${#CONFIGS[@]} benchmark config(s)"
echo "=========================================="
echo ""

for CONFIG in "${CONFIGS[@]}"; do
    REL="${CONFIG#${PROJECT_DIR}/}"
    echo "[$(date '+%H:%M:%S')] ${REL}"
    RUNS_ARG=""
    [ -n "${RUNS_PER_TARGET}" ] && RUNS_ARG="--runs-per-target ${RUNS_PER_TARGET}"
    sg "${SHARED_GROUP}" -c "singularity exec --bind /raid/user_danielpedrozo:/workspace --pwd /workspace/projects/info-gainme_dev '${SINGULARITY_IMAGE}' bash -c \"pip install --user -r requirements.txt 2>/dev/null; python3 benchmark_runner.py --config '${REL}' --servers-override '${SERVERS_OVERRIDE}' ${RUNS_ARG}\"" && echo "  ✓" || echo "  ✗"
    echo ""
done

echo "=========================================="
echo "Cleanup..."
kill $PID1 2>/dev/null
[ -n "$PID2" ] && kill $PID2 2>/dev/null
wait $PID1 $PID2 2>/dev/null
rm -f "${SERVERS_OVERRIDE}"
echo "Done - $(date)"
echo "=========================================="
