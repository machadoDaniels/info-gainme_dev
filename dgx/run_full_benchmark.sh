#!/bin/bash
#SBATCH --job-name=info-gainme-full
#SBATCH --partition=h100n2
#SBATCH --gres=gpu:2
#SBATCH --mem=60G
#SBATCH --time=2-00:00:00
#SBATCH --output=/raid/user_danielpedrozo/projects/info-gainme_dev/logs/%x-%j.out

umask 002

DEFAULT_CONFIGS_DIR="configs/full/4b/cot/"
CONFIGS_TARGET="${1:-${DEFAULT_CONFIGS_DIR}}"

PROJECT_DIR="/raid/user_danielpedrozo/projects/info-gainme_dev"
SHARED_GROUP="sd22"
SINGULARITY_IMAGE="/raid/user_danielpedrozo/images/vllm_openai_latest.sif"

# ============================================
# Auto-detect GPU allocation from SLURM
# ============================================
# SLURM provides CUDA_VISIBLE_DEVICES with actual GPU indices on the node
# e.g., CUDA_VISIBLE_DEVICES=0,1 or CUDA_VISIBLE_DEVICES=2,5
if [ -z "${CUDA_VISIBLE_DEVICES}" ]; then
    echo "ERROR: CUDA_VISIBLE_DEVICES not set by SLURM"
    exit 1
fi

# Parse GPU list and assign to models
IFS=',' read -ra GPU_ARRAY <<< "${CUDA_VISIBLE_DEVICES}"
TOTAL_GPUS=${#GPU_ARRAY[@]}

if [ ${TOTAL_GPUS} -lt 2 ]; then
    echo "ERROR: Job requires at least 2 GPUs (allocated: ${TOTAL_GPUS})"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    exit 1
fi

# Assign actual GPU indices from SLURM allocation
MODEL1_GPU=${GPU_ARRAY[0]}
MODEL2_GPU=${GPU_ARRAY[1]}

export MODEL1="Qwen/Qwen3-4B-Thinking-2507"
export MODEL1_NAME="Qwen3-4B-Thinking-2507"
export MODEL1_PORT=8021
export MODEL1_GPU_MEM=0.90
export MODEL1_MAX_LEN=32000
export MODEL1_REASONING_PARSER=""

export MODEL2="Qwen/Qwen3-8B"
export MODEL2_NAME="Qwen3-8B"
export MODEL2_PORT=8028
export MODEL2_GPU_MEM=0.90
export MODEL2_MAX_LEN=32000
export MODEL2_REASONING_PARSER=""

export VLLM_LOGGING_LEVEL=DEBUG
export HF_HOME=/workspace/hf-cache
source /raid/user_danielpedrozo/projects/info-gainme_dev/.env
export HF_TOKEN="${HF_TOKEN:?HF_TOKEN não definido no .env}"
export LOGS_DIR="/workspace/projects/info-gainme_dev/logs"

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

start_vllm_server() {
    local model=$1 name=$2 port=$3 gpu=$4 gpu_mem=$5 max_len=$6 log=$7 parser=${8:-""}
    echo "Starting ${name} (GPU ${gpu}:${port})..."
    export CUDA_VISIBLE_DEVICES=${gpu}

    local cmd="/usr/bin/python3 -m vllm.entrypoints.openai.api_server --model ${model} --served-model-name ${name} --download-dir /workspace/hf-cache/hub --port ${port} --host 0.0.0.0 --gpu-memory-utilization ${gpu_mem} --max-num-seqs 16 --max-model-len ${max_len} --enforce-eager"
    [ -n "${parser}" ] && cmd="${cmd} --reasoning-parser ${parser}"

    singularity exec --nv --bind /raid/user_danielpedrozo:/workspace --bind "/usr/lib/x86_64-linux-gnu/libcuda.so.1:/usr/local/cuda/compat/lib/libcuda.so.1" --bind /dev/shm:/dev/shm --pwd /workspace --env HF_TOKEN=${HF_TOKEN} --env VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL} --env HF_HOME=${HF_HOME} --env CUDA_VISIBLE_DEVICES=${gpu} "${SINGULARITY_IMAGE}" sh -c "mkdir -p $(dirname ${log}) && ${cmd} > ${log} 2>&1" &
    echo "$!"
}

PID1=$(start_vllm_server "${MODEL1}" "${MODEL1_NAME}" ${MODEL1_PORT} ${MODEL1_GPU} ${MODEL1_GPU_MEM} ${MODEL1_MAX_LEN} "${LOGS_DIR}/vllm-${SLURM_JOB_ID}-${MODEL1_NAME}.log" "${MODEL1_REASONING_PARSER}")
while ! curl -s http://localhost:${MODEL1_PORT}/v1/models > /dev/null 2>&1; do sleep 5; done
echo "✓ ${MODEL1_NAME} ready"
echo ""

PID2=$(start_vllm_server "${MODEL2}" "${MODEL2_NAME}" ${MODEL2_PORT} ${MODEL2_GPU} ${MODEL2_GPU_MEM} ${MODEL2_MAX_LEN} "${LOGS_DIR}/vllm-${SLURM_JOB_ID}-${MODEL2_NAME}.log" "${MODEL2_REASONING_PARSER}")
while ! curl -s http://localhost:${MODEL2_PORT}/v1/models > /dev/null 2>&1; do sleep 5; done
echo "✓ ${MODEL2_NAME} ready"
echo ""

echo "Creating servers override file..."
NODE_IP=$(python3 -c "import socket; print(socket.gethostbyname('$(hostname)'))" 2>/dev/null || echo "$(hostname)")

SERVERS_OVERRIDE="${PROJECT_DIR}/.servers_override_${SLURM_JOB_ID}.yaml"
cat > "${SERVERS_OVERRIDE}" <<EOF
servers:
  ${MODEL1_NAME}: http://${NODE_IP}:${MODEL1_PORT}/v1
  ${MODEL2_NAME}: http://${NODE_IP}:${MODEL2_PORT}/v1
EOF

echo "  ✓ ${SERVERS_OVERRIDE} created"
echo "  ✓ ${MODEL1_NAME} → http://${NODE_IP}:${MODEL1_PORT}/v1"
echo "  ✓ ${MODEL2_NAME} → http://${NODE_IP}:${MODEL2_PORT}/v1"
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
    sg "${SHARED_GROUP}" -c "singularity exec --bind /raid/user_danielpedrozo:/workspace --pwd /workspace/projects/info-gainme_dev '${SINGULARITY_IMAGE}' bash -c \"pip install --user -r requirements.txt 2>/dev/null; python3 benchmark_runner.py --config '${REL}' --servers-override '${SERVERS_OVERRIDE}'\"" && echo "  ✓" || echo "  ✗"
    echo ""
done

echo "=========================================="
echo "Cleanup..."
kill $PID1 $PID2 2>/dev/null
wait $PID1 $PID2 2>/dev/null
rm -f "${SERVERS_OVERRIDE}"
echo "Done - $(date)"
echo "=========================================="
