#!/bin/bash
#SBATCH --job-name=info-gainme-ext
#SBATCH --partition=h100n2
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=2-00:00:00
#SBATCH --output=/raid/user_danielpedrozo/projects/info-gainme_dev/logs/%x-%j.out
#
# Versão do run_full_benchmark.sh que NÃO sobe o seeker (já rodando externamente).
# Sobe apenas o oracle/pruner (MODEL1, default Qwen3-8B) num GPU local,
# e roda os benchmarks apontando o seeker para servers.yaml.
#
# Uso:
#   sbatch dgx/run_external_seeker_benchmark.sh configs/full/235b/no_cot/
#   sbatch dgx/run_external_seeker_benchmark.sh configs/full/llama-70b/no_cot/
#
# Vars override via --export:
#   MODEL1         HuggingFace model ID do oracle/pruner
#   MODEL1_NAME    served-model-name (deve bater com o nome em servers.yaml)
#   MODEL1_PORT    porta do vLLM oracle/pruner (default: 8000 + JOB_ID % 1000)
#   CONFIGS_TARGET pasta ou arquivo yaml

umask 002

# ============================================
# Configuration
# ============================================
CONFIGS_TARGET="${CONFIGS_TARGET:-configs/full/235b/no_cot/}"

BASE_PORT=$((8000 + (SLURM_JOB_ID % 1000)))
export MODEL1_PORT="${MODEL1_PORT:-${BASE_PORT}}"

export MODEL1="${MODEL1:-Qwen/Qwen3-8B}"
export MODEL1_NAME="${MODEL1_NAME:-Qwen3-8B}"
export MODEL1_GPU_MEM="${MODEL1_GPU_MEM:-0.90}"
export MODEL1_MAX_LEN="${MODEL1_MAX_LEN:-32000}"
export MODEL1_REASONING_PARSER="${MODEL1_REASONING_PARSER:-}"

PROJECT_DIR="/raid/user_danielpedrozo/projects/info-gainme_dev"
SHARED_GROUP="sd22"
SINGULARITY_IMAGE="/raid/user_danielpedrozo/images/vllm_openai_latest.sif"

# ============================================
# GPU Detection
# ============================================
if [ -z "${CUDA_VISIBLE_DEVICES}" ]; then
    echo "ERROR: CUDA_VISIBLE_DEVICES not set by SLURM"
    exit 1
fi

IFS=',' read -ra GPU_ARRAY <<< "${CUDA_VISIBLE_DEVICES}"
MODEL1_GPU=${GPU_ARRAY[0]}

export VLLM_LOGGING_LEVEL=DEBUG
export HF_HOME=/workspace/hf-cache
source "${PROJECT_DIR}/.env"
export HF_TOKEN="${HF_TOKEN:?HF_TOKEN não definido no .env}"
export LOGS_DIR="/workspace/projects/info-gainme_dev/logs"

mkdir -p "${PROJECT_DIR}/logs" "${PROJECT_DIR}/hf-cache" "${PROJECT_DIR}/outputs"
cd "${PROJECT_DIR}"

echo "=========================================="
echo "Info Gainme — External Seeker Benchmark"
echo "$(date)"
echo "Oracle/Pruner: ${MODEL1_NAME} on GPU ${MODEL1_GPU} (port ${MODEL1_PORT})"
echo "Seeker:        external (from servers.yaml)"
echo "Configs:       ${CONFIGS_TARGET}"
echo "=========================================="
echo ""

# ============================================
# Start oracle/pruner vLLM
# ============================================
VLLM_LOG="${LOGS_DIR}/info-gainme-ext-${SLURM_JOB_ID}-vllm-${MODEL1_NAME}.log"

CMD="/usr/bin/python3 -m vllm.entrypoints.openai.api_server \
  --model ${MODEL1} \
  --served-model-name ${MODEL1_NAME} \
  --download-dir /workspace/hf-cache/hub \
  --port ${MODEL1_PORT} \
  --host 0.0.0.0 \
  --gpu-memory-utilization ${MODEL1_GPU_MEM} \
  --max-num-seqs 16 \
  --max-model-len ${MODEL1_MAX_LEN} \
  --enforce-eager"
[ -n "${MODEL1_REASONING_PARSER}" ] && CMD="${CMD} --reasoning-parser ${MODEL1_REASONING_PARSER}"

echo "Starting ${MODEL1_NAME}..."
nohup singularity exec --nv \
    --bind /raid/user_danielpedrozo:/workspace \
    --bind "/usr/lib/x86_64-linux-gnu/libcuda.so.1:/usr/local/cuda/compat/lib/libcuda.so.1" \
    --bind /dev/shm:/dev/shm \
    --pwd /workspace \
    --env HF_TOKEN=${HF_TOKEN} \
    --env VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL} \
    --env HF_HOME=${HF_HOME} \
    --env CUDA_VISIBLE_DEVICES=${MODEL1_GPU} \
    "${SINGULARITY_IMAGE}" \
    sh -c "mkdir -p $(dirname ${VLLM_LOG}) && ${CMD} > ${VLLM_LOG} 2>&1" \
    >/dev/null 2>&1 &
PID1=$!

echo "  PID: ${PID1} | Log: ${VLLM_LOG}"
echo -n "  Aguardando ${MODEL1_NAME} ficar pronto..."
while ! curl -s http://localhost:${MODEL1_PORT}/v1/models > /dev/null 2>&1; do
    sleep 5
    echo -n "."
done
echo ""
echo "  ✓ ${MODEL1_NAME} pronto"
echo ""

# ============================================
# Servers override — apenas oracle/pruner local
# O seeker continua resolvido via servers.yaml
# ============================================
NODE_IP=$(python3 -c "import socket; print(socket.gethostbyname('$(hostname)'))" 2>/dev/null || echo "$(hostname)")
SERVERS_OVERRIDE="${PROJECT_DIR}/.servers_override_${SLURM_JOB_ID}.yaml"

cat > "${SERVERS_OVERRIDE}" <<EOF
servers:
  ${MODEL1_NAME}: http://${NODE_IP}:${MODEL1_PORT}/v1
EOF

echo "Servers override: ${SERVERS_OVERRIDE}"
echo "  ✓ ${MODEL1_NAME} → http://${NODE_IP}:${MODEL1_PORT}/v1"
echo "  (seeker endpoint lido de configs/servers.yaml)"
echo ""

# ============================================
# Resolve configs
# ============================================
[[ "${CONFIGS_TARGET}" != /* ]] && CONFIGS_TARGET="${PROJECT_DIR}/${CONFIGS_TARGET}"
if [[ -f "${CONFIGS_TARGET}" ]]; then
    CONFIGS=("${CONFIGS_TARGET}")
elif [[ -d "${CONFIGS_TARGET}" ]]; then
    mapfile -t CONFIGS < <(find "${CONFIGS_TARGET}" -name "*.yaml" -type f | sort)
else
    echo "ERROR: '${CONFIGS_TARGET}' não encontrado"
    kill $PID1 2>/dev/null
    exit 1
fi

echo "=========================================="
echo "Rodando ${#CONFIGS[@]} config(s)"
echo "=========================================="
echo ""

for CONFIG in "${CONFIGS[@]}"; do
    REL="${CONFIG#${PROJECT_DIR}/}"
    echo "[$(date '+%H:%M:%S')] ${REL}"
    sg "${SHARED_GROUP}" -c "
        singularity exec \
            --bind /raid/user_danielpedrozo:/workspace \
            --pwd /workspace/projects/info-gainme_dev \
            '${SINGULARITY_IMAGE}' \
            bash -c \"
                pip install --user -r requirements.txt 2>/dev/null
                python3 benchmark_runner.py --config '${REL}' --servers-override '${SERVERS_OVERRIDE}'
            \"
    " && echo "  ✓" || echo "  ✗"
    echo ""
done

echo "=========================================="
echo "Cleanup..."
kill $PID1 2>/dev/null
wait $PID1 2>/dev/null
rm -f "${SERVERS_OVERRIDE}"
echo "Done - $(date)"
echo "=========================================="
