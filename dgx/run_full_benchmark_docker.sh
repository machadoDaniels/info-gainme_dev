#!/bin/bash
# Roda vLLM + benchmarks via Docker (sem SLURM, sem Singularity).
# Equivalente ao run_full_benchmark.sh para nós sem SLURM (ex: dgx-A100).
#
# Uso:
#   GPUS=0,1 MODE=dual MODEL1_NAME=Phi-4-reasoning MODEL2_NAME=Qwen3-8B \
#     CONFIGS_TARGET=configs/full/phi4/cot/ \
#     bash dgx/run_full_benchmark_docker.sh
#
#   GPUS=0 MODE=single MODEL1_NAME=Qwen3-8B \
#     CONFIGS_TARGET=configs/full/8b/ \
#     bash dgx/run_full_benchmark_docker.sh
#
#   GPUS=0,1 MODE=seeker_only MODEL1_NAME=Qwen3-30B-A3B-Thinking-2507 MODEL2_NAME=Qwen3-8B \
#     CONFIGS_TARGET=configs/full/30b/cot/ \
#     bash dgx/run_full_benchmark_docker.sh
#
# Variáveis configuráveis (todas via env ou export antes de chamar):
#   GPUS                  índices de GPU separados por vírgula (padrão: todos)
#   MODE                  single | dual | seeker_only (auto-detectado se omitido)
#   MODEL1, MODEL1_NAME   seeker: HF repo id + served-model-name
#   MODEL2, MODEL2_NAME   oracle/pruner: HF repo id + served-model-name
#   MODEL1_PORT, MODEL2_PORT  portas (auto-calculadas por PID se omitidas)
#   MODEL1_GPU_MEM, MODEL2_GPU_MEM  gpu-memory-utilization (padrão 0.90)
#   MODEL1_MAX_LEN, MODEL2_MAX_LEN  max-model-len (padrão 32000)
#   MODEL1_REASONING_PARSER, MODEL2_REASONING_PARSER  auto-detectado por nome
#   VLLM_MAX_NUM_SEQS     (padrão 32)
#   VLLM_ENFORCE_EAGER    true|false (auto por GPU: A100→true, B200→false)
#   VLLM_ENGINE_READY_TIMEOUT_S  (padrão 1800; modelos grandes: 3600)
#   RUNS_PER_TARGET       sobrescreve dataset.runs_per_target do YAML
#   VLLM_IMAGE            imagem Docker (padrão vllm/vllm-openai:latest)

VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:latest}"

# ============================================
# Se não estiver em screen, relança dentro de um
# ============================================
if [[ -z "${STY}" && -z "${TMUX}" && -z "${DOCKER_BENCHMARK_INNER}" ]]; then
    TARGET_BASE=$(basename "${CONFIGS_TARGET:-configs/full/8b}" .yaml)
    SESSION_NAME="benchmark-${MODEL1_NAME:-model}-${TARGET_BASE}"
    export DOCKER_BENCHMARK_INNER=1
    echo "Iniciando screen '${SESSION_NAME}'..."
    echo "Acompanhe: screen -r ${SESSION_NAME}"
    # Re-exporta todo o ambiente para o screen
    screen -dmS "${SESSION_NAME}" env \
        GPUS="${GPUS:-}" \
        MODE="${MODE:-}" \
        MODEL1="${MODEL1:-}" MODEL1_NAME="${MODEL1_NAME:-}" \
        MODEL2="${MODEL2:-}" MODEL2_NAME="${MODEL2_NAME:-}" \
        MODEL1_PORT="${MODEL1_PORT:-}" MODEL2_PORT="${MODEL2_PORT:-}" \
        MODEL1_GPU_MEM="${MODEL1_GPU_MEM:-}" MODEL2_GPU_MEM="${MODEL2_GPU_MEM:-}" \
        MODEL1_MAX_LEN="${MODEL1_MAX_LEN:-}" MODEL2_MAX_LEN="${MODEL2_MAX_LEN:-}" \
        MODEL1_REASONING_PARSER="${MODEL1_REASONING_PARSER:-}" \
        MODEL2_REASONING_PARSER="${MODEL2_REASONING_PARSER:-}" \
        VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-}" \
        VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-}" \
        VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-}" \
        RUNS_PER_TARGET="${RUNS_PER_TARGET:-}" \
        VLLM_IMAGE="${VLLM_IMAGE}" \
        CONFIGS_TARGET="${CONFIGS_TARGET:-configs/full/8b/}" \
        DOCKER_BENCHMARK_INNER=1 \
        bash "$(realpath "$0")"
    exit 0
fi

# ============================================
# Configuration
# ============================================
CONFIGS_TARGET="${CONFIGS_TARGET:-configs/full/8b/}"
export MODE="${MODE:-}"

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Dynamic ports based on PID to avoid conflicts between concurrent runs.
BASE_PORT=$((8000 + ($$ % 500) * 10))
DEFAULT_MODEL1_PORT=$BASE_PORT
DEFAULT_MODEL2_PORT=$((BASE_PORT + 1))

port_in_use() { ss -tln 2>/dev/null | awk '{print $4}' | grep -qE ":$1$"; }
while port_in_use $DEFAULT_MODEL1_PORT || port_in_use $DEFAULT_MODEL2_PORT; do
    DEFAULT_MODEL1_PORT=$((DEFAULT_MODEL1_PORT + 2))
    DEFAULT_MODEL2_PORT=$((DEFAULT_MODEL2_PORT + 2))
done

export MODEL1_PORT="${MODEL1_PORT:-$DEFAULT_MODEL1_PORT}"
export MODEL2_PORT="${MODEL2_PORT:-$DEFAULT_MODEL2_PORT}"

# Look up HF repo id from configs/servers.yaml hf_paths.<served-model-name>.
lookup_hf_path() {
    python3 -c "
import yaml, sys
try:
    with open('${PROJECT_DIR}/configs/servers.yaml') as f:
        data = yaml.safe_load(f) or {}
    print((data.get('hf_paths') or {}).get('$1', ''))
except Exception:
    pass
" 2>/dev/null
}

# Use ${VAR:-} to treat both unset and empty-string the same way — the
# screen re-launch passes MODEL1="" which would otherwise bypass this lookup.
if [ -z "${MODEL1:-}" ] && [ -n "${MODEL1_NAME:-}" ]; then
    resolved=$(lookup_hf_path "${MODEL1_NAME}")
    [ -n "${resolved}" ] && MODEL1="${resolved}" && echo "Resolved MODEL1=${MODEL1} from hf_paths.${MODEL1_NAME}"
fi
if [ -z "${MODEL2:-}" ] && [ -n "${MODEL2_NAME:-}" ]; then
    resolved=$(lookup_hf_path "${MODEL2_NAME}")
    [ -n "${resolved}" ] && MODEL2="${resolved}" && echo "Resolved MODEL2=${MODEL2} from hf_paths.${MODEL2_NAME}"
fi

export MODEL1="${MODEL1:-Qwen/Qwen3-4B-Thinking-2507}"
export MODEL1_NAME="${MODEL1_NAME:-Qwen3-4B-Thinking-2507}"
export MODEL1_GPU_MEM="${MODEL1_GPU_MEM:-0.90}"
export MODEL1_MAX_LEN="${MODEL1_MAX_LEN:-32000}"

export MODEL2="${MODEL2:-Qwen/Qwen3-8B}"
export MODEL2_NAME="${MODEL2_NAME:-Qwen3-8B}"
export MODEL2_GPU_MEM="${MODEL2_GPU_MEM:-0.90}"
export MODEL2_MAX_LEN="${MODEL2_MAX_LEN:-32000}"

# Auto-detect reasoning parser from served-model-name.
auto_reasoning_parser() {
    local name="${1,,}"
    case "$name" in
        *gpt-oss*)            echo "openai_gptoss" ;;
        *qwen3*)              echo "qwen3" ;;
        *olmo*think*|*olmo*)  echo "olmo3" ;;
        *)                    echo "" ;;
    esac
}
[ -z "${MODEL1_REASONING_PARSER+x}" ] && MODEL1_REASONING_PARSER=$(auto_reasoning_parser "${MODEL1_NAME}")
[ -z "${MODEL2_REASONING_PARSER+x}" ] && MODEL2_REASONING_PARSER=$(auto_reasoning_parser "${MODEL2_NAME}")
[ "${MODEL1_REASONING_PARSER}" = "none" ] && MODEL1_REASONING_PARSER=""
[ "${MODEL2_REASONING_PARSER}" = "none" ] && MODEL2_REASONING_PARSER=""
export MODEL1_REASONING_PARSER MODEL2_REASONING_PARSER

# ============================================
# GPU Detection & Mode Selection
# ============================================
if [ -z "${GPUS:-}" ]; then
    GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd,)
    echo "Auto-detected GPUs: ${GPUS}"
fi

IFS=',' read -ra GPU_ARRAY <<< "${GPUS}"
export TOTAL_GPUS=${#GPU_ARRAY[@]}

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

MODEL1_TP=1
MODEL2_TP=1
if [ "${MODE}" = "single" ]; then
    MODEL1_GPU="${GPU_ARRAY[0]}"
    MODEL2_GPU="${GPU_ARRAY[0]}"
    echo "  → Single model: all agents on GPU ${MODEL1_GPU}"
elif [ "${MODE}" = "dual" ]; then
    if [ ${TOTAL_GPUS} -lt 2 ]; then
        echo "ERROR: MODE=dual requires 2+ GPUs, but only ${TOTAL_GPUS} allocated"
        exit 1
    fi
    MODEL1_GPU="${GPU_ARRAY[0]}"
    MODEL2_GPU="${GPU_ARRAY[1]}"
    echo "  → Dual model: seeker on GPU ${MODEL1_GPU}, oracle/pruner on GPU ${MODEL2_GPU}"
elif [ "${MODE}" = "seeker_only" ]; then
    MODEL1_GPU="${GPUS}"
    MODEL1_TP=${TOTAL_GPUS}
    echo "  → Seeker-only: seeker on GPUs ${MODEL1_GPU} (TP=${MODEL1_TP}), oracle/pruner from servers.yaml (${MODEL2_NAME})"
else
    echo "ERROR: MODE must be 'single', 'dual', or 'seeker_only', got: ${MODE}"
    exit 1
fi

# ============================================
# vLLM tuning
# ============================================
export VLLM_LOGGING_LEVEL=DEBUG
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-1800}"
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-32}"

if [ -z "${VLLM_ENFORCE_EAGER:-}" ]; then
    gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    if [[ "${gpu_name}" == *"B200"* ]]; then
        VLLM_ENFORCE_EAGER="false"
    else
        VLLM_ENFORCE_EAGER="true"
    fi
fi

# ============================================
# Load .env
# ============================================
if [ ! -f "${PROJECT_DIR}/.env" ]; then
    echo "ERROR: .env não encontrado em ${PROJECT_DIR}/.env"
    echo "Crie-o com: OPENAI_API_KEY=... e HF_TOKEN=..."
    exit 1
fi
source "${PROJECT_DIR}/.env"
export HF_TOKEN="${HF_TOKEN:?HF_TOKEN não definido no .env}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"

JOB_ID="docker-$$"
LOGS_DIR="${PROJECT_DIR}/logs"
CNAME1=""
CNAME2=""
SERVERS_OVERRIDE=""

mkdir -p "${LOGS_DIR}" "${HOME}/hf-cache" "${PROJECT_DIR}/outputs"
cd "${PROJECT_DIR}"

echo "=========================================="
echo "Info Gainme Full Benchmark (Docker) - $(date)"
echo "GPUs: ${GPUS} (total: ${TOTAL_GPUS})"
echo "Seeker:  ${MODEL1_NAME} on GPU ${MODEL1_GPU} (port ${MODEL1_PORT})"
if [ "${MODE}" = "seeker_only" ]; then
    echo "Oracle:  ${MODEL2_NAME} (external, via servers.yaml)"
else
    echo "Oracle:  ${MODEL2_NAME} on GPU ${MODEL2_GPU} (port ${MODEL2_PORT})"
fi
echo "Configs: ${CONFIGS_TARGET}"
echo "  max_num_seqs=${VLLM_MAX_NUM_SEQS} | enforce_eager=${VLLM_ENFORCE_EAGER}"
echo "=========================================="
echo ""

# ============================================
# Cleanup trap — garante remoção dos containers
# ============================================
cleanup() {
    echo ""
    echo "Cleanup..."
    [ -n "${CNAME1}" ] && docker stop "${CNAME1}" 2>/dev/null && docker rm "${CNAME1}" 2>/dev/null
    [ -n "${CNAME2}" ] && docker stop "${CNAME2}" 2>/dev/null && docker rm "${CNAME2}" 2>/dev/null
    [ -n "${SERVERS_OVERRIDE}" ] && rm -f "${SERVERS_OVERRIDE}"
    echo "Done - $(date)"
    echo "=========================================="
}
trap cleanup EXIT

# ============================================
# Helpers
# ============================================
sanitize_name() { echo "$1" | tr '/: .' '-'; }

start_vllm_server() {
    local model=$1 name=$2 port=$3 gpu=$4 gpu_mem=$5 max_len=$6 log=$7 parser=${8:-""} tp=${9:-1}
    local cname="vllm-$(sanitize_name "${name}")-${JOB_ID}"
    echo "Starting ${name} (GPU ${gpu}, port ${port}, TP=${tp})..." >&2

    local vllm_args="--model ${model} --served-model-name ${name} --port ${port} --host 0.0.0.0 --gpu-memory-utilization ${gpu_mem} --max-num-seqs ${VLLM_MAX_NUM_SEQS} --max-model-len ${max_len} --tensor-parallel-size ${tp}"
    [ "${VLLM_ENFORCE_EAGER}" = "true" ] && vllm_args="${vllm_args} --enforce-eager"
    [ -n "${parser}" ] && vllm_args="${vllm_args} --reasoning-parser ${parser}"

    mkdir -p "$(dirname "${log}")" 2>/dev/null || true

    # VLLM_PRE_INSTALL: space-separated pip packages to install before starting vLLM.
    # Useful when the image's transformers is too old for a new model architecture.
    # Example: VLLM_PRE_INSTALL="--upgrade transformers"
    local pre_install="${VLLM_PRE_INSTALL:-}"
    if [ -n "${pre_install}" ]; then
        docker run -d \
            --name "${cname}" \
            --gpus all \
            -e CUDA_VISIBLE_DEVICES="${gpu}" \
            --ipc=host \
            --network=host \
            -v "${HOME}/hf-cache:/root/.cache/huggingface" \
            -v "${PROJECT_DIR}/.pip-cache:/root/.cache/pip" \
            -e HF_TOKEN="${HF_TOKEN}" \
            -e VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL}" \
            -e VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S}" \
            --entrypoint bash \
            "${VLLM_IMAGE}" \
            -c "pip install --quiet ${pre_install} && python3 -m vllm.entrypoints.openai.api_server ${vllm_args}" \
            > /dev/null 2>&1
    else
        docker run -d \
            --name "${cname}" \
            --gpus all \
            -e CUDA_VISIBLE_DEVICES="${gpu}" \
            --ipc=host \
            --network=host \
            -v "${HOME}/hf-cache:/root/.cache/huggingface" \
            -e HF_TOKEN="${HF_TOKEN}" \
            -e VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL}" \
            -e VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S}" \
            "${VLLM_IMAGE}" \
            ${vllm_args} > /dev/null 2>&1
    fi

    # Forward container logs to file in background
    docker logs -f "${cname}" >> "${log}" 2>&1 &

    echo "${cname}"
}

wait_vllm_ready() {
    local cname=$1 port=$2 name=$3 timeout=${4:-1800}
    local elapsed=0
    local log="${LOGS_DIR}/info-gainme-${JOB_ID}-vllm-${name}.log"
    echo "Waiting up to ${timeout}s for ${name} on port ${port} (container=${cname})..."
    while ! curl -s "http://localhost:${port}/v1/models" > /dev/null 2>&1; do
        if ! docker inspect --format '{{.State.Running}}' "${cname}" 2>/dev/null | grep -q "true"; then
            echo "ERROR: container ${cname} para ${name} morreu antes de estar pronto"
            tail -n 50 "${log}" 2>/dev/null || true
            return 1
        fi
        if [ ${elapsed} -ge ${timeout} ]; then
            echo "ERROR: ${name} não ficou pronto após ${timeout}s — abortando"
            tail -n 50 "${log}" 2>/dev/null || true
            docker stop "${cname}" 2>/dev/null
            return 1
        fi
        sleep 5
        elapsed=$((elapsed + 5))
    done
    echo "✓ ${name} pronto após ${elapsed}s"
}

# ============================================
# Start vLLM server(s)
# ============================================
CNAME1=$(start_vllm_server "${MODEL1}" "${MODEL1_NAME}" ${MODEL1_PORT} "${MODEL1_GPU}" ${MODEL1_GPU_MEM} ${MODEL1_MAX_LEN} \
    "${LOGS_DIR}/info-gainme-${JOB_ID}-vllm-${MODEL1_NAME}.log" "${MODEL1_REASONING_PARSER}" "${MODEL1_TP}")
wait_vllm_ready "${CNAME1}" ${MODEL1_PORT} "${MODEL1_NAME}" "${VLLM_ENGINE_READY_TIMEOUT_S}" || exit 1
echo ""

CNAME2=""
if [ "${MODE}" = "dual" ]; then
    CNAME2=$(start_vllm_server "${MODEL2}" "${MODEL2_NAME}" ${MODEL2_PORT} "${MODEL2_GPU}" ${MODEL2_GPU_MEM} ${MODEL2_MAX_LEN} \
        "${LOGS_DIR}/info-gainme-${JOB_ID}-vllm-${MODEL2_NAME}.log" "${MODEL2_REASONING_PARSER}" "${MODEL2_TP}")
    wait_vllm_ready "${CNAME2}" ${MODEL2_PORT} "${MODEL2_NAME}" "${VLLM_ENGINE_READY_TIMEOUT_S}" || exit 1
    echo ""
elif [ "${MODE}" = "single" ]; then
    MODEL2_NAME="${MODEL1_NAME}"
    MODEL2_PORT=${MODEL1_PORT}
    echo "(Single mode: using ${MODEL1_NAME} for all agents)"
    echo ""
else
    echo "(Seeker-only: oracle/pruner resolved from servers.yaml for ${MODEL2_NAME})"
    echo ""
fi

# ============================================
# Servers override file
# ============================================
echo "Creating servers override file..."
NODE_IP=$(python3 -c "import socket; print(socket.gethostbyname('$(hostname)'))" 2>/dev/null || echo "127.0.0.1")

# Host path (for creating the file) and container path (for passing to benchmark_runner.py,
# which runs inside Docker where PROJECT_DIR is mounted as /workspace).
SERVERS_OVERRIDE="${PROJECT_DIR}/.servers_override_${JOB_ID}.yaml"
SERVERS_OVERRIDE_CONTAINER="/workspace/.servers_override_${JOB_ID}.yaml"
if [ "${MODE}" = "dual" ]; then
    cat > "${SERVERS_OVERRIDE}" <<EOF
servers:
  ${MODEL1_NAME}: http://${NODE_IP}:${MODEL1_PORT}/v1
  ${MODEL2_NAME}: http://${NODE_IP}:${MODEL2_PORT}/v1
EOF
else
    cat > "${SERVERS_OVERRIDE}" <<EOF
servers:
  ${MODEL1_NAME}: http://${NODE_IP}:${MODEL1_PORT}/v1
EOF
fi

echo "  ✓ ${SERVERS_OVERRIDE} criado"
echo "  ✓ ${MODEL1_NAME} → http://${NODE_IP}:${MODEL1_PORT}/v1"
[ "${MODE}" = "dual" ] && echo "  ✓ ${MODEL2_NAME} → http://${NODE_IP}:${MODEL2_PORT}/v1"
echo ""

# ============================================
# Find configs
# ============================================
[[ "${CONFIGS_TARGET}" != /* ]] && CONFIGS_TARGET="${PROJECT_DIR}/${CONFIGS_TARGET}"
if [[ -f "${CONFIGS_TARGET}" ]]; then
    CONFIGS=("${CONFIGS_TARGET}")
elif [[ -d "${CONFIGS_TARGET}" ]]; then
    mapfile -t CONFIGS < <(find "${CONFIGS_TARGET}" -name "*.yaml" -type f | sort)
else
    echo "ERROR: '${CONFIGS_TARGET}' não encontrado"
    exit 1
fi

echo "=========================================="
echo "Running ${#CONFIGS[@]} benchmark config(s)"
echo "=========================================="
echo ""

# ============================================
# Run benchmarks
# ============================================
for CONFIG in "${CONFIGS[@]}"; do
    REL="${CONFIG#${PROJECT_DIR}/}"
    echo "[$(date '+%H:%M:%S')] ${REL}"
    RUNS_ARG=""
    [ -n "${RUNS_PER_TARGET:-}" ] && RUNS_ARG="--runs-per-target ${RUNS_PER_TARGET}"

    docker run --rm \
        --network=host \
        --entrypoint bash \
        -v "${PROJECT_DIR}:/workspace" \
        -v "${PROJECT_DIR}/.pip-cache:/root/.cache/pip" \
        -w /workspace \
        -e OPENAI_API_KEY="${OPENAI_API_KEY}" \
        -e HF_TOKEN="${HF_TOKEN}" \
        "${VLLM_IMAGE}" \
        -c "pip install --quiet -r requirements.txt 2>/dev/null && python3 benchmark_runner.py --config '${REL}' --servers-override '${SERVERS_OVERRIDE_CONTAINER}' ${RUNS_ARG}" \
        && echo "  ✓" || echo "  ✗"
    echo ""
done
