#!/bin/bash
#SBATCH --job-name=info-gainme-benchmark
#SBATCH --partition=b200n1
#SBATCH --gres=gpu:0
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --output=/raid/user_danielpedrozo/projects/info-gainme_dev/logs/%x-%j.out

# ===============================================
# CONFIGURAÇÃO
# ===============================================
BENCHMARK_CONFIG="${1:-${BENCHMARK_CONFIG:-benchmark_config.yaml}}"

PROJECT_DIR="/raid/user_danielpedrozo/projects/info-gainme_dev"
SINGULARITY_IMAGE="/raid/user_danielpedrozo/images/vllm_sandbox"

# ===============================================
# MAIN
# ===============================================
mkdir -p "${PROJECT_DIR}/logs"

echo "=========================================="
echo "Info Gainme Benchmark - $(date)"
echo "Config: ${BENCHMARK_CONFIG}"
echo "Nó: $(hostname)"
echo "=========================================="

DEPS_DIR="${PROJECT_DIR}/.deps"
mkdir -p "${DEPS_DIR}"

singularity exec \
    --bind /raid/user_danielpedrozo:/workspace \
    --bind "/usr/lib/x86_64-linux-gnu/libcuda.so.1:/usr/local/cuda/compat/lib/libcuda.so.1" \
    --pwd /workspace/projects/info-gainme_dev \
    "${SINGULARITY_IMAGE}" \
    bash -c "
        pip install -r requirements.txt --target /workspace/projects/info-gainme_dev/.deps -q &&
        PYTHONPATH=/workspace/projects/info-gainme_dev/.deps:\$PYTHONPATH \
        /usr/bin/python3 benchmark_runner.py --config '${BENCHMARK_CONFIG}'
    "

echo "=========================================="
echo "Benchmark finalizado - $(date)"
echo "=========================================="
