#!/bin/bash
#SBATCH --job-name=clary-quest-benchmark
#SBATCH --partition=h100n2
#SBATCH --gres=gpu:0
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --output=/raid/user_danielpedrozo/projects/info-gainme_dev/logs/%x-%j.out
#SBATCH --error=/raid/user_danielpedrozo/projects/info-gainme_dev/logs/%x-%j.err

# ===============================================
# CONFIGURAÇÃO
# ===============================================
BENCHMARK_CONFIG="${BENCHMARK_CONFIG:-benchmark_config.yaml}"

PROJECT_DIR="/raid/user_danielpedrozo/projects/info-gainme_dev"
SINGULARITY_IMAGE="/raid/user_danielpedrozo/images/vllm-openai_latest.sif"

# ===============================================
# MAIN
# ===============================================
mkdir -p "${PROJECT_DIR}/logs"

echo "=========================================="
echo "Clary Quest Benchmark - $(date)"
echo "Config: ${BENCHMARK_CONFIG}"
echo "Nó: $(hostname)"
echo "=========================================="

singularity exec \
    --bind /raid/user_danielpedrozo:/workspace \
    --bind "/usr/lib/x86_64-linux-gnu/libcuda.so.1:/usr/local/cuda/compat/lib/libcuda.so.1" \
    --pwd /workspace/projects/info-gainme_dev \
    "${SINGULARITY_IMAGE}" \
    /usr/bin/python3 benchmark_runner.py --config "${BENCHMARK_CONFIG}"

echo "=========================================="
echo "Benchmark finalizado - $(date)"
echo "=========================================="
