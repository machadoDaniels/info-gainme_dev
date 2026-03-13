#!/bin/bash
#SBATCH --job-name=info-gainme-benchmark
#SBATCH --partition=h100n2
#SBATCH --gres=gpu:0
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --output='/raid/user_danielpedrozo/projects/info-gainme_dev/logs/%x-%j.out'

# Garante que novos arquivos herdem permissão de grupo (g+rw)
umask 002

# ===============================================
# CONFIGURAÇÃO
# ===============================================
BENCHMARK_CONFIG="${1:-${BENCHMARK_CONFIG:-configs/geo_full_no_cot.yaml}}"
SHARED_GROUP="sd22"

PROJECT_DIR="/raid/user_danielpedrozo/projects/info-gainme_dev"
SINGULARITY_IMAGE="/raid/user_danielpedrozo/images/vllm_openai_latest.sif"

# ===============================================
# MAIN
# ===============================================
mkdir -p "${PROJECT_DIR}/logs"
mkdir -p "${PROJECT_DIR}/.deps"
mkdir -p "${PROJECT_DIR}/outputs"

# Garante permissão de escrita para todos nos diretórios de saída
mkdir -p "${PROJECT_DIR}/outputs/logs"
chmod o+rwx "${PROJECT_DIR}/outputs" "${PROJECT_DIR}/outputs/logs" 2>/dev/null

echo "=========================================="
echo "Info Gainme Benchmark - $(date)"
echo "Config: ${BENCHMARK_CONFIG}"
echo "Nó: $(hostname)"
echo "=========================================="

sg "${SHARED_GROUP}" -c "
    singularity exec \
        --bind /raid/user_danielpedrozo:/workspace \
        --bind '/usr/lib/x86_64-linux-gnu/libcuda.so.1:/usr/local/cuda/compat/lib/libcuda.so.1' \
        --pwd /workspace/projects/info-gainme_dev \
        '${SINGULARITY_IMAGE}' \
        bash -c \"
            pip install --user -r requirements.txt
            echo 'iniciando benchmark...'
            python3 benchmark_runner.py --config '${BENCHMARK_CONFIG}'
        \"
"

echo "=========================================="
echo "Benchmark finalizado - $(date)"
echo "=========================================="
