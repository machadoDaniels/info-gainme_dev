#!/bin/bash
#SBATCH --job-name=vllm_conda
#SBATCH --partition=h100n3    
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=72G
#SBATCH --time=06:00:00
#SBATCH --ntasks=1              
#SBATCH --output=/raid/user_danielpedrozo/logs/%x-%j.out
#SBATCH --error=/raid/user_danielpedrozo/logs/%x-%j.err

# ===============================================
# CONFIGURAÇÕES DO MODELO - FACILITE A TROCA AQUI
# ===============================================
# Exemplos de modelos populares:
# export MODEL_NAME="Qwen/Qwen3-8B"           # Qwen3 8B
# export MODEL_NAME="Qwen/Qwen2.5-7B-Instruct" # Qwen2.5 7B Instruct
# export MODEL_NAME="meta-llama/Llama-2-7b-hf" # Llama 2 7B
# export MODEL_NAME="ftajwar/paprika_Meta-Llama-3.1-8B-Instruct" # Paprika Llama 3.1 8B

export MODEL_NAME="Qwen/Qwen3-30B-A3B-Thinking-2507"           # Nome completo do modelo no HuggingFace
export SERVED_NAME="Qwen3-30B-A3B-Thinking-2507"       
export GPU_MEMORY_UTIL=0.9                  # Uso da GPU (0.1-1.0)
export MAX_NUM_SEQS=32                      # Número máximo de sequências paralelas
export MAX_MODEL_LEN=140000                 # Comprimento máximo do contexto

# porta do servidor (interna ao nó)
export VLLM_PORT=8005

# variáveis de ambiente para vLLM
export VLLM_LOGGING_LEVEL=DEBUG

# Carrega conda diretamente sem usar module
source /cm/shared/apps/conda/etc/profile.d/conda.sh

# Inicializa conda para bash
eval "$(conda shell.bash hook)"

# Definir caminho personalizado para ambientes conda
export CONDA_ENVS_PATH=/raid/user_danielpedrozo/conda_envs/

# Verifica se o ambiente conda existe no caminho personalizado
CONDA_ENV_PATH="/raid/user_danielpedrozo/conda_envs/vllm_env"
if [ -d "$CONDA_ENV_PATH" ]; then
    echo "Ativando ambiente conda existente em $CONDA_ENV_PATH..."
    conda activate "$CONDA_ENV_PATH"
else
    echo "Não foi possível encontrar o ambiente conda vllm_env em /raid/user_danielpedrozo/conda_envs/..."
    exit 1
fi

# cache do HF no /raid para não baixar tudo de novo cada execução
export HF_HOME=/raid/user_danielpedrozo/hf-cache
source /raid/user_danielpedrozo/projects/info-gainme_dev/.env
export HF_TOKEN="${HF_TOKEN:?HF_TOKEN não definido no .env}"
export HOME=/raid/user_danielpedrozo/hf-cache

# garantir diretórios no /raid
mkdir -p /raid/user_danielpedrozo/logs
mkdir -p /raid/user_danielpedrozo/models
mkdir -p /raid/user_danielpedrozo/hf-cache

# Imprimir informações de conexão para o usuário
echo "=========================================="
echo "vLLM Server iniciando em: $(hostname)"
echo "Modelo: ${MODEL_NAME}"
echo "Nome da API: ${SERVED_NAME}"
echo "Porta: ${VLLM_PORT}"
echo "Uso da GPU: ${GPU_MEMORY_UTIL}"
echo ""
echo "Para criar túnel SSH da sua máquina local, execute:"
echo "  ssh -L LOCAL_PORT:$(hostname):${VLLM_PORT} user_danielpedrozo@dgx-H100-02"
echo "Depois acesse: http://localhost:LOCAL_PORT/v1/models"
echo "=========================================="

echo "GPUs disponíveis: $CUDA_VISIBLE_DEVICES"

# Suba o servidor vLLM sem Singularity
which vllm
vllm serve "${MODEL_NAME}" \
    --served-model-name "${SERVED_NAME}" \
    --download-dir /raid/user_danielpedrozo/models \
    --port ${VLLM_PORT} \
    --host 0.0.0.0 \
    --gpu-memory-utilization ${GPU_MEMORY_UTIL} \
    --max-num-seqs ${MAX_NUM_SEQS} \
    --max-model-len ${MAX_MODEL_LEN}
