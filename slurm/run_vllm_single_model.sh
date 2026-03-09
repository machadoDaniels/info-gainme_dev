#!/bin/bash
#SBATCH --job-name=akcit-rl-vllm
#SBATCH --partition=h100n2  
#SBATCH --gres=gpu:1
#SBATCH --mem=150G
#SBATCH --time=12:00:00
#SBATCH --output=/raid/user_danielpedrozo/projects/clary_quest/logs/%x-%j.out

# porta do servidor (interna ao nó)
export VLLM_PORT=8023
# Configuração do modelo
# export MODEL="Qwen/Qwen3-30B-A3B-Thinking-2507"
# export MODEL_NAME="Qwen3-30B-A3B-Thinking-2507"
# export MODEL_GPU_MEM=0.9
# export MODEL_REASONING_PARSER=""           
# export MODEL_MAX_LEN=140000                  


# export MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"
# export MODEL_NAME="Qwen3-30B-A3B-Instruct-2507"
# export MODEL_GPU_MEM=0.9
# export MODEL_REASONING_PARSER=""           
# export MODEL_MAX_LEN=140000     


export MODEL="Qwen/Qwen3-8B"
export MODEL_NAME="Qwen3-8B"
export MODEL_GPU_MEM=0.9
export MODEL_REASONING_PARSER="qwen3"           # Parser de reasoning opcional (ex: "o1", "qwen", etc.)
export MODEL_MAX_LEN=32000   



# variáveis de ambiente para vLLM
export VLLM_LOGGING_LEVEL=DEBUG
# Especificar GPU específica (descomente uma das linhas abaixo)
# export CUDA_VISIBLE_DEVICES=5,1


echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"

# cache do HF no /workspace para evitar problemas de permissão no /raid
export HF_HOME=/workspace/hf-cache
source /raid/user_danielpedrozo/projects/clary_quest/.env
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:?HUGGING_FACE_HUB_TOKEN não definido no .env}"

# garantir diretórios no /raid (estes serão mapeados para /workspace no container)
mkdir -p /raid/user_danielpedrozo/projects/clary_quest/logs
mkdir -p /raid/user_danielpedrozo/models
mkdir -p /raid/user_danielpedrozo/hf-cache

# Verificar GPUs disponíveis
echo "=========================================="
echo "Verificando GPUs disponíveis..."
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits
echo ""

# Imprimir informações de conexão para o usuário
echo "vLLM Server iniciando em: $(hostname)"
echo "Modelo: ${MODEL_NAME} na porta ${VLLM_PORT}"
echo ""
echo "Para criar túnel SSH da sua máquina local, execute:"
echo "  ssh -L LOCAL_PORT:$(hostname):${VLLM_PORT} user_danielpedrozo@dgx-H100-02"
echo ""
echo "Depois acesse:"
echo "  http://localhost:LOCAL_PORT/v1/models (${MODEL_NAME})"
echo "=========================================="

# Função para executar o servidor vLLM
start_vllm_server() {
    local model=$1
    local served_name=$2
    local port=$3
    local gpu_mem=$4
    local max_len=$5
    local log_file=$6
    local reasoning_parser=${7:-""}  # Parâmetro opcional, padrão vazio
    
    echo "Iniciando servidor para ${served_name} na porta ${port}..."
    
    # Criar diretório de log se não existir
    dir_name=$(dirname ${log_file})
    
    # Construir comando vLLM com parâmetros opcionais
    local vllm_cmd="/usr/bin/python3 -m vllm.entrypoints.openai.api_server \
      --model ${model} \
      --served-model-name ${served_name} \
      --download-dir /workspace/hf-cache/hub \
      --port ${port} \
      --host 0.0.0.0 \
      --gpu-memory-utilization ${gpu_mem} \
      --max-num-seqs 32 \
      --tensor-parallel-size 1 \
      --max-model-len ${max_len}"
    
    # Adicionar reasoning_parser se fornecido
    if [ -n "${reasoning_parser}" ]; then
        vllm_cmd="${vllm_cmd} --reasoning-parser ${reasoning_parser}"
    fi
    
    singularity exec \
         --nv \
         --bind /raid/user_danielpedrozo:/workspace \
         --bind "/usr/lib/x86_64-linux-gnu/libcuda.so.1:/usr/local/cuda/compat/lib/libcuda.so.1" \
         --pwd /workspace \
         --env HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN} \
         --env VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL} \
         /raid/user_danielpedrozo/images/vllm-openai_latest.sif \
         bash -c "mkdir -p ${dir_name} && ${vllm_cmd} > ${log_file} 2>&1" &
    
    echo "PID do processo ${served_name}: $!"
}

# Iniciar o servidor
start_vllm_server "${MODEL}" "${MODEL_NAME}" ${VLLM_PORT} ${MODEL_GPU_MEM} ${MODEL_MAX_LEN} "/workspace/projects/clary_quest/logs/${MODEL_NAME}.log" "${MODEL_REASONING_PARSER}"

# Aguardar o modelo carregar completamente
echo "Aguardando carregamento completo do modelo..."

while ! curl -s http://localhost:${VLLM_PORT}/v1/models > /dev/null; do
    echo "Aguardando carregamento completo do modelo..."
    sleep 5
done

echo "Servidor foi iniciado em background."
echo "Aguardando finalização do processo..."
echo "Para acessar os logs, execute:"
echo "  tail -f /raid/user_danielpedrozo/projects/clary_quest/logs/${MODEL_NAME}.log"

# Aguardar o processo terminar
wait
