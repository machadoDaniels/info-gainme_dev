#!/bin/bash
#SBATCH --job-name=vllm_multi_model_1gpu
#SBATCH --partition=h100n2          
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=8:00:00
#SBATCH --ntasks=1              
#SBATCH --output=/raid/user_danielpedrozo/logs/%x-%j.out
#SBATCH --error=/raid/user_danielpedrozo/logs/%x-%j.err

# portas dos servidores (interna ao nó)
export VLLM_PORT1=8001
export VLLM_PORT2=8002
 

export MODEL1="Qwen/Qwen3-8B"
export MODEL1_NAME="Qwen3-8B"
export MODEL1_GPU_MEM=0.9
export MODEL1_REASONING_PARSER=""           # Parser de reasoning opcional (ex: "o1", "qwen", etc.)
export MODEL1_MAX_LEN=32000                  # Comprimento máximo do contexto


export MODEL2="meta-llama/Llama-3.1-8B-Instruct"
export MODEL2_NAME="Llama-3.1-8B-Instruct"
# export MODEL2_GPU_MEM=0.3
export MODEL2_MAX_LEN=28000                  # Comprimento máximo do contexto
export MODEL2_REASONING_PARSER=""            # Parser de reasoning opcional (ex: "o1", "qwen", etc.)

# variáveis de ambiente para vLLM
export VLLM_LOGGING_LEVEL=DEBUG
# Especificar GPU específica (descomente uma das linhas abaixo)
# export CUDA_VISIBLE_DEVICES=2  # Usar GPU 1 (livre)
# export CUDA_VISIBLE_DEVICES=2  # Usar GPU 2  
# export CUDA_VISIBLE_DEVICES=3  # Usar GPU 3

# cache do HF no /workspace para evitar problemas de permissão no /raid
export HF_HOME=/workspace/hf-cache
source /raid/user_danielpedrozo/projects/clary_quest/.env
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:?HUGGING_FACE_HUB_TOKEN não definido no .env}"

# garantir diretórios no /raid (estes serão mapeados para /workspace no container)
mkdir -p /raid/user_danielpedrozo/logs
mkdir -p /raid/user_danielpedrozo/models
mkdir -p /raid/user_danielpedrozo/hf-cache

# Verificar GPUs disponíveis
echo "=========================================="
echo "Verificando GPUs disponíveis..."
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits
echo ""

# Imprimir informações de conexão para o usuário
echo "vLLM Servers iniciando em: $(hostname)"
echo "Dois modelos serão executados na mesma GPU:"
echo "  Modelo 1 (${MODEL1_NAME}): porta ${VLLM_PORT1} (42% VRAM)"
echo "  Modelo 2 (${MODEL2_NAME}): porta ${VLLM_PORT2} (42% VRAM)"
echo ""
echo "Para criar túneis SSH da sua máquina local, execute:"
echo "  ssh -L LOCAL_PORT1:$(hostname):${VLLM_PORT1} user_danielpedrozo@dgx-H100-02"
echo "  ssh -L LOCAL_PORT2:$(hostname):${VLLM_PORT2} user_danielpedrozo@dgx-H100-02"
echo ""
echo "Depois acesse:"
echo "  http://localhost:LOCAL_PORT1/v1/models (${MODEL1_NAME})"
echo "  http://localhost:LOCAL_PORT2/v1/models (${MODEL2_NAME})"
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
      --max-num-seqs 16 \
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

# Iniciar os dois servidores em background
start_vllm_server "${MODEL1}" "${MODEL1_NAME}" ${VLLM_PORT1} ${MODEL1_GPU_MEM} ${MODEL1_MAX_LEN} "/workspace/logs/${MODEL1_NAME}.log" "${MODEL1_REASONING_PARSER}"

# Aguardar o primeiro modelo carregar completamente antes de iniciar o segundo
echo "Aguardando carregamento completo do primeiro modelo..."

while ! curl -s http://localhost:${VLLM_PORT1}/v1/models > /dev/null; do
    echo "Aguardando carregamento completo do primeiro modelo..."
    sleep 5
done

# echo "Iniciando segundo modelo..."

# start_vllm_server "${MODEL2}" "${MODEL2_NAME}" ${VLLM_PORT2} ${MODEL2_GPU_MEM} ${MODEL2_MAX_LEN} "/workspace/logs/${MODEL2_NAME}.log" "${MODEL2_REASONING_PARSER}"

# while ! curl -s http://localhost:${VLLM_PORT2}/v1/models > /dev/null; do
#     echo "Aguardando carregamento completo do segundo modelo..."
#     sleep 5
# done

echo "Ambos os servidores foram iniciados em background."
echo "Aguardando finalização dos processos..."
echo "Para acessar os logs, execute:"

# Acesso dos logs
echo "  tail -f /raid/user_danielpedrozo/logs/${MODEL1_NAME}.log"
echo "  tail -f /raid/user_danielpedrozo/logs/${MODEL2_NAME}.log"

# Aguardar ambos os processos terminarem
wait