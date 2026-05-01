#!/bin/bash
#SBATCH --job-name=akcit-rl-vllm-multi-model
#SBATCH --partition=h100n2          
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --time=4-00:00:00
#SBATCH --output=/raid/user_danielpedrozo/projects/info-gainme_dev/logs/%x-%j.out

# portas dos servidores (interna ao nó)
export VLLM_PORT1=8025
export VLLM_PORT2=8026
 



export MODEL1="Qwen/Qwen3-0.6B"
export MODEL1_NAME="Qwen3-0.6B"
export MODEL1_GPU_MEM=0.6
export MODEL1_MAX_LEN=32000                
export MODEL1_REASONING_PARSER=""   

export MODEL2="Qwen/Qwen3-4B-Instruct-2507"
export MODEL2_NAME="Qwen3-4B-Instruct-2507"
export MODEL2_GPU_MEM=0.35
export MODEL2_REASONING_PARSER=""           
export MODEL2_MAX_LEN=32000                 


         

# variáveis de ambiente para vLLM
export VLLM_LOGGING_LEVEL=DEBUG
# Especificar GPU específica (descomente uma das linhas abaixo)
# export CUDA_VISIBLE_DEVICES=2  # Usar GPU 1 (livre)
# export CUDA_VISIBLE_DEVICES=2  # Usar GPU 2  
# export CUDA_VISIBLE_DEVICES=3  # Usar GPU 3

# cache do HF no /workspace para evitar problemas de permissão no /raid
export HF_HOME=/workspace/hf-cache
source /raid/user_danielpedrozo/projects/info-gainme_dev/.env
export HF_TOKEN="${HF_TOKEN:?HF_TOKEN não definido no .env}"
export LOGS_DIR="/workspace/projects/info-gainme_dev/logs"


# garantir diretórios no /raid (estes serão mapeados para /workspace no container)
mkdir -p /raid/user_danielpedrozo/projects/info-gainme_dev/logs
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
         --bind /dev/shm:/dev/shm \
         --pwd /workspace \
         --env HF_TOKEN=${HF_TOKEN} \
         --env VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL} \
         --env HF_HOME=${HF_HOME} \
         /raid/user_danielpedrozo/images/vllm_openai_latest.sif \
         sh -c "mkdir -p ${dir_name} && ${vllm_cmd} > ${log_file} 2>&1" &
    
    echo "PID do processo ${served_name}: $!"
}

# Iniciar os dois servidores em background
start_vllm_server "${MODEL1}" "${MODEL1_NAME}" ${VLLM_PORT1} ${MODEL1_GPU_MEM} ${MODEL1_MAX_LEN} "${LOGS_DIR}/vllm_multi_model_1gpu-${SLURM_JOB_ID}_${MODEL1_NAME}.log" "${MODEL1_REASONING_PARSER}"

# Aguardar o primeiro modelo carregar completamente antes de iniciar o segundo
echo "Aguardando carregamento completo do primeiro modelo..."

while ! curl -s http://localhost:${VLLM_PORT1}/v1/models > /dev/null; do
    echo "Aguardando carregamento completo do primeiro modelo..."
    sleep 5
done

# echo "Iniciando segundo modelo..."

start_vllm_server "${MODEL2}" "${MODEL2_NAME}" ${VLLM_PORT2} ${MODEL2_GPU_MEM} ${MODEL2_MAX_LEN} "${LOGS_DIR}/vllm_multi_model_1gpu-${SLURM_JOB_ID}_${MODEL2_NAME}.log" "${MODEL2_REASONING_PARSER}"

while ! curl -s http://localhost:${VLLM_PORT2}/v1/models > /dev/null; do
    echo "Aguardando carregamento completo do segundo modelo..."
    sleep 5
done

echo "Ambos os servidores foram iniciados em background."
echo "Aguardando finalização dos processos..."
echo "Para acessar os logs, execute:"

# Acesso dos logs
echo "  tail -f /raid/user_danielpedrozo/projects/info-gainme_dev/logs/vllm_multi_model_1gpu-${SLURM_JOB_ID}_${MODEL1_NAME}.log"
echo "  tail -f /raid/user_danielpedrozo/projects/info-gainme_dev/logs/vllm_multi_model_1gpu-${SLURM_JOB_ID}_${MODEL2_NAME}.log"

# Aguardar ambos os processos terminarem
wait