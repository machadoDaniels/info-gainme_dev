#!/bin/bash
#SBATCH --job-name=akcit-rl-vllm-single-model
#SBATCH --partition=b200n1
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --time=12:00:00
#SBATCH --output=/raid/user_danielpedrozo/projects/info-gainme_dev/logs/%x-%j.out

# porta do servidor (interna ao nó)
export VLLM_PORT=8029
# Configuração do modelo
# export MODEL="Qwen/Qwen3-30B-A3B-Thinking-2507"
# export MODEL_NAME="Qwen3-30B-A3B-Thinking-2507"
# export MODEL_GPU_MEM=0.9
# export MODEL_REASONING_PARSER=""           
# export MODEL_MAX_LEN=140000                  


export MODEL="Qwen/Qwen3-30B-A3B-Thinking-2507"
export MODEL_NAME="Qwen3-30B-A3B-Thinking-2507"
export MODEL_GPU_MEM=0.9
export MODEL_REASONING_PARSER=""           
export MODEL_MAX_LEN=32000     


# export MODEL="Qwen/Qwen3-8B"
# export MODEL_NAME="Qwen3-8B"
# export MODEL_GPU_MEM=0.9
# export MODEL_REASONING_PARSER="qwen3"           # Parser de reasoning opcional (ex: "o1", "qwen", etc.)
# export MODEL_MAX_LEN=32000   



# variáveis de ambiente para vLLM
export VLLM_LOGGING_LEVEL=DEBUG
# Especificar GPU específica (descomente uma das linhas abaixo)
# export CUDA_VISIBLE_DEVICES=5,1


echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "SLURM_GPUS_ON_NODE: ${SLURM_GPUS_ON_NODE}"

# cache do HF no /workspace para evitar problemas de permissão no /raid
export HF_HOME=/workspace/hf-cache
source /raid/user_danielpedrozo/projects/info-gainme_dev/.env
export HF_TOKEN="${HF_TOKEN:?HF_TOKEN não definido no .env}"

# garantir diretórios no /raid (estes serão mapeados para /workspace no container)
mkdir -p /raid/user_danielpedrozo/projects/info-gainme_dev/logs
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

# Matar qualquer processo usando a porta antes de iniciar
echo "Verificando se a porta ${VLLM_PORT} está em uso..."
fuser -k ${VLLM_PORT}/tcp 2>/dev/null && echo "Processo anterior na porta ${VLLM_PORT} encerrado." || echo "Porta ${VLLM_PORT} está livre."
sleep 2

# Iniciar o servidor vLLM
echo "Iniciando servidor para ${MODEL_NAME} na porta ${VLLM_PORT}..."

# Construir comando vLLM
vllm_cmd="/usr/bin/python3 -m vllm.entrypoints.openai.api_server \
  --model ${MODEL} \
  --served-model-name ${MODEL_NAME} \
  --download-dir /workspace/hf-cache/hub \
  --port ${VLLM_PORT} \
  --host 0.0.0.0 \
  --gpu-memory-utilization ${MODEL_GPU_MEM} \
  --max-num-seqs 32 \
  --tensor-parallel-size ${SLURM_GPUS_ON_NODE} \
  --max-model-len ${MODEL_MAX_LEN}"

# Adicionar reasoning_parser se fornecido
if [ -n "${MODEL_REASONING_PARSER}" ]; then
    vllm_cmd="${vllm_cmd} --reasoning-parser ${MODEL_REASONING_PARSER}"
fi

singularity exec \
     --nv \
     --bind /raid/user_danielpedrozo:/workspace \
     --bind "/usr/lib/x86_64-linux-gnu/libcuda.so.1:/usr/local/cuda/compat/lib/libcuda.so.1" \
     --pwd /workspace \
     --env HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN} \
     --env VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL} \
     /raid/user_danielpedrozo/images/vllm_openai_latest.sif \
     bash -c "${vllm_cmd}" &

echo "PID do processo ${MODEL_NAME}: $!"

# Aguardar o modelo carregar completamente
echo "Aguardando carregamento completo do modelo..."

while ! curl -s http://localhost:${VLLM_PORT}/v1/models > /dev/null; do
    echo "Aguardando carregamento completo do modelo..."
    sleep 5
done

echo "Servidor foi iniciado em background."
echo "Aguardando finalização do processo..."

# Aguardar o processo terminar
wait