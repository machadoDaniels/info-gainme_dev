#!/bin/bash
# Faz upload do diretório outputs/ para o HuggingFace Dataset usando Singularity.
# Uso:
#   ./dgx/upload_to_hf.sh                          # upload completo
#   ./dgx/upload_to_hf.sh --dry-run                # simula sem enviar
#   ./dgx/upload_to_hf.sh --repo-id akcit-rl/outro # repo alternativo
#   ./dgx/upload_to_hf.sh --num-workers 16         # mais workers

umask 002

SHARED_GROUP="sd22"
PROJECT_DIR="/raid/user_danielpedrozo/projects/info-gainme_dev"
SINGULARITY_IMAGE="/raid/user_danielpedrozo/images/vllm_openai_latest.sif"

# Repassa todos os argumentos CLI para o script Python
EXTRA_ARGS="$*"

echo "=========================================="
echo "Info Gainme - Upload to HuggingFace - $(date)"
echo "Repo padrão: akcit-rl/info-gainme"
if [ -n "${EXTRA_ARGS}" ]; then
    echo "Args extras: ${EXTRA_ARGS}"
fi
echo "=========================================="

sg "${SHARED_GROUP}" -c "
    singularity exec \
        --bind /raid/user_danielpedrozo:/workspace \
        --pwd /workspace/projects/info-gainme_dev \
        '${SINGULARITY_IMAGE}' \
        bash -c \"
            pip install --quiet --user huggingface_hub python-dotenv
            python3 scripts/upload_to_hf.py ${EXTRA_ARGS}
        \"
"

echo "=========================================="
echo "Upload finalizado - $(date)"
echo "=========================================="
