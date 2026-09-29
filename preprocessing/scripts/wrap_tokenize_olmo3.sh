#!/bin/bash
#SBATCH --job-name=tokenize_olmo3
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --time=12:00:00

set -euo pipefail

SCRIPT_DIR="/path/to/verify-ppt/preprocessing/scripts"
PREPROCESS_SCRIPT="${SCRIPT_DIR}/tokenize_olmo3.sh"
CONTAINER=${CONTAINER:-/path/to/containers/vllm_26.01-py3.sif}

chmod +x "${PREPROCESS_SCRIPT}"

singularity exec \
    -B /etc/ssl/certs:/etc/ssl/certs \
    -B /etc/pki/ca-trust:/etc/pki/ca-trust \
    -B "$HOME:$HOME" \
    "$CONTAINER" \
    /bin/bash "${PREPROCESS_SCRIPT}" "$@"
