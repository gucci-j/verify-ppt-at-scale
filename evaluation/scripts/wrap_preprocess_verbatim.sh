#!/bin/bash -l
#SBATCH --job-name=preprocess_verbatim
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30GB
#SBATCH --time=01:00:00

CONTAINER="${CONTAINER:-/path/to/containers/vllm_26.01-py3.sif}"

chmod +x /path/to/verify-ppt/evaluation/scripts/preprocess_verbatim.sh
singularity exec \
    -B /etc/ssl/certs:/etc/ssl/certs \
    -B /etc/pki/ca-trust:/etc/pki/ca-trust \
    -B $HOME:$HOME \
    "${CONTAINER}" \
    /path/to/verify-ppt/evaluation/scripts/preprocess_verbatim.sh
