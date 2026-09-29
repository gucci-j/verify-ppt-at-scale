#!/bin/bash -l
#SBATCH --job-name=convert_nt_to_hf
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --time=01:00:00

CONTAINER="${CONTAINER:-/path/to/containers/vllm_26.01-py3.sif}"
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)"
export MASTER_PORT="${MASTER_PORT:-29500}"

checkpoint_steps=(
    "1000"
    "2000"
    "3000"
    "4000"
    "5000"
    "6000"
    "7000"
    "8000"
    "9000"
    "10000"
)

chmod +x /path/to/verify-ppt/evaluation/scripts/convert_nt_to_hf.sh
for step in "${checkpoint_steps[@]}"; do
    checkpoint_path="$1/${step}"
    singularity exec \
        -B /etc/ssl/certs:/etc/ssl/certs \
        -B /etc/pki/ca-trust:/etc/pki/ca-trust \
        -B $HOME:$HOME \
        --nv "${CONTAINER}" \
        /path/to/verify-ppt/evaluation/scripts/convert_nt_to_hf.sh $checkpoint_path
done
