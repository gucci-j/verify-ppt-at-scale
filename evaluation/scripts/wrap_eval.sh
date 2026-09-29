#!/bin/bash -l
#SBATCH --job-name=eval
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64GB
#SBATCH --time=11:59:59


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

for step in "${checkpoint_steps[@]}"; do
    checkpoint_path="$1/${step}"
    checkpoint_path_hf=${checkpoint_path}_hf
    chmod +x /path/to/verify-ppt-at-scale/evaluation/scripts/eval.sh
    singularity exec \
        -B $HOME:$HOME \
        --nv /path/to/containers/vllm_26.01-py3.sif \
        /path/to/verify-ppt-at-scale/evaluation/scripts/eval.sh $checkpoint_path_hf

done
