#!/bin/bash

checkpoint_path=$1
if [ -z "$checkpoint_path" ]; then
    echo "Usage: $0 <checkpoint_path>"
    exit 1
fi

# Configs / caches
export TRANSFORMERS_VERBOSITY=debug
export HF_HOME="/path/to/cache"
export HF_HUB_CACHE="/path/to/cache"
export HF_DATASETS_CACHE="/path/to/cache"
export HF_DATASETS_TRUST_REMOTE_CODE=true
export TMPDIR=/tmp
export NCCL_DEBUG=INFO

results_base_dir=/path/to/verify-ppt-at-scale/evaluation/logs_verbatim
mkdir -p $results_base_dir
results_dir=$results_base_dir/$(echo "$checkpoint_path" | sed 's|/|__|g')

# Activate env
source /path/to/venv/verify-ppt_eval/bin/activate

# Run the evaluation script
cd /path/to/verify-ppt-at-scale/evaluation/src/verbatim
python eval_verbatim.py \
    --dataset-dir /path/to/data/verbatim/hf_dataset \
    --model $checkpoint_path \
    --output-dir $results_dir
