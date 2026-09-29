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

results_dir=/path/to/verify-ppt-at-scale/evaluation/logs_lmeval
mkdir -p $results_dir
zero_shot_eval_tasks=(
    hellaswag
    copa
    record
    piqa
    lambada_openai
    race
    sciq
    blimp
)
few_shot_eval_tasks=(
    arc_easy
    openbookqa
    social_iqa
)

# Activate env
source /path/to/venv/verify-ppt_eval/bin/activate

# Run the evaluation script
for task in "${zero_shot_eval_tasks[@]}"; do
    echo "Evaluating task: $task"
    python -m lm_eval --model hf \
        --model_args "pretrained=$checkpoint_path,trust_remote_code=True" \
        --tasks $task \
        --num_fewshot 0 \
        --batch_size auto:4 \
        --device cuda:0 \
        --output_path $results_dir/${task}
done

for task in "${few_shot_eval_tasks[@]}"; do
    echo "Evaluating task: $task"
    python -m lm_eval --model hf \
        --model_args "pretrained=$checkpoint_path,trust_remote_code=True" \
        --tasks $task \
        --num_fewshot 5 \
        --batch_size auto:4 \
        --device cuda:0 \
        --output_path $results_dir/${task}
done
