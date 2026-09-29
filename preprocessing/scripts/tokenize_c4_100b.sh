#!/bin/bash
set -euo pipefail

# Set configurations
export TRANSFORMERS_VERBOSITY=debug
export HF_HOME=/path/to/cache
export HF_HUB_CACHE=/path/to/cache
export HF_DATASETS_CACHE=/path/to/cache
export HF_DATASETS_TRUST_REMOTE_CODE=true
export TMPDIR=/tmp

# Activate the virtual environment
source /path/to/venv/verify-ppt_training/bin/activate

# Job-array sharding (Slurm-agnostic): the wrapper exports these per array task.
# Defaults run the whole job in one process, so this script still works standalone.
TASKS="${TASKS:-128}"
LOCAL_TASKS="${LOCAL_TASKS:-${TASKS}}"
LOCAL_RANK_OFFSET="${LOCAL_RANK_OFFSET:-0}"

# Run the preprocessing script
cd /path/to/verify-ppt-at-scale/preprocessing/src
python tokenize_c4.py \
    --output-root  "/path/to/data/c4" \
    --logging-root "/path/to/verify-ppt-at-scale/preprocessing/logs" \
    --cache-root   "/path/to/cache" \
    --workers 8 \
    --c4-split "${C4_SPLIT:-train[:65%]}" \
    --c4-alias "${C4_ALIAS:-c4_65}" \
    --tasks "${TASKS}" \
    --local-tasks "${LOCAL_TASKS}" \
    --local-rank-offset "${LOCAL_RANK_OFFSET}"
