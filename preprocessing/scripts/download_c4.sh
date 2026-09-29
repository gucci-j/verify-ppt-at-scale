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

# Run the preprocessing script
cd /path/to/verify-ppt/preprocessing/src
python tokenize_c4.py \
    --output-root "" \
    --logging-root "" \
    --cache-root /path/to/cache \
    --workers 4 \
    --c4-split "${C4_SPLIT:-train[:25%]}" \
    --c4-alias "${C4_ALIAS:-c4}" \
    --predownload \
    --predownload-only
