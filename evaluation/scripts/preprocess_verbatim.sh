#!/bin/bash

# Configs / caches
export TRANSFORMERS_VERBOSITY=debug
export HF_HOME="/path/to/cache"
export HF_HUB_CACHE="/path/to/cache"
export HF_DATASETS_CACHE="/path/to/cache"
export HF_DATASETS_TRUST_REMOTE_CODE=true
export TMPDIR=/tmp

# Activate env
source /path/to/venv/verify-ppt_eval/bin/activate

# Run the preprocessing script
cd /path/to/verify-ppt/evaluation/src/verbatim
python preprocess_verbatim.py \
    --input-dir /path/to/verify-ppt/external/verbatim-memory-in-NLMs/data/rnn_input_files \
    --output-dir /path/to/data/verbatim
