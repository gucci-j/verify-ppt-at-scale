#!/bin/bash
set -euo pipefail


BASELINE_STEPS=${BASELINE_STEPS:-500}
# "hf": read existing .ds from the Hub ([hf-repo-prefix]/smollm3-<alias>).
# "local": read from ${SOURCE_LOCAL_BASE}/<alias> instead (faster if available).
SOURCE_MODE=${SOURCE_MODE:-hf}
SOURCE_LOCAL_BASE=${SOURCE_LOCAL_BASE:-/path/to/data/smollm3}
SOURCE_HF_PREFIX=${SOURCE_HF_PREFIX:-[hf-repo-prefix]/smollm3}

export TRANSFORMERS_VERBOSITY=info
export HF_HOME=/path/to/cache
export HF_HUB_CACHE=/path/to/cache
export HF_DATASETS_CACHE=/path/to/cache
export HF_DATASETS_TRUST_REMOTE_CODE=true
export TMPDIR=/tmp

source /path/to/venv/verify-ppt_training/bin/activate

cd /path/to/verify-ppt/preprocessing/src
python extract_baseline_subset.py \
    --config-yaml "/path/to/verify-ppt/training/configs/smollm3/smollm3_500m.yaml" \
    --baseline-steps "${BASELINE_STEPS}" \
    --output-base "/path/to/data/smollm3_baseline" \
    --source-mode "${SOURCE_MODE}" \
    --source-hf-prefix "${SOURCE_HF_PREFIX}" \
    --source-local-base "${SOURCE_LOCAL_BASE}" \
    --tokenizer "HuggingFaceTB/SmolLM3-3B" \
    --eos-token-id 128001 \
    --token-size 4 \
    --skip-existing \
    --skip-streaming-from tokenize_smollm3
