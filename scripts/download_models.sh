#!/usr/bin/env bash
# Download the two pretrained backbones at the exact revisions used on A10 and verify every file.
# Usage: bash scripts/download_models.sh [models_dir]        (default: ./models)
# Mainland China: export HF_ENDPOINT=https://hf-mirror.com before running.
set -euo pipefail
cd "$(dirname "$0")/.."
DEST="${1:-models}"
mkdir -p "$DEST"
huggingface-cli download Qwen/Qwen2.5-7B-Instruct \
    --revision a09a35458c702b33eeacc393d103063234e8bc28 \
    --local-dir "$DEST/Qwen2.5-7B-Instruct" \
    --include '*.json' '*.txt' '*.safetensors'
huggingface-cli download google/siglip-base-patch16-224 \
    --revision 7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed \
    --local-dir "$DEST/siglip-base-patch16-224" \
    --include '*.json' '*.model' 'model.safetensors'
# The local-dir download cache must not sit inside the model folders.
rm -rf "$DEST/Qwen2.5-7B-Instruct/.cache" "$DEST/siglip-base-patch16-224/.cache"
(cd "$DEST" && sha256sum -c "$OLDPWD/scripts/models.sha256")
echo "models ready in $DEST"
