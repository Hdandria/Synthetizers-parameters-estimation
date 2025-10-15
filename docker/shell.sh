#!/bin/bash

# Docker interactive shell script
# Usage: ./docker/shell.sh

set -e

echo "Starting interactive shell in Docker container..."
echo "Working directory: /workspace"
echo "Using GPU: ${CUDA_VISIBLE_DEVICES:-0}"
echo ""
echo "Available commands:"
echo "  python src/train.py experiment=surge/baseline"
echo "  python src/eval.py ckpt_path=path/to/checkpoint.ckpt"
echo "  python src/data/vst/generate_vst_dataset.py --help"
echo "  nvidia-smi  # Check GPU status"
echo ""

# Start interactive shell
docker-compose run --rm shell

