#!/bin/bash

# Docker evaluation wrapper script
# Usage: ./docker/run_eval.sh <checkpoint_path> [mode] [additional_args...]

set -e

# Check if checkpoint path is provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <checkpoint_path> [mode] [additional_args...]"
    echo "Example: $0 logs/train/runs/2024-01-01_12-00-00/checkpoints/best.ckpt test"
    exit 1
fi

CHECKPOINT_PATH="$1"
MODE="${2:-test}"
shift 2 2>/dev/null || shift 1
ADDITIONAL_ARGS="$@"

# Build the evaluation command
CMD="python src/eval.py ckpt_path=$CHECKPOINT_PATH mode=$MODE"

if [ -n "$ADDITIONAL_ARGS" ]; then
    CMD="$CMD $ADDITIONAL_ARGS"
fi

echo "Running evaluation command: $CMD"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Mode: $MODE"
echo "Using GPU: ${CUDA_VISIBLE_DEVICES:-0}"

# Run the evaluation
docker-compose run --rm train bash -c "$CMD"

