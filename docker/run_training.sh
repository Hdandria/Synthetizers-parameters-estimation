#!/bin/bash

# Docker training wrapper script
# Usage: ./docker/run_training.sh [experiment] [additional_args...]

set -e

# Default values
EXPERIMENT=""
ADDITIONAL_ARGS=""

# Parse arguments
if [ $# -gt 0 ]; then
    EXPERIMENT="$1"
    shift
    ADDITIONAL_ARGS="$@"
fi

# Build the training command
CMD="python src/train.py"

if [ -n "$EXPERIMENT" ]; then
    CMD="$CMD experiment=$EXPERIMENT"
fi

if [ -n "$ADDITIONAL_ARGS" ]; then
    CMD="$CMD $ADDITIONAL_ARGS"
fi

echo "Running training command: $CMD"
echo "Using GPU: ${CUDA_VISIBLE_DEVICES:-0}"

# Run the training
docker-compose run --rm train bash -c "$CMD"

