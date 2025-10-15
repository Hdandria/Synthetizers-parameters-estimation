#!/bin/bash

# Run training locally with Docker Compose
set -e

# Default experiment
EXPERIMENT=${1:-surge/base}

echo "Running training with experiment: $EXPERIMENT"
echo "Using Docker Compose..."

# Run the training service
docker-compose run --rm train-service python src/train.py experiment=$EXPERIMENT paths=docker

echo "Training completed!"
