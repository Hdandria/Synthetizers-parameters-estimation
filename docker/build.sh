#!/bin/bash

# Build Docker image for synthesizer parameter estimation
set -e

echo "Building Docker image for synthesizer parameter estimation..."

# Build the image
docker build -t synth-param-estimation:latest .

echo "Build completed successfully!"
echo "Image: synth-param-estimation:latest"

# Show image size
echo "Image size:"
docker images synth-param-estimation:latest --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"
