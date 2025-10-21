#!/bin/bash

# Set up AWS CLI configuration
export AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
export AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
export AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION}
export AWS_ENDPOINT_URL=${AWS_ENDPOINT_URL}

# Basic sanity checks
set -euo pipefail

command -v aws >/dev/null 2>&1 || {
  echo "Error: aws CLI not found on host. Install AWS CLI v2 and re-run." >&2
  exit 1
}

if [[ -z "${S3_BUCKET:-}" || -z "${AWS_ENDPOINT_URL:-}" ]]; then
  echo "Error: S3_BUCKET and AWS_ENDPOINT_URL must be set in environment (.env)." >&2
  exit 1
fi

# Create local data directory
mkdir -p ./data
mkdir -p ./datasets
mkdir -p ./plugins
# Ensure outputs and logs exist and are writable for the container user
mkdir -p ./outputs ./logs
# Make writeable by any user inside the container to avoid UID/GID mismatch issues
chmod -R a+rwx ./outputs ./logs || true

# Download all required datasets
echo "Downloading datasets from S3..."

# Download surge-20k dataset
echo "Downloading surge-20k..."
aws s3 sync s3://${S3_BUCKET}/datasets/surge-20k/ ./datasets/surge-20k/ --endpoint-url=${AWS_ENDPOINT_URL} 

# Download surge-50k dataset  
echo "Downloading surge-50k..."
aws s3 sync s3://${S3_BUCKET}/datasets/surge-50k/ ./datasets/surge-50k/ --endpoint-url=${AWS_ENDPOINT_URL}

# Download surge-100k dataset
echo "Downloading surge-100k..."
aws s3 sync s3://${S3_BUCKET}/datasets/surge-100k/ ./datasets/surge-100k/ --endpoint-url=${AWS_ENDPOINT_URL}

echo "Dataset download complete!"

# Optionally download plugin (if present in bucket)
if aws s3 ls s3://${S3_BUCKET}/plugins/ --endpoint-url=${AWS_ENDPOINT_URL} >/dev/null 2>&1; then
  echo "Downloading plugins from S3..."
  aws s3 sync s3://${S3_BUCKET}/plugins/ ./plugins/ --endpoint-url=${AWS_ENDPOINT_URL}
  echo "Plugin download complete!"
else
  echo "No plugins/ directory found in bucket ${S3_BUCKET}; skipping plugin download."
fi

# Clean up any existing containers from previous runs
echo "Cleaning up old containers..."
docker ps -a --format '{{.Names}}' | grep '^flow-multi-' | xargs -r -I{} docker rm -f {} 2>/dev/null || true
echo "Cleanup complete!"

# List of all flow_multi experiments
EXPERIMENTS=(
  "flow_multi/full"
  "flow_multi/dataset_50k_200k"
  "flow_multi/dataset_100k_1m"
  "flow_multi/model_reduced_50"
)

# Run experiments sequentially (one at a time)
echo "Running ${#EXPERIMENTS[@]} experiments sequentially..."
for exp in "${EXPERIMENTS[@]}"; do
  container_name="flow-multi-$(echo $exp | tr '/' '-')"
  echo ""
  echo "========================================="
  echo "Starting experiment: $exp"
  echo "Container: $container_name"
  echo "========================================="
  
  # Run container (not detached, waits for completion)
  docker run --rm \
    --gpus all \
    --shm-size=32G \
    --name "$container_name" \
    -e PROJECT_ROOT=/workspace \
    -e WANDB_API_KEY=$WANDB_API_KEY \
    -e S3_BUCKET=$S3_BUCKET \
    -e S3_PLUGIN_PATH=$S3_PLUGIN_PATH \
    -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
    -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
    -e AWS_DEFAULT_REGION=$AWS_DEFAULT_REGION \
    -e AWS_ENDPOINT_URL=$AWS_ENDPOINT_URL \
    -v $(pwd)/datasets:/workspace/datasets:ro \
    -v $(pwd)/plugins:/workspace/plugins:ro \
    -v $(pwd)/outputs:/workspace/outputs \
    -v $(pwd)/logs:/workspace/logs \
    benjamindupuis/synth-param-estimation:latest \
    python src/train.py experiment=$exp
  
  exit_code=$?
  if [ $exit_code -eq 0 ]; then
    echo "✓ Experiment $exp completed successfully"
  else
    echo "✗ Experiment $exp failed with exit code $exit_code"
    echo "  Continuing to next experiment..."
  fi
done

echo ""
echo "========================================="
echo "All experiments completed!"
echo "Check outputs in ./outputs and W&B dashboard."
echo "========================================="