#!/bin/bash

# Set up AWS CLI configuration
export AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
export AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
export AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION}
export AWS_ENDPOINT_URL=${AWS_ENDPOINT_URL}

# Create local data directory
mkdir -p ./data

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


# List of all flow_multi experiments
EXPERIMENTS=(
  "flow_multi/base_full"
  "flow_multi/dataset_50k"
  "flow_multi/dataset_100k"
  "flow_multi/dataset_50k_200k"
  "flow_multi/dataset_100k_1m"
  "flow_multi/full"
  "flow_multi/tokens_64"
  "flow_multi/tokens_256"
  "flow_multi/lr_high"
  "flow_multi/lr_low"
  "flow_multi/weight_decay"
  "flow_multi/batch_size_high"
  "flow_multi/batch_size_low"
  "flow_multi/ffn_small"
  "flow_multi/projection_penalty_low"
  "flow_multi/model_reduced_50"
  "flow_multi/model_reduced_75"
  "flow_multi/model_enlarged"
)

# Launch each experiment in parallel
for exp in "${EXPERIMENTS[@]}"; do
  echo "Launching experiment: $exp"
  docker run -d \
    --gpus all \
    --name "flow-multi-$(echo $exp | tr '/' '-')" \
    -e PROJECT_ROOT=/workspace \
    -e WANDB_API_KEY=$WANDB_API_KEY \
    -e S3_BUCKET=$S3_BUCKET \
    -e S3_PLUGIN_PATH=$S3_PLUGIN_PATH \
    -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
    -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
    -e AWS_DEFAULT_REGION=$AWS_DEFAULT_REGION \
    -e AWS_ENDPOINT_URL=$AWS_ENDPOINT_URL \
    -v $(pwd)/outputs:/workspace/outputs \
    -v $(pwd)/logs:/workspace/logs \
    benjamindupuis/synth-param-estimation:latest \
    python src/train.py experiment=$exp paths=docker
done

echo "All experiments launched! Monitor via W&B dashboard."