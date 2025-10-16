#!/bin/bash

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
    synth-param-estimation:latest \
    python src/train.py experiment=$exp paths=docker
done

echo "All experiments launched! Monitor via W&B dashboard."