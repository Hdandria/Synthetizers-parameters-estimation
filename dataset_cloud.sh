#!/bin/bash
set -euo pipefail

# Colors
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly CYAN='\033[0;36m'
readonly BOLD='\033[1m'
readonly RESET='\033[0m'

# Configuration
ENV_FILE=".env"
SKIP_BUILD=false
STREAM_LOGS=false

# Dataset Generation Defaults
START_SHARD=0
NUM_SHARDS=13
SAMPLES_PER_SHARD=10000
DATASET_NAME="vital_20k"
PRESET_DIR="/workspace/datasets-mount/presets/vital"
PLUGIN_PATH="plugins/Vital.vst3"
WORKERS=11
VARIANCE=0.1
PARAM_SPEC="vital_simple"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --env) ENV_FILE="$2"; shift 2 ;;
    --skip-build) SKIP_BUILD=true; shift ;;
    --stream) STREAM_LOGS=true; shift ;;
    --name) DATASET_NAME="$2"; shift 2 ;;
    --tag) IMAGE_TAG="$2"; shift 2 ;;
    --start-shard) START_SHARD="$2"; shift 2 ;;
    --num-shards) NUM_SHARDS="$2"; shift 2 ;;
    --samples) SAMPLES_PER_SHARD="$2"; shift 2 ;;
    --help)
      echo "Usage: ./dataset_cloud.sh [OPTIONS]"
      echo "Options:"
      echo "  --name NAME       Dataset name (default: vital_20k)"
      echo "  --tag TAG         Docker image tag (default: dataset-gen-latest)"
      echo "  --start-shard N   Starting shard index (default: 0)"
      echo "  --num-shards N    Number of shards to generate in this job (default: 13)"
      echo "  --samples N       Samples per shard (default: 10000)"
      echo "  --env FILE        Custom env file (default: .env)"
      echo "  --skip-build      Skip Docker build"
      echo "  --stream          Stream logs"
      exit 0
      ;;
    *) 
      echo -e "${RED}Error: Unknown argument '$1'${RESET}"; exit 1;
      ;;
  esac
done

# Load environment
if [[ -f "$ENV_FILE" ]]; then
    set -a; source "$ENV_FILE"; set +a
    echo -e "${GREEN}[+] Environment loaded from $ENV_FILE${RESET}"
else
    echo -e "${RED}Error: $ENV_FILE not found${RESET}"; exit 1;
fi

# Verify required variables
for var in WANDB_API_KEY AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_ENDPOINT_URL; do
  [[ -z "${!var:-}" ]] && { echo -e "${RED}Error: $var not set${RESET}"; exit 1; }
done

echo -e "${CYAN}${BOLD}>>> Launching Dataset Generation: ${DATASET_NAME}${RESET}"

# OVH Setup
command -v ovhai &>/dev/null || { echo -e "${RED}Error: ovhai CLI not found${RESET}"; exit 1; }
ovhai me &>/dev/null || { echo -e "${RED}Error: Not logged in. Run: ovhai login${RESET}"; exit 1; }

# Configure S3 datastore
DS_ALIAS="s3-${OVH_REGION:-gra}"
REGION=$(echo "${OVH_REGION:-GRA}" | tr '[:upper:]' '[:lower:]')
if ovhai datastore list 2>/dev/null | grep -q "^${DS_ALIAS}"; then
  ovhai datastore update s3 "${DS_ALIAS}" "${AWS_ENDPOINT_URL}" "${REGION}" \
    "${AWS_ACCESS_KEY_ID}" "${AWS_SECRET_ACCESS_KEY}" --store-credentials-locally
else
  ovhai datastore add s3 "${DS_ALIAS}" "${AWS_ENDPOINT_URL}" "${REGION}" \
    "${AWS_ACCESS_KEY_ID}" "${AWS_SECRET_ACCESS_KEY}" --store-credentials-locally
fi

# Get config (Terraform or Env)
if [[ -d terraform/.terraform ]]; then
  cd terraform
  REGISTRY_URL=$(terraform output -raw registry_url 2>/dev/null || echo "${DOCKER_REGISTRY}")
  S3_BUCKET_DATASETS=$(terraform output -raw s3_bucket_datasets 2>/dev/null || echo "${S3_BUCKET}")
  cd ..
else
  REGISTRY_URL="${DOCKER_REGISTRY}"
  S3_BUCKET_DATASETS="${S3_BUCKET}"
fi

# Build & push image
IMAGE_TAG="${IMAGE_TAG:-dataset-gen-latest}"
FULL_IMAGE="${REGISTRY_URL}/synth-param-estimation:${IMAGE_TAG}"

[[ -n "${DOCKER_USERNAME:-}" && -n "${DOCKER_PASSWORD:-}" ]] && \
  echo "${DOCKER_PASSWORD}" | docker login --username "${DOCKER_USERNAME}" --password-stdin 2>/dev/null || true

if [[ "$SKIP_BUILD" == false ]]; then
  echo -e "${BLUE}[*] Building image: ${FULL_IMAGE}${RESET}"
  docker build -t "$FULL_IMAGE" .
  docker push "$FULL_IMAGE"
fi

# Submit job
JOB_NAME="gen-${DATASET_NAME}-$(date +%s)"
echo -e "${BLUE}[*] Submitting job: ${JOB_NAME}${RESET}"

# Note: Mounting datasets bucket as RW to save the new dataset
ovhai job run \
  --name "${JOB_NAME}" \
  --flavor "${FLAVOR:-ai1-1-cpu}" \
  --cpu 12 \
  --volume "${S3_BUCKET_DATASETS}@${DS_ALIAS}:/workspace/datasets-mount:rw" \
  --env PROJECT_ROOT=/workspace \
  --env MPLCONFIGDIR=/tmp/matplotlib \
  --unsecure-http \
  --output json \
  "${FULL_IMAGE}" \
  -- bash -c 'set -euo pipefail
    
    OUTPUT_DIR="/workspace/datasets-mount/'"${DATASET_NAME}"'"
    mkdir -p "$OUTPUT_DIR"
    
    START_SHARD='"$START_SHARD"'
    NUM_SHARDS='"$NUM_SHARDS"'
    END_SHARD=$((START_SHARD + NUM_SHARDS - 1))

    echo "Starting dataset generation..."
    echo "Range: Shards $START_SHARD to $END_SHARD"
    echo "Samples per shard: '"$SAMPLES_PER_SHARD"'"
    echo "Output directory: $OUTPUT_DIR"
    
    for i in $(seq $START_SHARD $END_SHARD); do
        SHARD_FILE="$OUTPUT_DIR/shard_$i.h5"
        
        if [ -f "$SHARD_FILE" ]; then
            echo "Shard $i already exists at $SHARD_FILE. Skipping..."
            continue
        fi

        echo "--------------------------------------------------"
        echo "Generating shard $i ($SHARD_FILE)..."
        echo "--------------------------------------------------"
        
        python src/data/vst/generate_preset_dataset.py \
            "$SHARD_FILE" \
            '"$SAMPLES_PER_SHARD"' \
            --preset_dir "'"$PRESET_DIR"'" \
            --num_workers '"$WORKERS"' \
            --plugin_path "'"$PLUGIN_PATH"'" \
            --perturbation_variance '"$VARIANCE"' \
            --param_spec "'"$PARAM_SPEC"'"
            
        echo "Shard $i completed."
    done
    
    echo "All shards processed."
  ' \
  | tee /tmp/job_output.json

JOB_ID=$(jq -r '.id // .uuid // empty' /tmp/job_output.json)
[[ -z "$JOB_ID" ]] && { echo -e "${RED}Error: Failed to get job ID${RESET}"; exit 1; }

echo ""
echo -e "${GREEN}${BOLD}[+] Job submitted${RESET}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo -e "Job ID:  ${GREEN}${JOB_ID}${RESET}"
echo -e "Status:  ${BLUE}./scripts/ovh/status.sh ${JOB_ID}${RESET}"
echo -e "Logs:    ${BLUE}./scripts/ovh/logs.sh ${JOB_ID}${RESET}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"

[[ "$STREAM_LOGS" == true ]] && { sleep 10; ovhai job logs "$JOB_ID" --follow; }
