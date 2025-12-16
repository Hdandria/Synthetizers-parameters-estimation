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

# Split Creation Defaults
SOURCE_DATASET=""
TARGET_DATASET=""
TRAIN_SHARDS=""
VAL_SHARDS=""
TEST_SHARDS=""

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --env) ENV_FILE="$2"; shift 2 ;;
    --skip-build) SKIP_BUILD=true; shift ;;
    --stream) STREAM_LOGS=true; shift ;;
    --source) SOURCE_DATASET="$2"; shift 2 ;;
    --target) TARGET_DATASET="$2"; shift 2 ;;
    --tag) IMAGE_TAG="$2"; shift 2 ;;
    --train-shards) TRAIN_SHARDS="$2"; shift 2 ;;
    --val-shards) VAL_SHARDS="$2"; shift 2 ;;
    --test-shards) TEST_SHARDS="$2"; shift 2 ;;
    --help)
      echo "Usage: ./create_splits_cloud.sh [OPTIONS]"
      echo "Options:"
      echo "  --source PATH         Source dataset path (relative to datasets/)"
      echo "  --target PATH         Target dataset path (relative to datasets/)"
      echo "  --train-shards LIST   Comma-separated shard indices for training (e.g., '0,1,2,3,4')"
      echo "  --val-shards LIST     Comma-separated shard indices for validation (e.g., '5')"
      echo "  --test-shards LIST    Comma-separated shard indices for testing (e.g., '6')"
      echo "  --tag TAG             Docker image tag (default: splits-gen-latest)"
      echo "  --env FILE            Custom env file (default: .env)"
      echo "  --skip-build          Skip Docker build"
      echo "  --stream              Stream logs"
      echo ""
      echo "Example:"
      echo "  ./create_splits_cloud.sh \\"
      echo "    --source vital_100k \\"
      echo "    --target vital_20k \\"
      echo "    --train-shards 0,1 \\"
      echo "    --val-shards 10 \\"
      echo "    --test-shards 11"
      exit 0
      ;;
    *) 
      echo -e "${RED}Error: Unknown argument '$1'${RESET}"; exit 1;
      ;;
  esac
done

# Validate required arguments
[[ -z "$SOURCE_DATASET" ]] && { echo -e "${RED}Error: --source required${RESET}"; exit 1; }
[[ -z "$TARGET_DATASET" ]] && { echo -e "${RED}Error: --target required${RESET}"; exit 1; }
[[ -z "$TRAIN_SHARDS" && -z "$VAL_SHARDS" && -z "$TEST_SHARDS" ]] && { 
  echo -e "${RED}Error: At least one of --train-shards, --val-shards, or --test-shards required${RESET}"; 
  exit 1; 
}

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

echo -e "${CYAN}${BOLD}>>> Creating Dataset Splits: ${SOURCE_DATASET} → ${TARGET_DATASET}${RESET}"

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
IMAGE_TAG="${IMAGE_TAG:-splits-gen-latest}"
FULL_IMAGE="${REGISTRY_URL}/synth-param-estimation:${IMAGE_TAG}"

[[ -n "${DOCKER_USERNAME:-}" && -n "${DOCKER_PASSWORD:-}" ]] && \
  echo "${DOCKER_PASSWORD}" | docker login --username "${DOCKER_USERNAME}" --password-stdin 2>/dev/null || true

if [[ "$SKIP_BUILD" == false ]]; then
  echo -e "${BLUE}[*] Building image: ${FULL_IMAGE}${RESET}"
  docker build -t "$FULL_IMAGE" .
  docker push "$FULL_IMAGE"
fi

# Submit job
JOB_NAME="splits-$(echo "$TARGET_DATASET" | tr '/' '-')-$(date +%s)"
echo -e "${BLUE}[*] Submitting job: ${JOB_NAME}${RESET}"

# Build shard arguments for the Python script
SHARD_ARGS=""
[[ -n "$TRAIN_SHARDS" ]] && SHARD_ARGS+=" --train-shards '$TRAIN_SHARDS'"
[[ -n "$VAL_SHARDS" ]] && SHARD_ARGS+=" --val-shards '$VAL_SHARDS'"
[[ -n "$TEST_SHARDS" ]] && SHARD_ARGS+=" --test-shards '$TEST_SHARDS'"

ovhai job run \
  --name "${JOB_NAME}" \
  --flavor "${FLAVOR:-ai1-1-cpu}" \
  --cpu 4 \
  --volume "${S3_BUCKET_DATASETS}@${DS_ALIAS}:/workspace/datasets-mount:rw" \
  --env PROJECT_ROOT=/workspace \
  --env MPLCONFIGDIR=/tmp/matplotlib \
  --unsecure-http \
  --output json \
  "${FULL_IMAGE}" \
  -- bash -c 'set -euo pipefail
    
    MOUNT_BASE=/workspace/datasets-mount/datasets
    [ -d "$MOUNT_BASE" ] || MOUNT_BASE=/workspace/datasets-mount
    [ -d "$MOUNT_BASE" ] || { echo "ERROR: datasets mount not found"; exit 1; }
    
    SOURCE_PATH="${MOUNT_BASE}/'"${SOURCE_DATASET}"'"
    TARGET_PATH="${MOUNT_BASE}/'"${TARGET_DATASET}"'"
    
    echo "==> Source dataset: $SOURCE_PATH"
    echo "==> Target dataset: $TARGET_PATH"
    
    # Verify source exists
    if [ ! -d "$SOURCE_PATH" ]; then
      echo "ERROR: Source dataset not found at $SOURCE_PATH"
      exit 1
    fi
    
    # List source shards
    echo "==> Available shards in source:"
    ls -lh "$SOURCE_PATH"/shard*.h5 || ls -lh "$SOURCE_PATH"/shard-*.h5 || true
    
    # Create target directory
    mkdir -p "$TARGET_PATH"
    
    echo "==> Creating splits"
    echo "Train shards: '"${TRAIN_SHARDS}"'"
    echo "Val shards:   '"${VAL_SHARDS}"'"
    echo "Test shards:  '"${TEST_SHARDS}"'"
    
    cd /workspace
    python scripts/dataset/create_subset_dataset.py \
      "$SOURCE_PATH" \
      "$TARGET_PATH" \
      '"${SHARD_ARGS}"'
    
    echo "==> Created splits:"
    ls -lh "$TARGET_PATH"/*.h5
    
    echo "==> Done!"
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
