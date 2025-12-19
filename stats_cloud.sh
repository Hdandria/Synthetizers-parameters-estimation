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

# Stats Generation Defaults
DATASET=""
SPLIT="train"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --env) ENV_FILE="$2"; shift 2 ;;
    --skip-build) SKIP_BUILD=true; shift ;;
    --stream) STREAM_LOGS=true; shift ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --split) SPLIT="$2"; shift 2 ;;
    --tag) IMAGE_TAG="$2"; shift 2 ;;
    --help)
      echo "Usage: ./stats_cloud.sh [OPTIONS]"
      echo "Options:"
      echo "  --dataset PATH    Dataset path relative to datasets/ (e.g., 'vital_100k')"
      echo "  --split NAME      Split file to compute stats from (default: 'train')"
      echo "  --tag TAG         Docker image tag (default: stats-gen-latest)"
      echo "  --env FILE        Custom env file (default: .env)"
      echo "  --skip-build      Skip Docker build"
      echo "  --stream          Stream logs"
      echo ""
      echo "Example:"
      echo "  ./stats_cloud.sh --dataset vital_100k"
      exit 0
      ;;
    *) 
      echo -e "${RED}Error: Unknown argument '$1'${RESET}"; exit 1;
      ;;
  esac
done

# Validate required arguments
[[ -z "$DATASET" ]] && { echo -e "${RED}Error: --dataset required${RESET}"; exit 1; }

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

echo -e "${CYAN}${BOLD}>>> Computing Dataset Stats: ${DATASET}/${SPLIT}.h5${RESET}"

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
IMAGE_TAG="${IMAGE_TAG:-stats-gen-latest}"
FULL_IMAGE="${REGISTRY_URL}/synth-param-estimation:${IMAGE_TAG}"

[[ -n "${DOCKER_USERNAME:-}" && -n "${DOCKER_PASSWORD:-}" ]] && \
  echo "${DOCKER_PASSWORD}" | docker login --username "${DOCKER_USERNAME}" --password-stdin 2>/dev/null || true

if [[ "$SKIP_BUILD" == false ]]; then
  echo -e "${BLUE}[*] Building image: ${FULL_IMAGE}${RESET}"
  docker build -t "$FULL_IMAGE" .
  docker push "$FULL_IMAGE"
fi

# Submit job
JOB_NAME="stats-$(echo "$DATASET" | tr '/' '-')-$(date +%s)"
echo -e "${BLUE}[*] Submitting job: ${JOB_NAME}${RESET}"

NUM_WORKERS="${DATA_NUM_WORKERS:-8}"

ovhai job run \
  --name "${JOB_NAME}" \
  --flavor "${FLAVOR:-ai1-1-cpu}" \
  --cpu "${NUM_WORKERS}" \
  --volume "${S3_BUCKET_DATASETS}@${DS_ALIAS}:/workspace/datasets-mount:rw" \
  --env PROJECT_ROOT=/workspace \
  --env MPLCONFIGDIR=/tmp/matplotlib \
  --env HDF5_VDS_PREFIX=/workspace/datasets-mount/datasets \
  --unsecure-http \
  --output json \
  "${FULL_IMAGE}" \
  -- bash -c 'set -euo pipefail
    
    MOUNT_BASE=/workspace/datasets-mount/datasets
    [ -d "$MOUNT_BASE" ] || MOUNT_BASE=/workspace/datasets-mount
    [ -d "$MOUNT_BASE" ] || { echo "ERROR: datasets mount not found"; exit 1; }
    
    DATASET_PATH="${MOUNT_BASE}/'"${DATASET}"'"
    SPLIT_FILE="${DATASET_PATH}/'"${SPLIT}"'.h5"
    
    echo "==> Dataset path: $DATASET_PATH"
    echo "==> Split file: $SPLIT_FILE"
    
    # Verify dataset exists
    if [ ! -d "$DATASET_PATH" ]; then
      echo "ERROR: Dataset not found at $DATASET_PATH"
      exit 1
    fi
    
    # List files
    echo "==> Files in dataset:"
    ls -lh "$DATASET_PATH"/*.h5 || true
    
    # Verify split file exists
    if [ ! -f "$SPLIT_FILE" ]; then
      echo "ERROR: Split file not found at $SPLIT_FILE"
      exit 1
    fi
    
    cd /workspace
    python scripts/dataset/get_dataset_stats.py "$SPLIT_FILE"
    
    echo "==> Stats file created:"
    ls -lh "$DATASET_PATH"/stats.npz
    
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
