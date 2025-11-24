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
EXPERIMENT_CONFIG=""
ENV_FILE=".env"
LOCAL_MODE=false
STREAM_LOGS=false
SKIP_BUILD=false

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --local) LOCAL_MODE=true; shift ;;
    --env) ENV_FILE="$2"; shift 2 ;;
    --stream) STREAM_LOGS=true; shift ;;
    --skip-build) SKIP_BUILD=true; shift ;;
    --help)
      echo "Usage: ./launch.sh <experiment> [OPTIONS]"
      echo "Options:"
      echo "  --local       Run locally with Docker"
      echo "  --stream      Stream logs (cloud only)"
      echo "  --skip-build  Skip Docker build"
      echo "  --env FILE    Custom env file (default: .env)"
      exit 0
      ;;
    *) 
      [[ -z "$EXPERIMENT_CONFIG" ]] && EXPERIMENT_CONFIG="$1" || { echo -e "${RED}Error: Unknown argument '$1'${RESET}"; exit 1; }
      shift
      ;;
  esac
done

# Validate inputs
[[ -z "$EXPERIMENT_CONFIG" ]] && { echo -e "${RED}Error: No experiment specified${RESET}"; exit 1; }
[[ ! -f "configs/experiment/${EXPERIMENT_CONFIG}.yaml" ]] && { echo -e "${RED}Error: Config not found${RESET}"; exit 1; }
[[ ! -f "$ENV_FILE" ]] && { echo -e "${RED}Error: $ENV_FILE not found${RESET}"; exit 1; }

echo -e "${CYAN}${BOLD}>>> Launching: ${EXPERIMENT_CONFIG}${RESET}"

# Load environment
set -a; source "$ENV_FILE"; set +a
echo -e "${GREEN}[+] Environment loaded${RESET}"

# Verify required variables
for var in WANDB_API_KEY AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_ENDPOINT_URL; do
  [[ -z "${!var:-}" ]] && { echo -e "${RED}Error: $var not set${RESET}"; exit 1; }
done

# Config
NUM_GPUS="${NUM_GPUS:-1}"
GPU_IDS="${GPU_IDS//[[:space:]]/}"
DATA_NUM_WORKERS="${DATA_NUM_WORKERS:-}"
DATASET_CHECK_VERBOSE="${DATASET_CHECK_VERBOSE:-false}"

HYDRA_OVERRIDES=("experiment=${EXPERIMENT_CONFIG}" "trainer.accelerator=gpu" "trainer.devices=${NUM_GPUS}")
[[ -n "${DATA_NUM_WORKERS}" ]] && HYDRA_OVERRIDES+=("data.num_workers=${DATA_NUM_WORKERS}")

# Helper: Extract dataset root from config
extract_dataset_root() {
  local cfg="$1" ds=""
  ds=$(grep -oP 'dataset_root:\s*\K.*' "$cfg" 2>/dev/null | tr -d "'\"{} " | head -1)
  if [[ -z "$ds" ]]; then
    local data_ovr=$(grep -oP 'override /data:\s*\K.*' "$cfg" 2>/dev/null | head -1)
    [[ -n "$data_ovr" && -f "configs/data/${data_ovr}.yaml" ]] && \
      ds=$(grep -oP 'dataset_root:\s*\K.*' "configs/data/${data_ovr}.yaml" 2>/dev/null | tr -d "'\"{} " | head -1)
  fi
  echo "$ds"
}

# Helper: Check dataset
check_dataset() {
  local path="$1" args=()
  [[ "${DATASET_CHECK_VERBOSE}" != "true" ]] && args+=(--quiet)
  python scripts/dataset/test_readability.py "$path" "${args[@]}" || return 1
  echo -e "${GREEN}[+] Dataset check passed${RESET}"
}

########################################
# LOCAL MODE
########################################
if [[ "$LOCAL_MODE" == true ]]; then
  echo -e "${CYAN}>>> LOCAL MODE${RESET}"
  
  [[ "$SKIP_BUILD" == false ]] && docker build -t synth-param-estimation:latest .
  
  DS_REL=$(extract_dataset_root "configs/experiment/${EXPERIMENT_CONFIG}.yaml")
  [[ -z "$DS_REL" ]] && { echo -e "${RED}Error: No dataset_root in config${RESET}"; exit 1; }
  
  DS_PATH="$(pwd)/${DS_REL}"
  [[ ! -d "$DS_PATH" ]] && { echo -e "${RED}Error: Dataset not found: $DS_PATH${RESET}"; exit 1; }
  
  export HDF5_VDS_PREFIX="$(pwd)"
  check_dataset "$DS_PATH"
  
  GPU_OPT="--gpus all"
  [[ -n "$GPU_IDS" ]] && GPU_OPT="--gpus device=${GPU_IDS}"
  
  docker run --rm $GPU_OPT --shm-size=32G \
    -e PROJECT_ROOT=/workspace \
    -e HDF5_VDS_PREFIX=/workspace \
    -e WANDB_API_KEY="$WANDB_API_KEY" \
    -e AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID" \
    -e AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY" \
    -e AWS_ENDPOINT_URL="$AWS_ENDPOINT_URL" \
    -e AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-gra}" \
    $([ -n "$GPU_IDS" ] && echo "-e CUDA_VISIBLE_DEVICES=${GPU_IDS}") \
    -v "$(pwd)/datasets:/workspace/datasets:ro" \
    -v "$(pwd)/outputs:/workspace/outputs" \
    -v "$(pwd)/logs:/workspace/logs" \
    synth-param-estimation:latest \
    python src/train.py "${HYDRA_OVERRIDES[@]}"
  
  echo -e "${GREEN}[+] Training complete${RESET}"
  exit 0
fi

########################################
# CLOUD MODE
########################################
echo -e "${CYAN}>>> CLOUD MODE (OVH)${RESET}"

command -v ovhai &>/dev/null || { echo -e "${RED}Error: ovhai CLI not found${RESET}"; exit 1; }
ovhai me &>/dev/null || { echo -e "${RED}Error: Not logged in. Run: ovhai login${RESET}"; exit 1; }

# Configure S3 datastore
DS_ALIAS="s3-${OVH_REGION:-gra}"
REGION=$(echo "${OVH_REGION:-GRA}" | tr '[:upper:]' '[:lower:]')
if ovhai datastore list 2>/dev/null | grep -q "^${DS_ALIAS}"; then
  # Update existing datastore to ensure credentials are fresh
  ovhai datastore update s3 "${DS_ALIAS}" "${AWS_ENDPOINT_URL}" "${REGION}" \
    "${AWS_ACCESS_KEY_ID}" "${AWS_SECRET_ACCESS_KEY}" --store-credentials-locally
else
  # Create new datastore
  ovhai datastore add s3 "${DS_ALIAS}" "${AWS_ENDPOINT_URL}" "${REGION}" \
    "${AWS_ACCESS_KEY_ID}" "${AWS_SECRET_ACCESS_KEY}" --store-credentials-locally
fi

# Get config
if [[ -d terraform/.terraform ]]; then
  cd terraform
  REGISTRY_URL=$(terraform output -raw registry_url 2>/dev/null || echo "${DOCKER_REGISTRY}")
  S3_BUCKET_DATASETS=$(terraform output -raw s3_bucket_datasets 2>/dev/null || echo "${S3_BUCKET}")
  S3_BUCKET_OUTPUTS=$(terraform output -raw s3_bucket_outputs 2>/dev/null || echo "${S3_BUCKET_OUTPUTS}")
  cd ..
else
  REGISTRY_URL="${DOCKER_REGISTRY}"
  S3_BUCKET_DATASETS="${S3_BUCKET}"
  S3_BUCKET_OUTPUTS="${S3_BUCKET_OUTPUTS}"
fi

# Build & push image
IMAGE_TAG="$(echo "$EXPERIMENT_CONFIG" | tr '/' '-')-$(date +%Y%m%d-%H%M%S)"
FULL_IMAGE="${REGISTRY_URL}/synth-param-estimation:${IMAGE_TAG}"

[[ -n "${DOCKER_USERNAME:-}" && -n "${DOCKER_PASSWORD:-}" ]] && \
  echo "${DOCKER_PASSWORD}" | docker login --username "${DOCKER_USERNAME}" --password-stdin 2>/dev/null || true

if [[ "$SKIP_BUILD" == false ]]; then
  echo -e "${BLUE}[*] Building image: ${FULL_IMAGE}${RESET}"
  docker build -t "$FULL_IMAGE" .
  docker push "$FULL_IMAGE"
fi

# Submit job
JOB_NAME="$(echo "$EXPERIMENT_CONFIG" | tr '/' '-')-$(date +%s)"
echo -e "${BLUE}[*] Submitting job: ${JOB_NAME}${RESET}"

ovhai job run \
  --name "${JOB_NAME}" \
  --flavor "${FLAVOR:-ai1-1-gpu}" \
  --gpu "${NUM_GPUS}" \
  --volume "${S3_BUCKET_DATASETS}@${DS_ALIAS}:/workspace/datasets-mount:ro" \
  --volume "${S3_BUCKET_OUTPUTS}@${DS_ALIAS}:/workspace/outputs:rw" \
  --env WANDB_API_KEY="${WANDB_API_KEY}" \
  --env PROJECT_ROOT=/workspace \
  --env MPLCONFIGDIR=/tmp/matplotlib \
  --env HDF5_VDS_PREFIX=/workspace/datasets-mount \
  --env AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID}" \
  --env AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY}" \
  --env AWS_ENDPOINT_URL="${AWS_ENDPOINT_URL}" \
  --env AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-gra}" \
  --env DATASET_CHECK_VERBOSE="${DATASET_CHECK_VERBOSE}" \
  $([ -n "$GPU_IDS" ] && echo "--env CUDA_VISIBLE_DEVICES=${GPU_IDS}") \
  --unsecure-http \
  --output json \
  "${FULL_IMAGE}" \
  -- bash -c 'set -euo pipefail
    MOUNT_BASE=/workspace/datasets-mount/datasets
    [ -d "$MOUNT_BASE" ] || MOUNT_BASE=/workspace/datasets-mount
    [ -d "$MOUNT_BASE" ] || { echo "ERROR: datasets mount not found"; exit 1; }
    
    CFG=configs/experiment/'"${EXPERIMENT_CONFIG}"'.yaml
    DS_REL=$(grep -oP "dataset_root:\s*\K.*" "$CFG" | tr -d "'\''\" " | head -1)
    
    if [ -z "$DS_REL" ]; then
      DATA_OVR=$(grep -oP "override /data:\s*\K.*" "$CFG" | head -1)
      if [ -n "$DATA_OVR" ] && [ -f "configs/data/${DATA_OVR}.yaml" ]; then
        DS_REL=$(grep -oP "dataset_root:\s*\K.*" "configs/data/${DATA_OVR}.yaml" | tr -d "'\''\" " | head -1)
      fi
    fi
    
    [ -z "$DS_REL" ] && { echo "ERROR: No dataset_root in config"; exit 1; }
    
    DS_NAME=$(basename "$DS_REL")
    DS_PATH="${MOUNT_BASE}/${DS_NAME}"
    
    echo "==> Dataset: $DS_PATH"
    ls -lah "$DS_PATH/" || true
    
    echo "==> Checking readability"
    QUIET=""
    [ "${DATASET_CHECK_VERBOSE:-}" != "true" ] && QUIET="--quiet"
    python scripts/dataset/test_readability.py "$DS_PATH" $QUIET
    
    echo "==> Starting training"
    cd /workspace
    python src/train.py '"${HYDRA_OVERRIDES[*]}"' data.dataset_root="$DS_PATH"
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
