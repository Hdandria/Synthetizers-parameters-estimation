#!/bin/bash
set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
RESET='\033[0m'

# Parse arguments
EXPERIMENT_CONFIG=""
ENV_FILE=".env"
LOCAL_MODE=false
STREAM_LOGS=false
SKIP_BUILD=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --local) LOCAL_MODE=true; shift ;;
    --env) ENV_FILE="$2"; shift 2 ;;
    --stream) STREAM_LOGS=true; shift ;;
    --skip-build) SKIP_BUILD=true; shift ;;
    --help)
      echo "Usage: ./launch.sh <experiment> [--local|--stream|--skip-build]"
      exit 0
      ;;
    *)
      [[ -z "$EXPERIMENT_CONFIG" ]] && EXPERIMENT_CONFIG="$1" || { echo "Error: Unknown argument '$1'"; exit 1; }
      shift
      ;;
  esac
done

# Validate inputs
[[ -z "$EXPERIMENT_CONFIG" ]] && { echo -e "${RED}Error: No experiment specified${RESET}"; exit 1; }
[[ ! -f "configs/experiment/${EXPERIMENT_CONFIG}.yaml" ]] && { echo -e "${RED}Error: Config not found: $EXPERIMENT_CONFIG${RESET}"; exit 1; }
[[ ! -f "$ENV_FILE" ]] && { echo -e "${RED}Error: $ENV_FILE not found${RESET}"; exit 1; }

echo -e "${CYAN}${BOLD}>>> Launching experiment: ${EXPERIMENT_CONFIG}${RESET}"

# Load environment
set -a; source "$ENV_FILE"; set +a
echo -e "${GREEN}[+] Environment loaded from ${ENV_FILE}${RESET}"

# Required variables check
for var in WANDB_API_KEY AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_ENDPOINT_URL; do
  [[ -z "${!var:-}" ]] && { echo -e "${RED}Error: $var not set${RESET}"; exit 1; }
done
echo -e "${GREEN}[+] Required environment variables set${RESET}"

# Hardware config
NUM_GPUS="${NUM_GPUS:-1}"
GPU_IDS_TRIMMED="${GPU_IDS//[[:space:]]/}"
DATA_NUM_WORKERS="${DATA_NUM_WORKERS:-}"

# Build Hydra overrides for cloud (will add dataset_root override)
HYDRA_OVERRIDES_CLOUD=("experiment=${EXPERIMENT_CONFIG}" "trainer.accelerator=gpu" "trainer.devices=${NUM_GPUS}")
[[ -n "${DATA_NUM_WORKERS}" ]] && HYDRA_OVERRIDES_CLOUD+=("data.num_workers=${DATA_NUM_WORKERS}")

################################################################################
# LOCAL MODE
################################################################################
if [[ "$LOCAL_MODE" == true ]]; then
  echo -e "${MAGENTA}${BOLD}============================================================${RESET}"
  echo -e "${MAGENTA}${BOLD}                    LOCAL MODE                            ${RESET}"
  echo -e "${MAGENTA}${BOLD}============================================================${RESET}"

  if [[ "$SKIP_BUILD" == false ]]; then
    echo -e "${BLUE}[*] Building Docker image...${RESET}"
    docker build -t synth-param-estimation:latest .
    echo -e "${GREEN}[+] Docker image built${RESET}"
  else
    echo -e "${YELLOW}[-] Skipping Docker build${RESET}"
  fi

  DOCKER_GPUS_OPT="--gpus all"
  [[ -n "${GPU_IDS_TRIMMED}" ]] && DOCKER_GPUS_OPT="--gpus device=${GPU_IDS_TRIMMED}"
  [[ -z "${GPU_IDS_TRIMMED}" && -n "${NUM_GPUS}" ]] && DOCKER_GPUS_OPT="--gpus ${NUM_GPUS}"

  # Build Hydra overrides for local
  HYDRA_OVERRIDES=("experiment=${EXPERIMENT_CONFIG}" "trainer.accelerator=gpu" "trainer.devices=${NUM_GPUS}")
  [[ -n "${DATA_NUM_WORKERS}" ]] && HYDRA_OVERRIDES+=("data.num_workers=${DATA_NUM_WORKERS}")

  echo -e "${BLUE}[*] Running training locally with Docker...${RESET}"
  echo -e "${CYAN}    GPUs: ${NUM_GPUS}${RESET}"
  [[ -n "${DATA_NUM_WORKERS}" ]] && echo -e "${CYAN}    Workers: ${DATA_NUM_WORKERS}${RESET}"

  PY_ARGS=(python src/train.py "${HYDRA_OVERRIDES[@]}")

  docker run --rm ${DOCKER_GPUS_OPT} --shm-size=32G \
    -e PROJECT_ROOT=/workspace \
    -e WANDB_API_KEY="$WANDB_API_KEY" \
    -e AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID" \
    -e AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY" \
    -e AWS_ENDPOINT_URL="$AWS_ENDPOINT_URL" \
    -e AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-gra}" \
    $([ -n "${GPU_IDS_TRIMMED}" ] && echo "-e CUDA_VISIBLE_DEVICES=${GPU_IDS_TRIMMED}") \
    -v "$(pwd)/datasets:/workspace/datasets:ro" \
    -v "$(pwd)/outputs:/workspace/outputs" \
    -v "$(pwd)/logs:/workspace/logs" \
    synth-param-estimation:latest \
    "${PY_ARGS[@]}"

  echo -e "${GREEN}${BOLD}[+] Local training completed${RESET}"
  exit 0
fi

################################################################################
# CLOUD MODE
################################################################################

echo -e "${CYAN}${BOLD}============================================================${RESET}"
echo -e "${CYAN}${BOLD}                    CLOUD MODE (OVH)                      ${RESET}"
echo -e "${CYAN}${BOLD}============================================================${RESET}"

# Configure ovhai CLI
command -v ovhai &> /dev/null || { echo -e "${RED}Error: ovhai CLI not found${RESET}"; exit 1; }
echo -e "${GREEN}[+] ovhai CLI found${RESET}"

echo -e "${BLUE}[*] Checking OVH AI Training authentication...${RESET}"

# Check if logged in
if ! ovhai me &>/dev/null; then
  echo -e "${RED}Error: Not logged in to ovhai CLI${RESET}"
  echo -e "${YELLOW}Please login first: ovhai login${RESET}"
  exit 1
fi

echo -e "${GREEN}[+] Authenticated with OVH AI Training${RESET}"

# Configure S3 datastore
echo -e "${BLUE}[*] Configuring S3 datastore...${RESET}"
DATASTORE_ALIAS="s3-${OVH_REGION:-gra}"
if ! ovhai datastore list 2>/dev/null | grep -q "^${DATASTORE_ALIAS}"; then
  REGION_LOWER=$(echo "${OVH_REGION:-GRA}" | tr '[:upper:]' '[:lower:]')
  ovhai datastore add s3 "${DATASTORE_ALIAS}" "${AWS_ENDPOINT_URL}" "${REGION_LOWER}" \
    "${AWS_ACCESS_KEY_ID}" "${AWS_SECRET_ACCESS_KEY}" --store-credentials-locally
  echo -e "${GREEN}[+] S3 datastore added: ${DATASTORE_ALIAS}${RESET}"
else
  echo -e "${GREEN}[+] S3 datastore found: ${DATASTORE_ALIAS}${RESET}"
fi

# Get registry and bucket info from Terraform or fallback to env
if [[ -d terraform/.terraform ]]; then
  echo -e "${BLUE}[*] Reading configuration from Terraform...${RESET}"
  cd terraform
  REGISTRY_URL=$(terraform output -raw registry_url 2>/dev/null || echo "${DOCKER_REGISTRY}")
  S3_BUCKET_DATASETS=$(terraform output -raw s3_bucket_datasets 2>/dev/null || echo "${S3_BUCKET}")
  S3_BUCKET_OUTPUTS=$(terraform output -raw s3_bucket_outputs 2>/dev/null || echo "${S3_BUCKET_OUTPUTS}")
  cd ..
  echo -e "${GREEN}[+] Terraform configuration loaded${RESET}"
else
  echo -e "${YELLOW}[!] Terraform not initialized, using environment variables${RESET}"
  REGISTRY_URL="${DOCKER_REGISTRY}"
  S3_BUCKET_DATASETS="${S3_BUCKET}"
  S3_BUCKET_OUTPUTS="${S3_BUCKET_OUTPUTS}"
fi

echo -e "${CYAN}    Registry: ${REGISTRY_URL}${RESET}"
echo -e "${CYAN}    Datasets: ${S3_BUCKET_DATASETS}${RESET}"
echo -e "${CYAN}    Outputs: ${S3_BUCKET_OUTPUTS}${RESET}"

# Build and push image
IMAGE_TAG="$(echo "$EXPERIMENT_CONFIG" | tr '/' '-')-$(date +%Y%m%d-%H%M%S)"
FULL_IMAGE="${REGISTRY_URL}/synth-param-estimation:${IMAGE_TAG}"

[[ -n "${DOCKER_USERNAME:-}" && -n "${DOCKER_PASSWORD:-}" ]] && \
  echo "${DOCKER_PASSWORD}" | docker login --username "${DOCKER_USERNAME}" --password-stdin 2>/dev/null || true

if [[ "$SKIP_BUILD" == false ]]; then
  echo -e "${BLUE}[*] Building and pushing Docker image...${RESET}"
  echo -e "${CYAN}    Image: ${FULL_IMAGE}${RESET}"
  docker build -t "$FULL_IMAGE" .
  docker push "$FULL_IMAGE"
  echo -e "${GREEN}[+] Docker image pushed${RESET}"
else
  echo -e "${YELLOW}[-] Skipping Docker build and push${RESET}"
fi


# Submit job
echo -e "${BLUE}[*] Submitting job to OVH AI Training...${RESET}"
JOB_NAME="$(echo "$EXPERIMENT_CONFIG" | tr '/' '-')-$(date +%s)"
echo -e "${CYAN}    Job name: ${JOB_NAME}${RESET}"
echo -e "${CYAN}    Flavor: ${FLAVOR:-ai1-1-gpu}${RESET}"
echo -e "${CYAN}    GPUs: ${NUM_GPUS}${RESET}"
[[ -n "${DATA_NUM_WORKERS}" ]] && echo -e "${CYAN}    Workers: ${DATA_NUM_WORKERS}${RESET}"

ovhai job run \
  --name "${JOB_NAME}" \
  --flavor "${FLAVOR:-ai1-1-gpu}" \
  --gpu "${NUM_GPUS}" \
  --volume "${S3_BUCKET_DATASETS}@${DATASTORE_ALIAS}:/workspace/datasets-mount:ro" \
  --volume "${S3_BUCKET_OUTPUTS}@${DATASTORE_ALIAS}:/workspace/outputs:rw" \
  --env WANDB_API_KEY="${WANDB_API_KEY}" \
  --env PROJECT_ROOT=/workspace \
  --env MPLCONFIGDIR=/tmp/matplotlib \
  --env AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID}" \
  --env AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY}" \
  --env AWS_ENDPOINT_URL="${AWS_ENDPOINT_URL}" \
  --env AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-gra}" \
  $([ -n "${GPU_IDS_TRIMMED}" ] && echo "--env CUDA_VISIBLE_DEVICES=${GPU_IDS_TRIMMED}") \
  --unsecure-http \
  --output json \
  "${FULL_IMAGE}" \
  -- bash -c "\
      if [ -d /workspace/datasets-mount/datasets ]; then \
        MOUNT_BASE='/workspace/datasets-mount/datasets'; \
      elif [ -d /workspace/datasets-mount ]; then \
        MOUNT_BASE='/workspace/datasets-mount'; \
      else \
        echo 'ERROR: datasets mount not found'; exit 1; \
      fi && \
      CONFIG_FILE='configs/experiment/${EXPERIMENT_CONFIG}.yaml' && \
      DATASET_REL_PATH=\$(grep -oP 'dataset_root:\s*\K.*' \"\${CONFIG_FILE}\" | tr -d \"'\\\"\" | xargs) && \
      DATASET_NAME=\$(basename \"\${DATASET_REL_PATH}\") && \
      DATASET_PATH=\"\${MOUNT_BASE}/\${DATASET_NAME}\" && \
      echo \"Config dataset_root: \${DATASET_REL_PATH}\" && \
      echo \"Extracted dataset name: \${DATASET_NAME}\" && \
      echo \"Resolved absolute path: \${DATASET_PATH}\" && \
      ls -lah \"\${DATASET_PATH}/\" && \
      cd /workspace && \
      python src/train.py ${HYDRA_OVERRIDES_CLOUD[*]} data.dataset_root=\"\${DATASET_PATH}\"\
    " \
  | tee /tmp/job_output.json

JOB_ID=$(jq -r '.id // .uuid // empty' /tmp/job_output.json)
[[ -z "$JOB_ID" ]] && { echo -e "${RED}Error: Failed to get job ID${RESET}"; exit 1; }

echo ""
echo -e "${GREEN}${BOLD}[+] Job submitted successfully!${RESET}"
echo -e "${CYAN}------------------------------------------------------------${RESET}"
echo -e "${BOLD}Job ID:${RESET}    ${GREEN}${JOB_ID}${RESET}"
echo -e "${BOLD}Monitor:${RESET}   ${BLUE}./scripts/ovh/status.sh ${JOB_ID}${RESET}"
echo -e "${BOLD}Logs:${RESET}      ${BLUE}./scripts/ovh/logs.sh ${JOB_ID}${RESET}"
echo -e "${CYAN}------------------------------------------------------------${RESET}"

[[ "$STREAM_LOGS" == true ]] && { echo -e "${BLUE}[*] Streaming logs...${RESET}"; sleep 10; ovhai job logs "$JOB_ID" --follow; }
