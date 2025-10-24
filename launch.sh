#!/bin/bash
set -euo pipefail

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
[[ -z "$EXPERIMENT_CONFIG" ]] && { echo "Error: No experiment specified"; exit 1; }
[[ ! -f "configs/experiment/${EXPERIMENT_CONFIG}.yaml" ]] && { echo "Error: Config not found: $EXPERIMENT_CONFIG"; exit 1; }
[[ ! -f "$ENV_FILE" ]] && { echo "Error: $ENV_FILE not found"; exit 1; }

# Load environment
set -a; source "$ENV_FILE"; set +a

# Required variables check
for var in WANDB_API_KEY AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_ENDPOINT_URL OVH_APPLICATION_KEY OVH_APPLICATION_SECRET OVH_CONSUMER_KEY OVH_PROJECT_ID; do
  [[ -z "${!var:-}" ]] && { echo "Error: $var not set"; exit 1; }
done

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
  [[ "$SKIP_BUILD" == false ]] && docker build -t synth-param-estimation:latest .
  
  DOCKER_GPUS_OPT="--gpus all"
  [[ -n "${GPU_IDS_TRIMMED}" ]] && DOCKER_GPUS_OPT="--gpus device=${GPU_IDS_TRIMMED}"
  [[ -z "${GPU_IDS_TRIMMED}" && -n "${NUM_GPUS}" ]] && DOCKER_GPUS_OPT="--gpus ${NUM_GPUS}"

  # Build Hydra overrides for local
  HYDRA_OVERRIDES=("experiment=${EXPERIMENT_CONFIG}" "trainer.accelerator=gpu" "trainer.devices=${NUM_GPUS}")
  [[ -n "${DATA_NUM_WORKERS}" ]] && HYDRA_OVERRIDES+=("data.num_workers=${DATA_NUM_WORKERS}")
  
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
  
  exit 0
fi

################################################################################
# CLOUD MODE
################################################################################

# Configure ovhai CLI
command -v ovhai &> /dev/null || { echo "Error: ovhai CLI not found"; exit 1; }

mkdir -p ~/.ovhcloud
cat > ~/.ovhcloud/config.yaml <<EOF
default:
  endpoint: ${OVH_ENDPOINT:-ovh-eu}
  application_key: ${OVH_APPLICATION_KEY}
  application_secret: ${OVH_APPLICATION_SECRET}
  consumer_key: ${OVH_CONSUMER_KEY}
  project: ${OVH_PROJECT_ID}
EOF

# Configure S3 datastore
DATASTORE_ALIAS="s3-${OVH_REGION:-gra}"
if ! ovhai datastore list 2>/dev/null | grep -q "^${DATASTORE_ALIAS}"; then
  REGION_LOWER=$(echo "${OVH_REGION:-GRA}" | tr '[:upper:]' '[:lower:]')
  ovhai datastore add s3 "${DATASTORE_ALIAS}" "${AWS_ENDPOINT_URL}" "${REGION_LOWER}" \
    "${AWS_ACCESS_KEY_ID}" "${AWS_SECRET_ACCESS_KEY}" --store-credentials-locally
fi

# Get registry and bucket info from Terraform or fallback to env
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

# Build and push image
IMAGE_TAG="$(echo "$EXPERIMENT_CONFIG" | tr '/' '-')-$(date +%Y%m%d-%H%M%S)"
FULL_IMAGE="${REGISTRY_URL}/synth-param-estimation:${IMAGE_TAG}"

[[ -n "${DOCKER_USERNAME:-}" && -n "${DOCKER_PASSWORD:-}" ]] && \
  echo "${DOCKER_PASSWORD}" | docker login --username "${DOCKER_USERNAME}" --password-stdin 2>/dev/null || true

if [[ "$SKIP_BUILD" == false ]]; then
  docker build -t "$FULL_IMAGE" .
  docker push "$FULL_IMAGE"
fi


# Submit job
JOB_NAME="$(echo "$EXPERIMENT_CONFIG" | tr '/' '-')-$(date +%s)"

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
[[ -z "$JOB_ID" ]] && { echo "Failed to get job ID"; exit 1; }

echo "Job ID: ${JOB_ID}"
echo "Monitor: ./scripts/status.sh ${JOB_ID}"
echo "Logs: ./scripts/logs.sh ${JOB_ID}"

[[ "$STREAM_LOGS" == true ]] && { sleep 10; ovhai job logs "$JOB_ID" --follow; }
