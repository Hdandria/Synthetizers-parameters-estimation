#!/bin/bash
################################################################################
# Launch ML Training Job
# Usage: ./launch.sh <experiment> [--local|--stream|--skip-build]
################################################################################

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Default values
EXPERIMENT_CONFIG=""
ENV_FILE=".env"
LOCAL_MODE=false
STREAM_LOGS=false
SKIP_BUILD=false

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --local)
      LOCAL_MODE=true
      shift
      ;;
    --env)
      ENV_FILE="$2"
      shift 2
      ;;
    --stream)
      STREAM_LOGS=true
      shift
      ;;
    --skip-build)
      SKIP_BUILD=true
      shift
      ;;
    --help)
      grep "^#" "$0" | grep -v "^#!/" | sed 's/^# \?//'
      exit 0
      ;;
    *)
      if [[ -z "$EXPERIMENT_CONFIG" ]]; then
        EXPERIMENT_CONFIG="$1"
      else
        echo -e "${RED}Error: Unknown argument '$1'${NC}"
        exit 1
      fi
      shift
      ;;
  esac
done

# Validate experiment config provided
if [[ -z "$EXPERIMENT_CONFIG" ]]; then
  echo -e "${RED}Error: No experiment config specified${NC}"
  echo "Usage: ./launch.sh <experiment_config> [options]"
  echo "Example: ./launch.sh surge/base"
  exit 1
fi

# Validate experiment config exists
CONFIG_FILE="configs/experiment/${EXPERIMENT_CONFIG}.yaml"
if [[ ! -f "$CONFIG_FILE" ]]; then
  echo -e "${RED}Error: Config file not found: $CONFIG_FILE${NC}"
  echo "Available experiments:"
  find configs/experiment -name "*.yaml" -type f | sed 's|configs/experiment/||' | sed 's|.yaml||' | sort
  exit 1
fi

echo -e "${GREEN}🚀 Launching Experiment: ${EXPERIMENT_CONFIG}${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Load environment variables
if [[ ! -f "$ENV_FILE" ]]; then
  echo -e "${RED}Error: Environment file not found: $ENV_FILE${NC}"
  echo "Create one from the example: cp .env.example .env"
  exit 1
fi

echo -e "${BLUE}📄 Loading environment from: $ENV_FILE${NC}"
set -a
source "$ENV_FILE"
set +a

# Export OVH credentials for ovhai CLI and Terraform
export OVH_ENDPOINT="${OVH_ENDPOINT:-ovh-eu}"
export OVH_APPLICATION_KEY
export OVH_APPLICATION_SECRET
export OVH_CONSUMER_KEY

# Export Terraform variables from .env
export TF_VAR_ovh_endpoint="${OVH_ENDPOINT}"
export TF_VAR_ovh_application_key="${OVH_APPLICATION_KEY}"
export TF_VAR_ovh_application_secret="${OVH_APPLICATION_SECRET}"
export TF_VAR_ovh_consumer_key="${OVH_CONSUMER_KEY}"
export TF_VAR_ovh_project_id="${OVH_PROJECT_ID}"
export TF_VAR_region="${OVH_REGION:-GRA}"
export TF_VAR_environment="${OVH_ENVIRONMENT:-dev}"
export TF_VAR_docker_registry_fallback="${DOCKER_REGISTRY}"
export TF_VAR_s3_bucket_datasets="${S3_BUCKET}"
export TF_VAR_s3_bucket_outputs="${S3_BUCKET_OUTPUTS}"

# Validate required variables
REQUIRED_VARS=("WANDB_API_KEY" "AWS_ACCESS_KEY_ID" "AWS_SECRET_ACCESS_KEY" "AWS_ENDPOINT_URL" "OVH_APPLICATION_KEY" "OVH_APPLICATION_SECRET" "OVH_CONSUMER_KEY" "OVH_PROJECT_ID")
for var in "${REQUIRED_VARS[@]}"; do
  if [[ -z "${!var:-}" ]]; then
    echo -e "${RED}Error: Required variable $var not set in $ENV_FILE${NC}"
    exit 1
  fi
done

################################################################################
# LOCAL MODE
################################################################################
if [[ "$LOCAL_MODE" == true ]]; then
  echo -e "${YELLOW}🏠 Running in LOCAL mode${NC}"
  
  # Build image if needed
  if [[ "$SKIP_BUILD" == false ]]; then
    echo -e "${BLUE}🐳 Building Docker image...${NC}"
    docker build -t synth-param-estimation:latest .
  fi
  
  # Run locally
  echo -e "${GREEN}▶️  Starting training locally...${NC}"
  docker run --rm \
    --gpus all \
    --shm-size=32G \
    -e PROJECT_ROOT=/workspace \
    -e WANDB_API_KEY="$WANDB_API_KEY" \
    -e AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID" \
    -e AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY" \
    -e AWS_ENDPOINT_URL="$AWS_ENDPOINT_URL" \
    -e AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-gra}" \
    -v "$(pwd)/datasets:/workspace/datasets:ro" \
    -v "$(pwd)/outputs:/workspace/outputs" \
    -v "$(pwd)/logs:/workspace/logs" \
    synth-param-estimation:latest \
    python src/train.py experiment="$EXPERIMENT_CONFIG"
  
  echo -e "${GREEN}✅ Local training complete!${NC}"
  exit 0
fi

################################################################################
# CLOUD MODE (OVH AI Training)
################################################################################

echo -e "${YELLOW}☁️  Running on OVH AI Training${NC}"

# Check if ovhai CLI is installed
if ! command -v ovhai &> /dev/null; then
  echo -e "${RED}Error: ovhai CLI not found${NC}"
  echo "Install it with: ./scripts/setup.sh"
  echo "Or manually: https://cli.bhs.ai.cloud.ovh.net/install.sh | bash"
  exit 1
fi

# Configure ovhai CLI
mkdir -p ~/.ovhcloud
cat > ~/.ovhcloud/config.yaml <<EOF
default:
  endpoint: ${OVH_ENDPOINT}
  application_key: ${OVH_APPLICATION_KEY}
  application_secret: ${OVH_APPLICATION_SECRET}
  consumer_key: ${OVH_CONSUMER_KEY}
  project: ${OVH_PROJECT_ID}
EOF

# Check if Terraform is set up
if [[ ! -d "terraform/.terraform" ]]; then
  echo -e "${YELLOW}⚠️  Terraform not initialized. Run: cd terraform && terraform init${NC}"
  echo -e "${YELLOW}   Or run: ./scripts/setup.sh${NC}"
  exit 1
fi

# Read config from Terraform (optional - falls back to .env)
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

# Build and push Docker image
IMAGE_NAME="synth-param-estimation"
IMAGE_TAG="$(echo "$EXPERIMENT_CONFIG" | tr '/' '-')-$(date +%Y%m%d-%H%M%S)"
FULL_IMAGE="${REGISTRY_URL}/${IMAGE_NAME}:${IMAGE_TAG}"

# Docker login
if [[ -n "${DOCKER_USERNAME:-}" ]] && [[ -n "${DOCKER_PASSWORD:-}" ]]; then
  echo "${DOCKER_PASSWORD}" | docker login --username "${DOCKER_USERNAME}" --password-stdin 2>/dev/null || true
fi

if [[ "$SKIP_BUILD" == false ]]; then
  echo -e "${BLUE}🐳 Building Docker image: ${IMAGE_TAG}${NC}"
  docker build -t "$FULL_IMAGE" .
  
  echo -e "${BLUE}⬆️  Pushing to registry...${NC}"
  docker push "$FULL_IMAGE"
else
  echo -e "${YELLOW}⏭️  Skipping build (--skip-build)${NC}"
fi

# Extract dataset name from experiment config
DATASET_NAME=$(echo "$EXPERIMENT_CONFIG" | sed 's/.*dataset_\([0-9]*k\).*/\1/' | sed 's/.*surge-\([0-9]*k\).*/\1/')
if [[ -z "$DATASET_NAME" ]]; then
  # Default to surge-100k if no dataset specified
  DATASET_NAME="100k"
fi

echo -e "${BLUE}📥 Will download surge-${DATASET_NAME} dataset on remote server${NC}"

# Submit job
echo -e "${GREEN}🎯 Submitting job...${NC}"
JOB_NAME="$(echo "$EXPERIMENT_CONFIG" | tr '/' '-')-$(date +%s)"
FLAVOR="${FLAVOR:-ai1-1-gpu}"

# Construct ovhai command with S3 mount and download script
OVHAI_CMD="ovhai job run \
  --name \"${JOB_NAME}\" \
  --flavor ${FLAVOR} \
  --volume \"${S3_BUCKET_DATASETS}@GRA:/workspace/s3-data:RO:cache\" \
  --volume \"${S3_BUCKET_OUTPUTS}@GRA:/workspace/outputs:RW\" \
  --env WANDB_API_KEY=\"${WANDB_API_KEY}\" \
  --env PROJECT_ROOT=/workspace \
  --env MPLCONFIGDIR=/tmp/matplotlib \
  --env AWS_ACCESS_KEY_ID=\"${AWS_ACCESS_KEY_ID}\" \
  --env AWS_SECRET_ACCESS_KEY=\"${AWS_SECRET_ACCESS_KEY}\" \
  --env AWS_ENDPOINT_URL=\"${AWS_ENDPOINT_URL}\" \
  --env AWS_DEFAULT_REGION=\"${AWS_DEFAULT_REGION:-gra}\" \
  --unsecure-http \
  --output json \
  \"${FULL_IMAGE}\" \
  -- bash -c 'mkdir -p /workspace/datasets && aws s3 sync s3://${S3_BUCKET_DATASETS}/datasets/surge-${DATASET_NAME}/ /workspace/datasets/surge-${DATASET_NAME}/ --endpoint-url=${AWS_ENDPOINT_URL} && python src/train.py experiment=\"${EXPERIMENT_CONFIG}\"'"

# Submit job
set +e
JOB_OUTPUT=$(eval "$OVHAI_CMD" 2>&1)
OVHAI_EXIT_CODE=$?
set -e

if [[ $OVHAI_EXIT_CODE -ne 0 ]]; then
  echo -e "${RED}Failed to submit job${NC}"
  echo "$JOB_OUTPUT"
  exit 1
fi

JOB_ID=$(echo "$JOB_OUTPUT" | jq -r '.id // .uuid // empty' 2>/dev/null)
if [[ -z "$JOB_ID" ]]; then
  echo -e "${RED}Failed to get job ID${NC}"
  echo "$JOB_OUTPUT"
  exit 1
fi

echo ""
echo -e "${GREEN}✅ Job submitted: ${JOB_ID}${NC}"
echo ""
echo -e "Monitor: ./scripts/status.sh ${JOB_ID}"
echo -e "Logs:    ./scripts/logs.sh ${JOB_ID}"
echo -e "Stop:    ovhai job stop ${JOB_ID}"
echo -e "W&B:     https://wandb.ai/${WANDB_ENTITY}/${WANDB_PROJECT}"

# Stream logs if requested
if [[ "$STREAM_LOGS" == true ]]; then
  echo ""
  echo "Waiting for job to start..."
  sleep 10
  ovhai job logs "$JOB_ID" --follow
fi
