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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# Configuration
ENV_FILE=".env"
SKIP_BUILD=false
STREAM_LOGS=false
LOCAL_MODE=false # Added for future compatibility, though mainly for cloud now

# Default Args
EXPERIMENT_CONFIG=""
CKPT_PATH=""
DATASET_NAME=""
SPLIT="test"
GPU_COUNT=1
SHARDS=""
LIMIT_N=""
LIMIT_AUDIO=""
PLUGIN="surge"
EVAL_TAG=""

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --experiment) EXPERIMENT_CONFIG="$2"; shift 2 ;;
    --ckpt) CKPT_PATH="$2"; shift 2 ;;
    --dataset) DATASET_NAME="$2"; shift 2 ;;
    --split) SPLIT="$2"; shift 2 ;;
    --env) ENV_FILE="$2"; shift 2 ;;
    --skip-build) SKIP_BUILD=true; shift ;;
    --stream) STREAM_LOGS=true; shift ;;
    --help)
      echo "Usage: ./scripts/eval/evaluate.sh [OPTIONS]"
      echo "Required:"
      echo "  --experiment CONFIG   Experiment config (e.g. flow_multi/vital_100k)"
      echo "  --ckpt PATH           Checkpoint path relative to outputs/ (wildcards allowed, e.g. 'outputs/train/.../last.ckpt')"
      echo "  --dataset NAME        Dataset name relative to datasets/ (e.g. vital_100k)"
      echo ""
      echo "Options:"
      echo "  --split NAME          Dataset split (default: test)"
      echo "  --env FILE            Custom env file (default: .env)"
      echo "  --skip-build          Skip Docker build"
      echo "  --stream              Stream logs"
      echo "  --limit N             Limit number of batches (approx N*batch_size samples)"
      echo "  --limit-audio N       Limit number of synth parameters to decode and render into WAV"
      echo "  --shards LIST         Comma-separated list of shards to mount (e.g. '0,1,2' or '0 1 2'). Overrides full dataset mount."
      exit 0
      ;;
    --limit) LIMIT_N="$2"; shift 2 ;;
    --limit-audio) LIMIT_AUDIO="$2"; shift 2 ;;
    --shards) SHARDS="$2"; shift 2 ;;
    --plugin) PLUGIN="$2"; shift 2 ;;
    --tag) EVAL_TAG="$2"; shift 2 ;;
    *)
      echo -e "${RED}Error: Unknown argument '$1'${RESET}"; exit 1;
      ;;
  esac
done

# Validate required arguments
[[ -z "$EXPERIMENT_CONFIG" ]] && { echo -e "${RED}Error: --experiment required${RESET}"; exit 1; }
[[ -z "$CKPT_PATH" ]] && { echo -e "${RED}Error: --ckpt required${RESET}"; exit 1; }

# Helper function to extract dataset_root from a YAML config
extract_dataset_root() {
  local config_file="$1"
  local ds_rel=""

  # Try to find dataset_root directly in the experiment config
  ds_rel=$(grep -oP "dataset_root:\s*\K.*" "$config_file" | tr -d "'\''\" " | head -1 || true)

  if [ -z "$ds_rel" ]; then
    # If not found, check for an override /data and then look in the corresponding data config
    local data_override=$(grep -oP "override /data:\s*\K.*" "$config_file" | head -1 || true)
    if [ -n "$data_override" ] && [ -f "configs/data/${data_override}.yaml" ]; then
      ds_rel=$(grep -oP "dataset_root:\s*\K.*" "configs/data/${data_override}.yaml" | tr -d "'\''\" " | head -1 || true)
    fi
  fi
  # Clean the path: remove leading ./ or /
  ds_rel=${ds_rel#./}
  ds_rel=${ds_rel#/}
  echo "$ds_rel"
}

DS_REL=$(extract_dataset_root "configs/experiment/${EXPERIMENT_CONFIG}.yaml")
if [[ -z "$DS_REL" ]]; then
    if [[ -n "$DATASET_NAME" ]]; then
        echo -e "${YELLOW}Warning: Could not detect dataset from config, using provided --dataset: $DATASET_NAME${RESET}"
        DS_REL="$DATASET_NAME"
    else
        echo -e "${RED}Error: Could not detect dataset from config and --dataset not provided${RESET}"; exit 1;
    fi
else
    echo -e "${BLUE}[*] Detected dataset: ${DS_REL}${RESET}"
    DATASET_NAME="$DS_REL"
fi

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

echo -e "${CYAN}${BOLD}>>> Launching Cloud Evaluation${RESET}"
echo "Experiment: $EXPERIMENT_CONFIG"
echo "Checkpoint: $CKPT_PATH"
echo "Dataset:    $DATASET_NAME ($SPLIT)"

# OVH Setup
command -v ovhai &>/dev/null || { echo -e "${RED}Error: ovhai CLI not found${RESET}"; exit 1; }
ovhai me &>/dev/null || { echo -e "${RED}Error: Not logged in. Run: ovhai login${RESET}"; exit 1; }

# Configure S3 datastore
DS_ALIAS="s3-${OVH_REGION:-gra}"
REGION=$(echo "${OVH_REGION:-GRA}" | tr '[:upper:]' '[:lower:]')
# Idempotent datastore setup
if ovhai datastore list 2>/dev/null | grep -q "^${DS_ALIAS}"; then
  ovhai datastore update s3 "${DS_ALIAS}" "${AWS_ENDPOINT_URL}" "${REGION}" \
    "${AWS_ACCESS_KEY_ID}" "${AWS_SECRET_ACCESS_KEY}" --store-credentials-locally
else
  ovhai datastore add s3 "${DS_ALIAS}" "${AWS_ENDPOINT_URL}" "${REGION}" \
    "${AWS_ACCESS_KEY_ID}" "${AWS_SECRET_ACCESS_KEY}" --store-credentials-locally
fi

# Get config
if [[ -d terraform/.terraform ]]; then
  cd terraform
  REGISTRY_URL=$(terraform output -raw registry_url 2>/dev/null || echo "${DOCKER_REGISTRY}")
  S3_BUCKET_DATASETS="${S3_BUCKET_DATASETS:-$(terraform output -raw s3_bucket_datasets 2>/dev/null || echo "${S3_BUCKET}")}"
  S3_BUCKET_OUTPUTS="${S3_BUCKET_OUTPUTS:-$(terraform output -raw s3_bucket_outputs 2>/dev/null || echo "${S3_BUCKET_OUTPUTS}")}"
  cd ..
else
  REGISTRY_URL="${DOCKER_REGISTRY}"
  S3_BUCKET_DATASETS="${S3_BUCKET}"
  S3_BUCKET_OUTPUTS="${S3_BUCKET_OUTPUTS}"
fi

# Build & push image
# We use a unique tag for the eval job to ensure we're running the# Build & push image
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

# 1. Output bucket (Split mount for optimization)
# Mount train checkpoints (RO) - assumes checkpoints are in outputs/train
# 2. Plugins (always RO)
# 3. Dataset
# We mount the FULL dataset folder but WITHOUT :cache.
if [[ -n "$SHARDS" ]]; then
    echo -e "${YELLOW}Warning: --shards ignored for downloading full dataset.${RESET}"
fi

ovhai job run \
  --name "${JOB_NAME}" \
  --flavor "${FLAVOR:-ai1-1-gpu}" \
  --gpu "${GPU_COUNT}" \
  --env WANDB_API_KEY="${WANDB_API_KEY}" \
  --env PROJECT_ROOT=/workspace \
  --env MPLCONFIGDIR=/tmp/matplotlib \
  --env HDF5_VDS_PREFIX=/workspace/datasets-mount \
  --env AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID}" \
  --env AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY}" \
  --env AWS_ENDPOINT_URL="${AWS_ENDPOINT_URL}" \
  --env AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-gra}" \
  --env S3_BUCKET_DATASETS="${S3_BUCKET_DATASETS}" \
  --env S3_BUCKET_OUTPUTS="${S3_BUCKET_OUTPUTS}" \
  $([ -n "${DATASET_CHECK_VERBOSE:-}" ] && echo "--env DATASET_CHECK_VERBOSE=${DATASET_CHECK_VERBOSE}") \
  $([ -n "${GPU_IDS:-}" ] && echo "--env CUDA_VISIBLE_DEVICES=${GPU_IDS}") \
  --unsecure-http \
  --output json \
  "${FULL_IMAGE}" \
  -- bash -c 'set -euo pipefail
    
    # Optimize S3 transfer settings
    aws configure set default.s3.max_concurrent_requests 20
    aws configure set default.s3.max_queue_size 10000
    aws configure set default.s3.multipart_threshold 64MB
    aws configure set default.s3.multipart_chunksize 16MB
    
    # 1. Download full dataset bucket (mirroring volume mount behavior)
    MOUNT_BASE=/workspace/datasets-mount
    echo "==> Downloading full dataset bucket from ${S3_BUCKET_DATASETS} to ${MOUNT_BASE}..."
    mkdir -p "$MOUNT_BASE"
    aws s3 sync "s3://${S3_BUCKET_DATASETS}" "$MOUNT_BASE" \
      --endpoint-url "${AWS_ENDPOINT_URL}" \
      --region "${AWS_DEFAULT_REGION:-gra}" \
      --no-progress \
      --only-show-errors

    # Adjust MOUNT_BASE if datasets subdir exists (standard logic)
    [ -d "$MOUNT_BASE/datasets" ] && MOUNT_BASE="$MOUNT_BASE/datasets"
    
    [ -d "$MOUNT_BASE" ] || { echo "ERROR: datasets mount not found"; exit 1; }
    
    CFG=configs/experiment/'"${EXPERIMENT_CONFIG}"'.yaml
    DS_REL_IN_JOB=$(grep -oP "dataset_root:\s*\K.*" "$CFG" | tr -d "'\''\" " | head -1 || true)
    
    if [ -z "$DS_REL_IN_JOB" ]; then
      DATA_OVR=$(grep -oP "override /data:\s*\K.*" "$CFG" | head -1 || true)
      if [ -n "$DATA_OVR" ] && [ -f "configs/data/${DATA_OVR}.yaml" ]; then
        DS_REL_IN_JOB=$(grep -oP "dataset_root:\s*\K.*" "configs/data/${DATA_OVR}.yaml" | tr -d "'\''\" " | head -1 || true)
      fi
    fi
    
    [ -z "$DS_REL_IN_JOB" ] && { echo "ERROR: No dataset_root in config"; exit 1; }
    
    DS_NAME=$(basename "$DS_REL_IN_JOB")
    DATASET_ROOT="${MOUNT_BASE}/${DS_NAME}"
    
    echo "==> Dataset: $DATASET_ROOT"
    ls -lah "$DATASET_ROOT/" || true
    
    CKPT_PATH="'"${CKPT_PATH}"'"
    EXPERIMENT="'"${EXPERIMENT_CONFIG}"'"
    SPLIT="'"${SPLIT}"'"
    LIMIT_N="'"${LIMIT_N}"'"
    LIMIT_AUDIO="'"${LIMIT_AUDIO}"'"
    PLUGIN="'"${PLUGIN}"'"
    EVAL_TAG="'"${EVAL_TAG}"'"
    
    # 1. Resolve Checkpoint
    
    # Checkpoint paths are often given relative to the outputs bucket or full s3 URIs
    CKPT_CLEAN=${CKPT_PATH#s3://} # Remove s3:// prefix
    # If the user passed outputs/..., remove it for bucket relative pathing
    CKPT_CLEAN=${CKPT_CLEAN#outputs/}
    CKPT_CLEAN=${CKPT_CLEAN#/}    # Remove leading slash
    CKPT_CLEAN=${CKPT_CLEAN#./}   # Remove leading ./
    
    # We download all checkpoints from this specific experiment run to avoid missing the specific epoch/last.ckpt
    # The run path is usually `train/experiment_name/date/time/checkpoints/`
    RUN_DIR=$(dirname $(dirname "$CKPT_CLEAN"))
    LOCAL_CKPT_DIR="/workspace/outputs/${RUN_DIR}/checkpoints"
    
    echo "==> Downloading checkpoints from s3://${S3_BUCKET_OUTPUTS}/${RUN_DIR}/checkpoints to ${LOCAL_CKPT_DIR}..."
    mkdir -p "$LOCAL_CKPT_DIR"
    aws s3 sync "s3://${S3_BUCKET_OUTPUTS}/${RUN_DIR}/checkpoints" "$LOCAL_CKPT_DIR" \
      --endpoint-url "${AWS_ENDPOINT_URL}" \
      --region "${AWS_DEFAULT_REGION:-gra}" \
      --no-progress \
      --only-show-errors

    FULL_CKPT_PATH="/workspace/outputs/${CKPT_CLEAN}"
    echo "==> Resolving checkpoint: $FULL_CKPT_PATH"
    
    if [[ ! -f "$FULL_CKPT_PATH" ]]; then
        echo "Could not find exact file at $FULL_CKPT_PATH, attempting search in ${LOCAL_CKPT_DIR}..."
        RESOLVED=$(find "$LOCAL_CKPT_DIR" -name "$(basename "$FULL_CKPT_PATH")" | head -n 1)
        if [[ -n "$RESOLVED" ]]; then
            FULL_CKPT_PATH="$RESOLVED"
            echo "Found: $FULL_CKPT_PATH"
        else
            echo "ERROR: Checkpoint not found!"
            find "$LOCAL_CKPT_DIR" -maxdepth 2 -name "*.ckpt" || true
            exit 1
        fi
    fi

    # 2. Setup Output Directories
    # Structure: outputs/evaluations/<run_name>[_<tag>]/...
    RUN_NAME=$(basename $(dirname $(dirname "$FULL_CKPT_PATH")))
    if [[ -n "$EVAL_TAG" ]]; then
        EVAL_DIR="/workspace/outputs/evaluations/${RUN_NAME}_${EVAL_TAG}"
    else
        EVAL_DIR="/workspace/outputs/evaluations/${RUN_NAME}"
    fi
    PRED_DIR="${EVAL_DIR}/predictions"
    AUDIO_DIR="${EVAL_DIR}/audio"
    METRICS_DIR="${EVAL_DIR}/metrics"
    
    mkdir -p "$EVAL_DIR" "$PRED_DIR" "$AUDIO_DIR" "$METRICS_DIR"
    
    echo "==> Output Directory: $EVAL_DIR"

    # 3. Running Prediction
    echo "==> [1/3] Running Prediction..."
    
    # Check if we should override dataset root logic
    # The evals usually assume local paths, so we must be careful with data.dataset_root
    
    # Limit handling
    LIMIT_ARGS=""
    if [[ -n "${LIMIT_N}" ]]; then
        # Limit the number of batches to predict
        LIMIT_ARGS="+trainer.limit_predict_batches=${LIMIT_N}"
    fi

    python src/eval.py \
        experiment="$EXPERIMENT" \
        ckpt_path="$FULL_CKPT_PATH" \
        mode=predict \
        callbacks=prediction_writer \
        paths.output_dir="$PRED_DIR" \
        data.predict_file="${DATASET_ROOT}/${SPLIT}.h5" \
        data.dataset_root="$DATASET_ROOT" \
        trainer.accelerator=gpu \
        $LIMIT_ARGS
        
    echo "Predictions saved to $PRED_DIR"

    # 4. Render Audio
    echo "==> [2/3] Rendering Audio..."
    
    # Locate the actual directory containing predictions (it might be nested)
    PRED_SUBDIR=$(find "$PRED_DIR" -type f -name "pred-0.pt" -exec dirname {} \; | head -n 1)
    [ -z "$PRED_SUBDIR" ] && PRED_SUBDIR="$PRED_DIR" # Fallback if not nested
    
    # Plugin paths - typically inside the image or downloaded from S3
    PLUGIN_BASE="/workspace/datasets-mount/plugins"
    chmod -R +x "$PLUGIN_BASE" || true
    
    if [[ "$PLUGIN" == "vital" ]]; then
        PLUGIN_PATH="${PLUGIN_BASE}/Vital.vst3"
        PRESET_PATH="/workspace/presets/vital-base.vstpreset" 
    else
        PLUGIN_PATH="${PLUGIN_BASE}/Surge XT.vst3"
        PRESET_PATH="/workspace/presets/surge-base.vstpreset"
    fi
    
    LIMIT_AUDIO_ARGS=""
    if [[ -n "${LIMIT_AUDIO}" ]]; then
        LIMIT_AUDIO_ARGS="--limit-audio ${LIMIT_AUDIO}"
    fi

    python scripts/render/predict_vst_audio.py \
        "$PRED_SUBDIR" \
        "$AUDIO_DIR" \
        --plugin_path "$PLUGIN_PATH" \
        --preset_path "$PRESET_PATH" \
        --param_spec vital_simple_legacy \
        --no-params \
        --skip-spectrogram \
        $LIMIT_AUDIO_ARGS
        
    echo "Audio saved to $AUDIO_DIR"

    # 5. Compute Metrics
    echo "==> [3/3] Computing Metrics..."
    
    python scripts/eval/compute_audio_metrics_no_pesto.py \
        "$AUDIO_DIR" \
        "$METRICS_DIR" \
        --num_workers 8
        
    echo "Metrics saved to $METRICS_DIR/summary_stats.csv"

    # 6. Upload Outputs
    if [[ -n "$EVAL_TAG" ]]; then
        S3_TARGET="s3://${S3_BUCKET_OUTPUTS}/evaluations/${RUN_NAME}_${EVAL_TAG}"
    else
        S3_TARGET="s3://${S3_BUCKET_OUTPUTS}/evaluations/${RUN_NAME}"
    fi

    echo "==> Uploading outputs to $S3_TARGET..."
    aws s3 sync "$EVAL_DIR" "$S3_TARGET" \
      --endpoint-url "${AWS_ENDPOINT_URL}" \
      --region "${AWS_DEFAULT_REGION:-gra}" \
      --no-progress \
      --only-show-errors
    
    echo "==> Evaluation Complete!"
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
