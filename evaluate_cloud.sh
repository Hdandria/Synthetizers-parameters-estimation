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
LOCAL_MODE=false # Added for future compatibility, though mainly for cloud now

# Default Args
EXPERIMENT_CONFIG=""
CKPT_PATH=""
DATASET_NAME=""
SPLIT="test"
GPU_COUNT=1
SHARDS=""
LIMIT_N=""
PLUGIN="surge"

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
      echo "Usage: ./evaluate_cloud.sh [OPTIONS]"
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
      echo "  --shards LIST         Comma-separated list of shards to mount (e.g. '0,1,2' or '0 1 2'). Overrides full dataset mount."
      exit 0
      ;;
    --limit) LIMIT_N="$2"; shift 2 ;;
    --shards) SHARDS="$2"; shift 2 ;;
    --plugin) PLUGIN="$2"; shift 2 ;;
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
  ds_rel=$(grep -oP "dataset_root:\s*\K.*" "$config_file" | tr -d "'\''\" " | head -1)

  if [ -z "$ds_rel" ]; then
    # If not found, check for an override /data and then look in the corresponding data config
    local data_override=$(grep -oP "override /data:\s*\K.*" "$config_file" | head -1)
    if [ -n "$data_override" ] && [ -f "configs/data/${data_override}.yaml" ]; then
      ds_rel=$(grep -oP "dataset_root:\s*\K.*" "configs/data/${data_override}.yaml" | tr -d "'\''\" " | head -1)
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
  S3_BUCKET_DATASETS=$(terraform output -raw s3_bucket_datasets 2>/dev/null || echo "${S3_BUCKET}")
  S3_BUCKET_OUTPUTS=$(terraform output -raw s3_bucket_outputs 2>/dev/null || echo "${S3_BUCKET_OUTPUTS}")
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

# Granular Mounts with Caching:
VOLUMES_ARGS=()
# 1. Output bucket (Split mount for optimization)
# Mount train checkpoints (RO) - assumes checkpoints are in outputs/train
VOLUMES_ARGS+=("--volume" "${S3_BUCKET_OUTPUTS}@${DS_ALIAS}:/workspace/outputs:RW:cache")
# 2. Plugins (always RO)
VOLUMES_ARGS+=("--volume" "${S3_BUCKET_DATASETS}@${DS_ALIAS}/plugins:/workspace/datasets-mount/plugins:RO:cache")

# 3. Dataset
# We mount the FULL dataset folder but WITHOUT :cache.
# using :RO (Read Only, no-cache).
# This is critical because:
# 1. :cache downloads everything (Too slow).
# 2. Granular file mounting causes IsADirectoryError (OVH limitation).
# 3. :RO is network-streamed (FUSE), providing instant startup and pay-for-what-you-read.

if [[ -n "$SHARDS" ]]; then
    echo -e "${YELLOW}Warning: --shards ignored for mounting to prevent IsADirectoryError.${RESET}"
    echo -e "${YELLOW}         Mounting full dataset in NO-CACHE mode (Instant startup).${RESET}"
fi

VOLUMES_ARGS+=("--volume" "${S3_BUCKET_DATASETS}@${DS_ALIAS}/${DS_REL}:/workspace/datasets-mount/${DS_REL}:RO")

ovhai job run \
  --name "${JOB_NAME}" \
  --flavor "${FLAVOR:-ai1-1-gpu}" \
  --gpu "${GPU_COUNT}" \
  "${VOLUMES_ARGS[@]}" \
  --env WANDB_API_KEY="${WANDB_API_KEY}" \
  --env PROJECT_ROOT=/workspace \
  --env MPLCONFIGDIR=/tmp/matplotlib \
  --env HDF5_VDS_PREFIX=/workspace/datasets-mount \
  --env AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID}" \
  --env AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY}" \
  --env AWS_ENDPOINT_URL="${AWS_ENDPOINT_URL}" \
  --env AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-gra}" \
  $([ -n "${DATASET_CHECK_VERBOSE:-}" ] && echo "--env DATASET_CHECK_VERBOSE=${DATASET_CHECK_VERBOSE}") \
  $([ -n "${GPU_IDS:-}" ] && echo "--env CUDA_VISIBLE_DEVICES=${GPU_IDS}") \
  --unsecure-http \
  --output json \
  "${FULL_IMAGE}" \
  -- bash -c 'set -euo pipefail
    MOUNT_BASE=/workspace/datasets-mount
    [ -d "$MOUNT_BASE" ] || { echo "ERROR: datasets mount not found"; exit 1; }
    
    CFG=configs/experiment/'"${EXPERIMENT_CONFIG}"'.yaml
    DS_REL_IN_JOB=$(grep -oP "dataset_root:\s*\K.*" "$CFG" | tr -d "'\''\" " | head -1)
    
    if [ -z "$DS_REL_IN_JOB" ]; then
      DATA_OVR=$(grep -oP "override /data:\s*\K.*" "$CFG" | head -1)
      if [ -n "$DATA_OVR" ] && [ -f "configs/data/${DATA_OVR}.yaml" ]; then
        DS_REL_IN_JOB=$(grep -oP "dataset_root:\s*\K.*" "configs/data/${DATA_OVR}.yaml" | tr -d "'\''\" " | head -1)
      fi
    fi
    
    [ -z "$DS_REL_IN_JOB" ] && { echo "ERROR: No dataset_root in config"; exit 1; }
    
    DATASET_ROOT="${MOUNT_BASE}/${DS_REL_IN_JOB}"
    
    echo "==> Dataset: $DATASET_ROOT"
    ls -lah "$DATASET_ROOT/" || true
    
    CKPT_PATH="'"${CKPT_PATH}"'"
    EXPERIMENT="'"${EXPERIMENT_CONFIG}"'"
    SPLIT="'"${SPLIT}"'"
    LIMIT_N="'"${LIMIT_N}"'"
    PLUGIN="'"${PLUGIN}"'"
    
    # 1. Resolve Checkpoint
    
    # Clean path
    CKPT_CLEAN=${CKPT_PATH#s3://} # Remove s3:// prefix
    CKPT_CLEAN=${CKPT_CLEAN#/}    # Remove leading slash
    CKPT_CLEAN=${CKPT_CLEAN#./}   # Remove leading ./
    
    FULL_CKPT_PATH="/workspace/${CKPT_CLEAN}"
    
    echo "==> Resolving checkpoint: $FULL_CKPT_PATH"
    
    # Handle wildcards if passed (though single quotes in bash -c might prevent expansion, 
    # the find command can handle it if we pass the dir)
    if [[ ! -f "$FULL_CKPT_PATH" ]]; then
        echo "Could not find exact file at $FULL_CKPT_PATH, attempting search..."
        # Try to find it if it looks like a directory or pattern
        RESOLVED=$(find "$(dirname "$FULL_CKPT_PATH")" -name "$(basename "$FULL_CKPT_PATH")" | head -n 1)
        if [[ -n "$RESOLVED" ]]; then
            FULL_CKPT_PATH="$RESOLVED"
            echo "Found: $FULL_CKPT_PATH"
        else
            echo "ERROR: Checkpoint not found!"
            find /workspace/outputs -maxdepth 4 -name "*.ckpt" | head -n 5
            exit 1
        fi
    fi

    # 2. Setup Output Directories
    # Structure: outputs/evaluations/<run_name>/...
    RUN_NAME=$(basename $(dirname $(dirname "$FULL_CKPT_PATH")))
    EVAL_DIR="/workspace/outputs/evaluations/${RUN_NAME}"
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
    PRED_SUBDIR=$(find "$PRED_DIR" -type d -name "predictions" | head -n 1)
    [ -z "$PRED_SUBDIR" ] && PRED_SUBDIR="$PRED_DIR" # Fallback if not nested
    
    # Plugin paths - typically inside the image at /workspace/plugins
    if [[ "$PLUGIN" == "vital" ]]; then
        PLUGIN_PATH="/workspace/plugins/Vital.vst3/Contents/x86_64-linux/Vital.so"
        PRESET_PATH="/workspace/presets/vital-base.vital" 
    else
        PLUGIN_PATH="/workspace/plugins/Surge XT.vst3/Contents/x86_64-linux/Surge XT.so"
        PRESET_PATH="/workspace/presets/surge-base.vstpreset"
    fi
    
    python scripts/render/predict_vst_audio.py \
        "$PRED_SUBDIR" \
        "$AUDIO_DIR" \
        --plugin_path "$PLUGIN_PATH" \
        --preset_path "$PRESET_PATH" \
        --skip-spectrogram
        
    echo "Audio saved to $AUDIO_DIR"

    # 5. Compute Metrics
    echo "==> [3/3] Computing Metrics..."
    
    python scripts/eval/compute_audio_metrics_no_pesto.py \
        "$AUDIO_DIR" \
        "$METRICS_DIR" \
        --num_workers 8
        
    echo "Metrics saved to $METRICS_DIR/summary_stats.csv"
    
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
