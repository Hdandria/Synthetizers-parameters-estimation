#!/bin/bash
################################################################################
# Evaluate Model and Compute Audio Metrics
# This script:
# 1. Runs prediction on test/val set using trained checkpoint
# 2. Renders predictions to audio using VST plugin
# 3. Computes audio metrics
#
# Usage (when called from launch.sh):
#   Auto-detects checkpoint from training output
# Usage (manual):
#   ./scripts/evaluate_with_metrics.sh <checkpoint_path> <experiment_config> <dataset_split> <dataset_root>
################################################################################

set -euo pipefail

# Arguments can be passed or auto-detected
CKPT_PATH="${1:-}"
EXPERIMENT_CONFIG="${2:-}"
DATASET_SPLIT="${3:-test}"
DATASET_ROOT="${4:-}"

# If checkpoint path contains wildcards, resolve it
if [[ "$CKPT_PATH" == *"*"* ]]; then
    echo "🔍 Resolving checkpoint path with wildcards: $CKPT_PATH"
    # First try to find best checkpoint (epoch_*.ckpt), fallback to last.ckpt
    RESOLVED_CKPT=$(find /workspace/outputs/train -name "epoch_*.ckpt" -type f 2>/dev/null | sort | tail -n 1)
    if [[ -z "$RESOLVED_CKPT" ]]; then
        RESOLVED_CKPT=$(find /workspace/outputs/train -name "last.ckpt" -type f 2>/dev/null | sort | tail -n 1)
    fi
    if [[ -n "$RESOLVED_CKPT" ]]; then
        CKPT_PATH="$RESOLVED_CKPT"
        echo "✅ Found checkpoint: $CKPT_PATH"
    else
        echo "❌ Error: No checkpoint found matching pattern"
        exit 1
    fi
fi

if [[ -z "$CKPT_PATH" ]] || [[ -z "$EXPERIMENT_CONFIG" ]]; then
    echo "Usage: $0 <checkpoint_path> <experiment_config> [dataset_split] [dataset_root]"
    echo "Example: $0 /workspace/outputs/train/flow_multi/run-123/checkpoints/last.ckpt flow_multi/dataset_50k test /workspace/datasets-mount/datasets/surge-100k"
    exit 1
fi

# Extract run name from checkpoint path for output directory
RUN_DIR=$(dirname $(dirname "$CKPT_PATH"))
RUN_NAME=$(basename "$RUN_DIR")

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎯 Evaluating model with audio metrics"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Checkpoint: $CKPT_PATH"
# Determine if this is a best checkpoint or last checkpoint
if [[ "$CKPT_PATH" == *"epoch_"* ]]; then
    echo "Type: Best checkpoint (lowest train/param_mse)"
else
    echo "Type: Last checkpoint"
fi
echo "Experiment: $EXPERIMENT_CONFIG"
echo "Dataset split: $DATASET_SPLIT"
echo "Run name: $RUN_NAME"
echo ""

# Set up output directories
EVAL_DIR="/workspace/outputs/evaluations/${RUN_NAME}"
PRED_DIR="${EVAL_DIR}/predictions"
AUDIO_DIR="${EVAL_DIR}/audio"
METRICS_DIR="${EVAL_DIR}/metrics"

mkdir -p "$EVAL_DIR" "$PRED_DIR" "$AUDIO_DIR" "$METRICS_DIR"

echo "📁 Output directories:"
echo "  Predictions: $PRED_DIR"
echo "  Audio: $AUDIO_DIR"
echo "  Metrics: $METRICS_DIR"
echo ""

# Step 1: Generate predictions
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 1/3: Generating predictions on ${DATASET_SPLIT} set..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Build the eval command
EVAL_CMD="python src/eval.py \
    experiment=\"${EXPERIMENT_CONFIG}\" \
    ckpt_path=\"${CKPT_PATH}\" \
    mode=predict \
    callbacks=prediction_writer \
    paths.output_dir=\"${PRED_DIR}\" \
    data.predict_file=\"${DATASET_SPLIT}.h5\""

# Add dataset_root override if provided
if [[ -n "$DATASET_ROOT" ]]; then
    EVAL_CMD="${EVAL_CMD} data.dataset_root=\"${DATASET_ROOT}\""
    echo "Using dataset root: $DATASET_ROOT"
fi

echo "Running: $EVAL_CMD"
eval "$EVAL_CMD"

echo "✅ Predictions generated"
echo ""

# Step 2: Render predictions to audio using VST
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 2/3: Rendering predictions to audio..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Find the predictions subdirectory
PRED_SUBDIR=$(find "$PRED_DIR" -type d -name "predictions" | head -n 1)
if [[ -z "$PRED_SUBDIR" ]]; then
    echo "❌ Error: predictions directory not found in $PRED_DIR"
    exit 1
fi

# Use paths from environment variables (set in .env)
PLUGIN_PATH="${VST_PLUGIN_PATH:-/workspace/plugins/Surge XT.vst3/Contents/x86_64-linux/Surge XT.so}"

# Infer preset from experiment config
PRESET_NAME="surge-base"
PRESET_PATH="${VST_PRESET_PATH:-/workspace/presets/${PRESET_NAME}.vstpreset}"

echo "Using plugin: $PLUGIN_PATH"
echo "Using preset: $PRESET_PATH"

python scripts/predict_vst_audio.py \
    "$PRED_SUBDIR" \
    "$AUDIO_DIR" \
    --plugin_path "$PLUGIN_PATH" \
    --preset_path "$PRESET_PATH" \
    --skip-spectrogram

echo "✅ Audio rendering complete"
echo ""

# Step 3: Compute audio metrics
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 3/3: Computing audio metrics..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python scripts/compute_audio_metrics_no_pesto.py \
    "$AUDIO_DIR" \
    "$METRICS_DIR" \
    --num_workers 8

echo "✅ Metrics computation complete"
echo ""

# Display summary
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 Evaluation Summary"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [[ -f "${METRICS_DIR}/summary_stats.csv" ]]; then
    echo "Results saved to: ${METRICS_DIR}/summary_stats.csv"
    echo ""
    cat "${METRICS_DIR}/summary_stats.csv"
else
    echo "⚠️ Summary stats not found"
fi

echo ""
echo "✅ Evaluation complete! All results saved to: $EVAL_DIR"
