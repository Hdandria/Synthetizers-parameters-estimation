#!/bin/bash
################################################################################
# Local Evaluation Script (No Docker)
#
# Usage:
#   ./scripts/evaluate_locally.sh <checkpoint_path> <experiment_config> <dataset_split> <dataset_root>
#
# Example:
#   ./scripts/evaluate_locally.sh \
#     ./logs/dataset_20k_40k-2025-10-24_12-59-16/checkpoints/last.ckpt \
#     flow_multi/dataset_20k_40k \
#     test \
#     ./datasets/surge-20k
################################################################################

set -euo pipefail

# Arguments
CKPT_PATH="${1:-}"
EXPERIMENT_CONFIG="${2:-}"
DATASET_SPLIT="${3:-test}"
DATASET_ROOT="${4:-}"

if [[ -z "$CKPT_PATH" ]] || [[ -z "$EXPERIMENT_CONFIG" ]] || [[ -z "$DATASET_ROOT" ]]; then
    echo "Usage: $0 <checkpoint_path> <experiment_config> <dataset_split> <dataset_root>"
    echo "Example: $0 ./logs/dataset_20k_40k-2025-10-24_12-59-16/checkpoints/last.ckpt flow_multi/dataset_20k_40k test ./datasets/surge-20k"
    exit 1
fi

# Extract run name from checkpoint path for output directory
RUN_DIR=$(dirname $(dirname "$CKPT_PATH"))
RUN_NAME=$(basename "$RUN_DIR")

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎯 Evaluating model with audio metrics"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Checkpoint: $CKPT_PATH"
if [[ "$CKPT_PATH" == *"epoch_"* ]]; then
    echo "Type: Best checkpoint (lowest train/param_mse)"
else
    echo "Type: Last checkpoint"
fi
echo "Experiment: $EXPERIMENT_CONFIG"
echo "Dataset split: $DATASET_SPLIT"
echo "Dataset root: $DATASET_ROOT"
echo "Run name: $RUN_NAME"
echo ""

# Set up output directories
EVAL_DIR="./outputs/evaluations/${RUN_NAME}"
PRED_DIR="${EVAL_DIR}/predictions"
AUDIO_DIR="${EVAL_DIR}/audio"
METRICS_DIR="${EVAL_DIR}/metrics"

mkdir -p "$EVAL_DIR" "$PRED_DIR" "$AUDIO_DIR" "$METRICS_DIR"

echo "📁 Output directories:"
echo "  Predictions: $PRED_DIR"
echo "  Audio: $AUDIO_DIR"
echo "  Metrics: $METRICS_DIR"
echo ""

# Set PROJECT_ROOT environment variable
export PROJECT_ROOT=$(pwd)

# Step 1: Generate predictions
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 1/3: Generating predictions on ${DATASET_SPLIT} set..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

PREDICT_FILE_FULL="${DATASET_ROOT}/${DATASET_SPLIT}.h5"
EVAL_CMD="uv run python src/eval.py \
    experiment=\"${EXPERIMENT_CONFIG}\" \
    ckpt_path=\"${CKPT_PATH}\" \
    mode=predict \
    callbacks=prediction_writer \
    callbacks.rich_progress_bar=null \
    paths=default \
    paths.output_dir=\"${PRED_DIR}\" \
    data.predict_file=\"${PREDICT_FILE_FULL}\" \
    data.dataset_root=\"${DATASET_ROOT}\" \
    model.compile=false \
    trainer.accelerator=gpu \
    trainer.precision=16-mixed \
    +trainer.limit_predict_batches=3 \
    data.num_workers=6"
echo "Running: $EVAL_CMD"
eval "$EVAL_CMD"

echo "✅ Predictions generated"
echo ""

# Step 2: Render predictions to audio using VST
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 2/3: Rendering predictions to audio..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# For robustness, we handle both pred-*.pt directly under PRED_DIR and in subfolders
found_pred_dir=$(find "$PRED_DIR" -type f -name "pred-*.pt" -printf '%h\n' | head -n 1 || true)
if [[ -n "$found_pred_dir" ]]; then
    PRED_SUBDIR="$found_pred_dir"
    echo "Found predictions at: $PRED_SUBDIR"
else
    echo "❌ Error: No pred-*.pt files found under $PRED_DIR"
    echo "Please ensure Step 1 has been run successfully"
    exit 1
fi

# Plugin and preset paths (adjust these to your system)
# Note: Use the shorter path - pedalboard will find the .so file automatically
PLUGIN_PATH="${VST_PLUGIN_PATH:-plugins/Surge XT.vst3}"
PRESET_PATH="${VST_PRESET_PATH:-./presets/surge-base.vstpreset}"

echo "Using plugin: $PLUGIN_PATH"
echo "Using preset: $PRESET_PATH"

# Skip the file existence check since pedalboard handles the path internally
# if [[ ! -f "$PLUGIN_PATH" ]]; then
#     echo "⚠️  Warning: Plugin not found at $PLUGIN_PATH"
#     echo "Please set VST_PLUGIN_PATH environment variable to your Surge XT .so file"
#     echo "Example: export VST_PLUGIN_PATH=/path/to/Surge XT.vst3/Contents/x86_64-linux/Surge XT.so"
#     exit 1
# fi

# Render using the 'surge_simple' param spec by default (matches models with 92 params).
uv run python scripts/predict_vst_audio.py \
    "$PRED_SUBDIR" \
    "$AUDIO_DIR" \
    --plugin_path "$PLUGIN_PATH" \
    --preset_path "$PRESET_PATH" \
    --param_spec surge_simple \
    --skip-spectrogram

echo "✅ Audio rendering complete"
echo ""

# Step 3: Compute audio metrics
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 3/3: Computing audio metrics..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

uv run python scripts/compute_audio_metrics_no_pesto.py \
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