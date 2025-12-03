#!/bin/bash
set -e

# Configuration
NUM_SHARDS=13
SAMPLES_PER_SHARD=10000
OUTPUT_DIR="data/datasets/vital_20k"
PRESET_DIR="data/presets/vital"
PLUGIN_PATH="plugins/Vital.vst3"
WORKERS=30
VARIANCE=0.1
PARAM_SPEC="vital_simple"

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

echo "Starting dataset generation..."
echo "Total shards: $NUM_SHARDS"
echo "Samples per shard: $SAMPLES_PER_SHARD"
echo "Output directory: $OUTPUT_DIR"

for i in $(seq 0 $(($NUM_SHARDS - 1))); do
    SHARD_FILE="$OUTPUT_DIR/shard_$i.h5"
    
    if [ -f "$SHARD_FILE" ]; then
        echo "Shard $i already exists at $SHARD_FILE. Skipping..."
        continue
    fi

    echo "--------------------------------------------------"
    echo "Generating shard $i ($SHARD_FILE)..."
    echo "--------------------------------------------------"
    
    uv run src/data/vst/generate_preset_dataset.py \
        "$SHARD_FILE" \
        $SAMPLES_PER_SHARD \
        --preset_dir "$PRESET_DIR" \
        --num_workers $WORKERS \
        --plugin_path "$PLUGIN_PATH" \
        --perturbation_variance $VARIANCE \
        --param_spec "$PARAM_SPEC"
        
    echo "Shard $i completed."
done

echo "All shards processed."
