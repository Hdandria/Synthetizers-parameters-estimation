#!/bin/bash
set -e

# Locate .env file
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="$PROJECT_ROOT/.env"

# Load environment variables for AWS CLI commands
if [ -f "$ENV_FILE" ]; then
    echo "Loading environment variables from $ENV_FILE"
    set -a
    source <(grep -v '^#' "$ENV_FILE" | grep -v '^$' | sed 's/#.*$//')
    set +a
else
    echo "Warning: .env file not found at $ENV_FILE"
    exit 1
fi

# Configuration
FIRST_SHARD=0
LAST_SHARD=9
SAMPLES_PER_SHARD=10000
OUTPUT_DIR="datasets/vital_uniform_1M"
BASE_PRESET="presets/vital-base.vstpreset"
PLUGIN_PATH="plugins/Vital.vst3"
WORKERS=20
PARAM_SPEC="vital"

# S3 Configuration (from .env)
S3_ENDPOINT="${AWS_ENDPOINT_URL}"
S3_REGION="${AWS_DEFAULT_REGION}"
S3_PREFIX="datasets/vital_uniform_1M"  # Prefix for organizing files in the bucket

# Function to upload file to S3 and delete locally
upload_to_s3() {
    local file_path="$1"
    local file_name=$(basename "$file_path")
    local s3_path="s3://${S3_BUCKET}/${S3_PREFIX}/${file_name}"
    
    echo "Uploading $file_name to S3..."
    if uv run aws s3 cp "$file_path" "$s3_path" \
        --endpoint-url "$S3_ENDPOINT" \
        --region "$S3_REGION"; then
        echo "✓ Upload successful: $file_name"
        echo "Deleting local file to free up space..."
        rm "$file_path"
        echo "✓ Local file deleted: $file_name"
        return 0
    else
        echo "✗ Upload failed for $file_name. Keeping local file."
        return 1
    fi
}

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

echo "Starting uniform dataset generation..."
echo "Shard range: $FIRST_SHARD to $LAST_SHARD"
echo "Samples per shard: $SAMPLES_PER_SHARD"
echo "Output directory: $OUTPUT_DIR"

for i in $(seq $FIRST_SHARD $LAST_SHARD); do
    SHARD_FILE="$OUTPUT_DIR/shard_$i.h5"
    
    # Check if file already exists locally
    if [ -f "$SHARD_FILE" ]; then
        echo "Shard $i already exists at $SHARD_FILE. Skipping generation..."
        echo "Attempting to upload existing file..."
        upload_to_s3 "$SHARD_FILE"
        continue
    fi
    
    # Check if file already exists on S3
    S3_PATH="s3://${S3_BUCKET}/${S3_PREFIX}/shard_$i.h5"
    if uv run aws s3 ls "$S3_PATH" --endpoint-url "$S3_ENDPOINT" --region "$S3_REGION" > /dev/null 2>&1; then
        echo "Shard $i already exists on S3. Skipping..."
        continue
    fi

    echo "--------------------------------------------------"
    echo "Generating uniform shard $i ($SHARD_FILE)..."
    echo "--------------------------------------------------"
    
    uv run --env-file "$ENV_FILE" src/data/vst/generate_vst_dataset.py \
        "$SHARD_FILE" \
        $SAMPLES_PER_SHARD \
        --preset_path "$BASE_PRESET" \
        --num_workers $WORKERS \
        --plugin_path "$PLUGIN_PATH" \
        --param_spec "$PARAM_SPEC"
        
    echo "Shard $i completed."
    
    # Upload to S3 and delete local file if successful
    upload_to_s3 "$SHARD_FILE"
done

echo "All shards processed and uploaded to S3."
