#!/bin/bash
source <(grep -v '^#' .env | grep -v '^$' | sed 's/#.*$//')

SOURCE_PREFIX="datasets/vital_100k"
DEST_PREFIX="datasets/vital_1M"
BUCKET="uniform-datasets"

# List all objects and copy them server-side
aws s3api list-objects-v2 \
    --bucket "$BUCKET" \
    --prefix "$SOURCE_PREFIX/" \
    --endpoint-url "$AWS_ENDPOINT_URL" \
    --region "$AWS_DEFAULT_REGION" \
    --query 'Contents[].Key' \
    --output text | tr '\t' '\n' | while read key; do
    
    # Skip empty lines
    if [[ -z "$key" ]]; then
        continue
    fi
    
    # Skip if it's just the directory marker
    if [[ "$key" == */ ]]; then
        continue
    fi
    
    # Calculate destination key
    dest_key="${key/$SOURCE_PREFIX/$DEST_PREFIX}"
    
    echo "Copying: $key -> $dest_key"
    
    # Server-side copy
    aws s3api copy-object \
        --bucket "$BUCKET" \
        --copy-source "$BUCKET/$key" \
        --key "$dest_key" \
        --endpoint-url "$AWS_ENDPOINT_URL" \
        --region "$AWS_DEFAULT_REGION"
done
