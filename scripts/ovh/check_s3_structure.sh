#!/bin/bash
# Check S3 bucket structure

set -a
source .env
set +a

echo "Checking bucket structure..."
echo ""
echo "=== Direct bucket root ==="
aws s3 ls s3://uniform-datasets/ --endpoint-url="$AWS_ENDPOINT_URL"

echo ""
echo "=== datasets/ prefix ==="
aws s3 ls s3://uniform-datasets/datasets/ --endpoint-url="$AWS_ENDPOINT_URL"

echo ""
echo "=== /datasets/ prefix ==="
aws s3 ls "s3://uniform-datasets//datasets/" --endpoint-url="$AWS_ENDPOINT_URL" 2>&1 || echo "Not found"

echo ""
echo "=== surge-100k files (first 10) ==="
aws s3 ls s3://uniform-datasets/ --endpoint-url="$AWS_ENDPOINT_URL" --recursive | grep surge-100k | head -10
