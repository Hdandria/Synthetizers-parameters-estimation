#!/bin/bash
# Test S3 access with OVH Object Storage

set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}🔍 Testing OVH S3 Access${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Load environment variables
if [ -f .env ]; then
    set -a
    source <(grep -v '^#' .env | sed -e '/^$/d' -e 's/#.*$//' -e 's/[[:space:]]*$//')
    set +a
    echo -e "${GREEN}✓${NC} Loaded .env file"
else
    echo -e "${RED}✗${NC} .env file not found!"
    exit 1
fi

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo -e "${RED}✗${NC} AWS CLI not installed. Installing..."
    pip install awscli
fi

echo ""
echo -e "${YELLOW}Testing S3 Configuration:${NC}"
echo "  Endpoint: $AWS_ENDPOINT_URL"
echo "  Region: $AWS_DEFAULT_REGION"
echo "  Access Key: ${AWS_ACCESS_KEY_ID:0:8}..."
echo ""

# Test 1: List buckets
echo -e "${YELLOW}Test 1: Listing all buckets${NC}"
if aws s3 ls --endpoint-url=$AWS_ENDPOINT_URL 2>&1 | tee /tmp/s3_test.log; then
    echo -e "${GREEN}✓${NC} Successfully listed buckets"
else
    echo -e "${RED}✗${NC} Failed to list buckets"
    cat /tmp/s3_test.log
    exit 1
fi

echo ""
echo -e "${YELLOW}Test 2: Checking datasets bucket ($S3_BUCKET)${NC}"
if aws s3 ls s3://$S3_BUCKET/ --endpoint-url=$AWS_ENDPOINT_URL --recursive --human-readable --summarize 2>&1 | head -20; then
    echo -e "${GREEN}✓${NC} Successfully accessed datasets bucket"
else
    echo -e "${RED}✗${NC} Failed to access datasets bucket"
    exit 1
fi

echo ""
echo -e "${YELLOW}Test 3: Checking outputs bucket ($S3_BUCKET_OUTPUTS)${NC}"
if aws s3 ls s3://$S3_BUCKET_OUTPUTS/ --endpoint-url=$AWS_ENDPOINT_URL 2>&1; then
    echo -e "${GREEN}✓${NC} Successfully accessed outputs bucket"
else
    echo -e "${RED}✗${NC} Failed to access outputs bucket"
    exit 1
fi

echo ""
echo -e "${YELLOW}Test 4: Testing write permissions (outputs bucket)${NC}"
TEST_FILE="/tmp/test_s3_access_$(date +%s).txt"
echo "Test file created at $(date)" > $TEST_FILE

if aws s3 cp $TEST_FILE s3://$S3_BUCKET_OUTPUTS/test/ --endpoint-url=$AWS_ENDPOINT_URL 2>&1; then
    echo -e "${GREEN}✓${NC} Successfully wrote to outputs bucket"
    
    # Clean up test file
    aws s3 rm s3://$S3_BUCKET_OUTPUTS/test/$(basename $TEST_FILE) --endpoint-url=$AWS_ENDPOINT_URL 2>&1
    echo -e "${GREEN}✓${NC} Cleaned up test file"
else
    echo -e "${RED}✗${NC} Failed to write to outputs bucket"
    exit 1
fi

rm -f $TEST_FILE

echo ""
echo -e "${YELLOW}Test 5: Checking specific dataset path${NC}"
DATASET_CHECK="surge-100k"
if aws s3 ls s3://$S3_BUCKET/$DATASET_CHECK/ --endpoint-url=$AWS_ENDPOINT_URL 2>&1; then
    echo -e "${GREEN}✓${NC} Found $DATASET_CHECK in datasets bucket"
else
    echo -e "${YELLOW}⚠${NC} Dataset $DATASET_CHECK not found (may need to be uploaded)"
fi

echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✓ All S3 access tests passed!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
