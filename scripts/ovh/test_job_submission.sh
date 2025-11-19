#!/bin/bash
# Test OVH job submission with the fixed dataset mounting

set -euo pipefail

# Colors
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly CYAN='\033[0;36m'
readonly RED='\033[0;31m'
readonly BOLD='\033[1m'
readonly RESET='\033[0m'

echo -e "${CYAN}${BOLD}>>> Testing OVH Job Submission${RESET}"
echo ""

# Check for experiment argument
if [ $# -lt 1 ]; then
    echo -e "${YELLOW}Usage: $0 <experiment> [--skip-build]${RESET}"
    echo ""
    echo "Available experiments:"
    ls -1 configs/experiment/flow_multi/*.yaml | sed 's/.*\//  - flow_multi\//' | sed 's/\.yaml$//'
    echo ""
    echo "Example:"
    echo -e "  ${BLUE}$0 flow_multi/dataset_100k${RESET}"
    exit 1
fi

EXPERIMENT="$1"
SKIP_BUILD=""
[ "${2:-}" = "--skip-build" ] && SKIP_BUILD="--skip-build"

echo -e "${GREEN}[1/4] Verifying S3 access...${RESET}"
if ! ./scripts/ovh/test_s3_access.sh > /tmp/s3_test.log 2>&1; then
    echo -e "${RED}✗ S3 access test failed${RESET}"
    cat /tmp/s3_test.log
    exit 1
fi
echo -e "${GREEN}    ✓ S3 access verified${RESET}"

echo ""
echo -e "${GREEN}[2/4] Checking OVH CLI...${RESET}"
if ! command -v ovhai &>/dev/null; then
    echo -e "${RED}✗ ovhai CLI not found${RESET}"
    echo ""
    echo "Install with:"
    echo "  curl -o /tmp/ovhai https://cli.gra.ai.cloud.ovh.net/install.sh && bash /tmp/ovhai"
    exit 1
fi

if ! ovhai me &>/dev/null; then
    echo -e "${RED}✗ Not logged in to OVH${RESET}"
    echo ""
    echo "Login with:"
    echo "  ovhai login"
    exit 1
fi
echo -e "${GREEN}    ✓ OVH CLI ready${RESET}"

echo ""
echo -e "${GREEN}[3/4] Checking experiment config...${RESET}"
CONFIG_FILE="configs/experiment/${EXPERIMENT}.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}✗ Config not found: $CONFIG_FILE${RESET}"
    exit 1
fi

DATASET_ROOT=$(grep -oP 'dataset_root:\s*\K.*' "$CONFIG_FILE" | tr -d "'\" " | head -1)
if [ -z "$DATASET_ROOT" ]; then
    echo -e "${RED}✗ No dataset_root found in config${RESET}"
    exit 1
fi

DATASET_NAME=$(basename "$DATASET_ROOT")
echo -e "${GREEN}    ✓ Config: ${CONFIG_FILE}${RESET}"
echo -e "${GREEN}    ✓ Dataset: ${DATASET_NAME}${RESET}"

echo ""
echo -e "${GREEN}[4/4] Submitting job...${RESET}"
echo -e "${BLUE}    Experiment: ${EXPERIMENT}${RESET}"
echo -e "${BLUE}    Dataset: ${DATASET_NAME}${RESET}"
echo ""

# Run launch.sh
./launch.sh "$EXPERIMENT" $SKIP_BUILD

echo ""
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo -e "${GREEN}${BOLD}✓ Job submitted successfully${RESET}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
