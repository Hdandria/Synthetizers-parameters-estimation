#!/bin/bash
################################################################################
# Check Training Job Status
################################################################################

set -euo pipefail

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

if [[ $# -eq 0 ]]; then
  echo "Usage: ./scripts/status.sh <job-id>"
  echo ""
  echo "List all your jobs:"
  echo "  ovhai job list"
  exit 1
fi

JOB_ID=$1

if ! command -v ovhai &> /dev/null; then
  echo -e "${RED}Error: ovhai CLI not found${NC}"
  echo "Install it with: ./scripts/setup.sh"
  exit 1
fi

echo -e "${BLUE}📊 Fetching job status...${NC}"
echo ""

# Get job details
JOB_INFO=$(ovhai job get "$JOB_ID" -o json 2>/dev/null || echo "{}")

if [[ "$JOB_INFO" == "{}" ]]; then
  echo -e "${RED}❌ Job not found: $JOB_ID${NC}"
  exit 1
fi

# Parse job info
NAME=$(echo "$JOB_INFO" | jq -r '.spec.name // .name // "N/A"')
STATE=$(echo "$JOB_INFO" | jq -r '.status.state // .state // "UNKNOWN"')
CREATED=$(echo "$JOB_INFO" | jq -r '.status.createdAt // .createdAt // "N/A"')
UPDATED=$(echo "$JOB_INFO" | jq -r '.status.updatedAt // .updatedAt // "N/A"')
GPU=$(echo "$JOB_INFO" | jq -r '.spec.resources.gpu // .resources.gpu // 0')
GPU_MODEL=$(echo "$JOB_INFO" | jq -r '.spec.resources.gpuModel // .resources.gpuModel // "N/A"')
IMAGE=$(echo "$JOB_INFO" | jq -r '.spec.image // .image // "N/A"')

# Color code status
case $STATE in
  RUNNING)
    STATE_COLOR="${GREEN}${STATE}${NC}"
    ;;
  QUEUED|PENDING|INITIALIZING)
    STATE_COLOR="${YELLOW}${STATE}${NC}"
    ;;
  DONE)
    STATE_COLOR="${GREEN}${STATE}${NC}"
    ;;
  FAILED|ERROR|INTERRUPTED)
    STATE_COLOR="${RED}${STATE}${NC}"
    ;;
  *)
    STATE_COLOR="${BLUE}${STATE}${NC}"
    ;;
esac

# Display
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "  Job ID:      ${BLUE}${JOB_ID}${NC}"
echo -e "  Name:        ${NAME}"
echo -e "  Status:      ${STATE_COLOR}"
echo -e "  GPU:         ${GPU}x ${GPU_MODEL}"
echo -e "  Image:       ${IMAGE}"
echo -e "  Created:     ${CREATED}"
echo -e "  Updated:     ${UPDATED}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Show logs if running or failed
if [[ "$STATE" == "RUNNING" ]]; then
  echo -e "${BLUE}Recent logs:${NC}"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  ovhai job logs "$JOB_ID" --tail 20
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo ""
  echo -e "${YELLOW}Commands:${NC}"
  echo -e "  Stream logs:  ${BLUE}./scripts/logs.sh ${JOB_ID}${NC}"
  echo -e "  Stop job:     ${BLUE}ovhai job stop ${JOB_ID}${NC}"
elif [[ "$STATE" =~ ^(FAILED|ERROR)$ ]]; then
  echo -e "${RED}Job failed! Last logs:${NC}"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  ovhai job logs "$JOB_ID" --tail 50
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
elif [[ "$STATE" == "DONE" ]]; then
  echo -e "${GREEN}✅ Job completed successfully!${NC}"
  echo ""
  echo -e "View logs: ${BLUE}./scripts/logs.sh ${JOB_ID}${NC}"
fi
