#!/bin/bash
################################################################################
# List All Running Jobs
################################################################################

set -euo pipefail

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

if ! command -v ovhai &> /dev/null; then
  echo -e "${RED}Error: ovhai CLI not found${NC}"
  echo "Install it with: ./scripts/ovh/setup.sh"
  exit 1
fi

echo -e "${BLUE}📊 Fetching all jobs...${NC}"
echo ""

# Get all jobs
JOBS=$(ovhai job list -o json 2>/dev/null || echo "[]")

if [[ "$JOBS" == "[]" ]]; then
  echo "No jobs found."
  exit 0
fi

# Parse and display
echo "┌────────────────────────────────────────────────────────────────────────────┐"
echo "│ Active Jobs                                                                 │"
echo "├─────────────────────┬──────────────────────┬────────────┬──────────────────┤"
echo "│ Job ID              │ Name                 │ Status     │ GPU              │"
echo "├─────────────────────┼──────────────────────┼────────────┼──────────────────┤"

echo "$JOBS" | jq -r '.[] | [
  .id // .uuid,
  .spec.name // .name // "N/A",
  .status.state // .state // "UNKNOWN",
  "\(.spec.resources.gpu // .resources.gpu // 0)x \(.spec.resources.gpuModel // .resources.gpuModel // \"N/A\")"
] | @tsv' | while IFS=$'\t' read -r id name state gpu; do
  # Truncate name if too long
  name_short=$(echo "$name" | cut -c1-20)

  # Color code status
  case $state in
    RUNNING)
      state_colored="${GREEN}${state}${NC}"
      ;;
    QUEUED|PENDING|INITIALIZING)
      state_colored="${YELLOW}${state}${NC}"
      ;;
    DONE)
      state_colored="${BLUE}${state}${NC}"
      ;;
    FAILED|ERROR|INTERRUPTED)
      state_colored="${RED}${state}${NC}"
      ;;
    *)
      state_colored="${state}"
      ;;
  esac

  printf "│ %-19s │ %-20s │ %-10b │ %-16s │\n" \
    "${id:0:19}" "$name_short" "$state_colored" "${gpu:0:16}"
done

echo "└─────────────────────┴──────────────────────┴────────────┴──────────────────┘"
echo ""
echo -e "${YELLOW}Commands:${NC}"
echo -e "  Status:  ${BLUE}./scripts/ovh/status.sh <job-id>${NC}"
echo -e "  Logs:    ${BLUE}./scripts/ovh/logs.sh <job-id> --follow${NC}"
echo -e "  Stop:    ${BLUE}ovhai job stop <job-id>${NC}"
