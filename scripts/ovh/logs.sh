#!/bin/bash
################################################################################
# Stream Training Job Logs
################################################################################

set -euo pipefail

# Colors
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

if [[ $# -eq 0 ]]; then
  echo "Usage: ./scripts/ovh/logs.sh <job-id> [--follow]"
  echo ""
  echo "Examples:"
  echo "  ./scripts/ovh/logs.sh abc123              # Show logs"
  echo "  ./scripts/ovh/logs.sh abc123 --follow     # Stream logs"
  echo "  ./scripts/ovh/logs.sh abc123 --tail 100   # Last 100 lines"
  exit 1
fi

JOB_ID=$1
shift

if ! command -v ovhai &> /dev/null; then
  echo -e "${RED}Error: ovhai CLI not found${NC}"
  echo "Install it with: ./scripts/ovh/setup.sh"
  exit 1
fi

echo -e "${BLUE}📜 Fetching logs for job: ${JOB_ID}${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Forward all other arguments to ovhai
ovhai job logs "$JOB_ID" "$@"
