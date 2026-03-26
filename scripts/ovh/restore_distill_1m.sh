#!/bin/bash
set -euo pipefail

DS_ALIAS="s3-${OVH_REGION:-gra}"

aws s3 sync ./outputs/distill_1M s3://outputs-2/evaluations/10-42-46 \
    --endpoint-url "${AWS_ENDPOINT_URL}" \
    --region "${AWS_DEFAULT_REGION:-gra}" \
    --delete \
    --exact-timestamps \
    --dryrun
