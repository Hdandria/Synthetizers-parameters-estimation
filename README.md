# Quick Start

## Setup

```bash
./scripts/setup.sh
cp .env.example .env && nano .env  # Add credentials
./launch.sh flow_multi/dataset_20k_40k # Run on server
```

## Credentials

Add to `.env`:

```bash
# S3 (OVH Manager → Object Storage)
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...

# Docker Hub (https://hub.docker.com/settings/security)
DOCKER_USERNAME=...
DOCKER_PASSWORD=dckr_pat_...

# W&B (https://wandb.ai/authorize)
WANDB_API_KEY=...
```

## Daily Use

```bash
./launch.sh <experiment>         # Launch
./scripts/status.sh <job-id>     # Monitor
./scripts/logs.sh <job-id>       # Logs
ovhai job stop <job-id>          # Stop
```
## Extending

Edit `terraform/main.tf` to add:
- OVH container registry (`create_registry = true`)
- IAM policies
- Cost budgets
- Multi-cloud (AWS/GCP providers)

## Troubleshooting

**Docker permission:** `sudo usermod -aG docker $USER && newgrp docker`
**Missing config:** `find configs/experiment -name "*.yaml"`
