# Quick Start

## Setup

```bash
./scripts/setup.sh
cp .env.example .env && nano .env  # Add credentials
./launch.sh surge/base --local     # Test
./launch.sh surge/base             # Run
```

## Credentials

Add to `.env`:

```bash
# OVH (get at https://www.ovh.com/auth/api/createToken)
OVH_APPLICATION_KEY=...
OVH_APPLICATION_SECRET=...
OVH_CONSUMER_KEY=...
OVH_PROJECT_ID=...

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

## Troubleshooting

**Docker permission:** `sudo usermod -aG docker $USER && newgrp docker`
**Missing config:** `find configs/experiment -name "*.yaml"`
