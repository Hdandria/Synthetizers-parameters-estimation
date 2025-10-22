# Setup

## What It Does

`launch.sh` reads `.env`, configures OVH/Docker/Terraform, builds image, submits training job to OVH AI Training.

## Architecture

```
.env → launch.sh → ovhai CLI → OVH AI Training (GPU)
             ↓
        Terraform (optional, manages registry)
```

## Files

```
launch.sh          # Main script
.env               # All credentials
terraform/         # Optional infrastructure
scripts/           # Helpers
configs/experiment/  # Your experiments
```

## Extending

Edit `terraform/main.tf` to add:
- OVH container registry (`create_registry = true`)
- IAM policies
- Cost budgets
- Multi-cloud (AWS/GCP providers)

See [QUICKSTART.md](QUICKSTART.md) for usage.
