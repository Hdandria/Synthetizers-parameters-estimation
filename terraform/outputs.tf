################################################################################
# Terraform Outputs
################################################################################
# These outputs are used by launch.sh to configure job submission
################################################################################

output "infrastructure_summary" {
  description = "Summary of deployed infrastructure"
  value = {
    region             = var.region
    environment        = var.environment
    registry_enabled   = var.create_registry
    s3_bucket_datasets = var.s3_bucket_datasets
    s3_bucket_outputs  = var.s3_bucket_outputs
  }
}

output "quick_reference" {
  description = "Quick reference for common commands"
  value = <<-EOT

  ┌─────────────────────────────────────────────────────────────┐
  │ Infrastructure Deployed Successfully!                        │
  └─────────────────────────────────────────────────────────────┘

  Registry:  ${var.create_registry ? ovh_cloud_project_containerregistry.main[0].url : var.docker_registry_fallback}
  Region:    ${var.region}
  S3 Datasets: ${var.s3_bucket_datasets}
  S3 Outputs:  ${var.s3_bucket_outputs}

  Next steps:
  1. Run an experiment:
     ./launch.sh surge/base

  2. Check job status:
  ./scripts/ovh/status.sh <job-id>

  3. Stream logs:
  ./scripts/ovh/logs.sh <job-id>

  EOT
}
