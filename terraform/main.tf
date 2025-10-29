################################################################################
# Terraform Configuration for ML Training Infrastructure
################################################################################
# This file manages the STATIC infrastructure for your ML training:
#   - S3 buckets for datasets and outputs
#   - Container registry for Docker images
#   - (Future: IAM policies, networks, etc.)
#
# Note: AI Training jobs themselves are managed via ovhai CLI, not Terraform
################################################################################

terraform {
  required_version = ">= 1.5"

  required_providers {
    ovh = {
      source  = "ovh/ovh"
      version = "~> 0.48"
    }
  }

  # Optional: Store state remotely (recommended for teams)
  # Uncomment and configure after initial setup
  # backend "s3" {
  #   bucket   = "terraform-state-synth"
  #   key      = "ml-training/terraform.tfstate"
  #   region   = "gra"
  #   endpoint = "https://s3.gra.io.cloud.ovh.net"
  #
  #   skip_credentials_validation = true
  #   skip_requesting_account_id  = true
  #   skip_metadata_api_check     = true
  # }
}

################################################################################
# Provider Configuration
################################################################################

provider "ovh" {
  endpoint           = var.ovh_endpoint
  application_key    = var.ovh_application_key
  application_secret = var.ovh_application_secret
  consumer_key       = var.ovh_consumer_key
}

################################################################################
# Container Registry
################################################################################

resource "ovh_cloud_project_containerregistry" "main" {
  count = var.create_registry ? 1 : 0

  service_name = var.ovh_project_id
  plan_id      = var.registry_plan
  region       = var.region
  name         = var.registry_name
}

# Registry user for CI/CD (optional)
resource "ovh_cloud_project_containerregistry_user" "ci_user" {
  count = var.create_registry && var.create_registry_user ? 1 : 0

  service_name = var.ovh_project_id
  registry_id  = ovh_cloud_project_containerregistry.main[0].id
  email        = var.registry_user_email
  login        = var.registry_user_login
}

################################################################################
# S3 Storage (via OVH Object Storage)
################################################################################

# Note: OVH Object Storage (S3-compatible) doesn't have direct Terraform resources
# Buckets are typically created via AWS provider or manually via OVH console
# This is a placeholder for future implementation or manual creation tracking

# If you want to manage S3 buckets via Terraform, you would need to:
# 1. Add AWS provider with OVH S3 endpoint
# 2. Use aws_s3_bucket resources
# See variables.tf for bucket names to create manually for now

################################################################################
# Outputs - Used by launch.sh
################################################################################

output "registry_url" {
  description = "Container registry URL for Docker push"
  value       = var.create_registry ? ovh_cloud_project_containerregistry.main[0].url : var.docker_registry_fallback
}

output "registry_name" {
  description = "Container registry name"
  value       = var.create_registry ? ovh_cloud_project_containerregistry.main[0].name : "none"
}

output "s3_bucket_datasets" {
  description = "S3 bucket name for datasets"
  value       = var.s3_bucket_datasets
}

output "s3_bucket_outputs" {
  description = "S3 bucket name for outputs"
  value       = var.s3_bucket_outputs
}

output "s3_endpoint" {
  description = "S3 endpoint URL"
  value       = "https://s3.${var.region}.io.cloud.ovh.net"
}

output "region" {
  description = "OVH region"
  value       = var.region
}
