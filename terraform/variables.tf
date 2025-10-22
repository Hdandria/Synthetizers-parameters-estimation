################################################################################
# Terraform Variables
################################################################################

################################################################################
# OVH Provider Credentials
################################################################################

variable "ovh_endpoint" {
  description = "OVH API endpoint (ovh-eu, ovh-ca, ovh-us)"
  type        = string
  default     = "ovh-eu"
}

variable "ovh_application_key" {
  description = "OVH API application key"
  type        = string
  sensitive   = true
}

variable "ovh_application_secret" {
  description = "OVH API application secret"
  type        = string
  sensitive   = true
}

variable "ovh_consumer_key" {
  description = "OVH API consumer key"
  type        = string
  sensitive   = true
}

variable "ovh_project_id" {
  description = "OVH Public Cloud project ID"
  type        = string
}

################################################################################
# Infrastructure Configuration
################################################################################

variable "region" {
  description = "OVH region (GRA, BHS, etc.)"
  type        = string
  default     = "GRA"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "dev"
}

################################################################################
# Container Registry
################################################################################

variable "create_registry" {
  description = "Whether to create OVH container registry (set false to use Docker Hub)"
  type        = bool
  default     = false  # Start simple, enable later
}

variable "registry_name" {
  description = "Container registry name"
  type        = string
  default     = "synth-param-estimation"
}

variable "registry_plan" {
  description = "Registry plan (S, M, L)"
  type        = string
  default     = "M"
}

variable "create_registry_user" {
  description = "Whether to create a registry user for CI/CD"
  type        = bool
  default     = false
}

variable "registry_user_email" {
  description = "Email for registry user"
  type        = string
  default     = ""
}

variable "registry_user_login" {
  description = "Login for registry user"
  type        = string
  default     = "ci-bot"
}

variable "docker_registry_fallback" {
  description = "Docker registry to use if OVH registry not created"
  type        = string
  default     = "benjamindupuis"
}

################################################################################
# S3 Storage
################################################################################

variable "s3_bucket_datasets" {
  description = "S3 bucket name for datasets (create manually in OVH console for now)"
  type        = string
  default     = "uniform-100k"
}

variable "s3_bucket_outputs" {
  description = "S3 bucket name for outputs (create manually in OVH console for now)"
  type        = string
  default     = "outputs"
}

variable "s3_bucket_models" {
  description = "S3 bucket name for saved models (optional)"
  type        = string
  default     = "models"
}

################################################################################
# Tags & Metadata
################################################################################

variable "project_name" {
  description = "Project name for tagging"
  type        = string
  default     = "synth-param-estimation"
}

variable "tags" {
  description = "Tags to apply to resources"
  type        = map(string)
  default = {
    Project     = "synth-param-estimation"
    ManagedBy   = "terraform"
    Environment = "dev"
  }
}
