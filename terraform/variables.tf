variable "aws_region" {
  description = "The AWS region to deploy to"
  default     = "us-west-2"
}

variable "project_name" {
  description = "The name of the project"
  default     = "million-song-mlops"
}

variable "environment" {
  description = "The deployment environment"
  default     = "dev"
}

variable "vpc_cidr" {
  description = "CIDR block for the VPC"
  default     = "10.0.0.0/16"
}

variable "sagemaker_instance_type" {
  description = "The instance type for SageMaker notebook"
  default     = "ml.t3.medium"
}