output "data_lake_bucket" {
  description = "The name of the S3 bucket used as data lake"
  value       = aws_s3_bucket.data_lake.id
}

output "glue_database_name" {
  description = "The name of the Glue catalog database"
  value       = aws_glue_catalog_database.msd_database.name
}

output "sagemaker_notebook_url" {
  description = "The URL of the SageMaker notebook instance"
  value       = aws_sagemaker_notebook_instance.msd_notebook.url
}

output "ecr_repository_url" {
  description = "The URL of the ECR repository"
  value       = aws_ecr_repository.model_repo.repository_url
}

output "cloudwatch_log_group" {
  description = "The name of the CloudWatch log group"
  value       = aws_cloudwatch_log_group.msd_logs.name
}

output "lambda_function_name" {
  description = "The name of the Lambda function"
  value       = aws_lambda_function.automation.function_name
}

output "sns_topic_arn" {
  description = "The ARN of the SNS topic for notifications"
  value       = aws_sns_topic.notifications.arn
}