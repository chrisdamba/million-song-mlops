# Mage AI Setup and Usage

This document outlines the process of setting up and using Mage AI for workflow orchestration in our Million Song Dataset MLOps project.

## Table of Contents
1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Creating a Pipeline](#creating-a-pipeline)
6. [Running a Pipeline](#running-a-pipeline)
7. [Scheduling Pipelines](#scheduling-pipelines)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)

## Introduction

Mage AI is an open-source data pipeline and MLOps tool that helps to streamline the entire machine learning lifecycle. We use it for:
- Data preprocessing
- Feature engineering
- Model training
- Model deployment orchestration

## Prerequisites

- Python 3.8+
- Docker (for containerized deployment)
- Access to the project's S3 bucket and other AWS resources
- AWS credentials configured

## Installation

1. Install Mage AI:

```bash
pip install mage-ai
```

2. Create a new Mage AI project:

```bash
mage init million_song_project
cd million_song_project
```

## Configuration

1. Update the `io_config.yaml` file in your Mage AI project directory to include your AWS credentials and S3 bucket information:

```yaml
aws:
  access_key_id: ${AWS_ACCESS_KEY_ID}
  secret_access_key: ${AWS_SECRET_ACCESS_KEY}
  region_name: ${AWS_DEFAULT_REGION}

s3:
  bucket: ${S3_BUCKET_NAME}
```

2. Create a `.env` file in your Mage AI project directory with your actual AWS credentials and S3 bucket name:

```
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=your_aws_region
S3_BUCKET_NAME=your_s3_bucket_name
```

## Creating the Pipeline

1. In your Mage AI project directory, create a new pipeline:

```bash
mage add pipeline million_song_pipeline
```

2. Replace the content of the generated pipeline file with the code from `src/mage_ai_pipelines/million_song_pipeline.py`.

## Running the Pipeline

To run the pipeline:

```bash
mage run million_song_pipeline
```

## Scheduling the Pipeline

1. Create a schedule for your pipeline in the `schedules/million_song_schedule.py` file:

```python
from mage_ai.orchestration.triggers.time import TimeTrigger

trigger = TimeTrigger(
    schedule='0 0 * * *',  # Run daily at midnight
    pipeline_uuid='million_song_pipeline'
)
```

2. Enable the schedule:

```bash
mage schedule enable million_song_schedule
```

This setup integrates Mage AI into your existing project structure:

1. It uses your existing data loading, model training, and prediction functions.
2. It creates a Mage AI pipeline that orchestrates the entire process from data loading to saving recommendations.
3. It allows for easy scheduling of the pipeline.

## Best Practices

1. **Modular Design**: Break down your pipeline into reusable components.
2. **Error Handling**: Implement proper error handling and logging in each step.
3. **Data Validation**: Add data validation checks between pipeline steps.
4. **Version Control**: Keep your Mage AI project under version control.
5. **Environment Management**: Use environment variables for sensitive information.
6. **Testing**: Write unit tests for individual pipeline components.

## Troubleshooting

1. **Pipeline Failures**:
   - Check the Mage AI logs for detailed error messages.
   - Ensure all required dependencies are installed.
   - Verify AWS credentials and permissions.

2. **Data Loading Issues**:
   - Confirm S3 bucket and file paths are correct.
   - Check network connectivity to AWS services.

3. **Scheduling Problems**:
   - Verify the cron expression in the schedule file.
   - Ensure the Mage AI scheduler service is running.

4. **Resource Constraints**:
   - Monitor resource usage and adjust as necessary.
   - Consider using Mage AI's distributed execution capabilities for large-scale processing.

For more information, consult the [Mage AI documentation](https://docs.mage.ai/) or reach out to your DevOps team.

