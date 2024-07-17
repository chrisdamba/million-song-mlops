# Model Deployment

This document outlines the process of deploying the trained model for the Million Song Dataset MLOps project using AWS SageMaker.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Packaging the Model](#packaging-the-model)
3. [Deploying to SageMaker](#deploying-to-sagemaker)
4. [Testing the Endpoint](#testing-the-endpoint)
5. [Continuous Deployment](#continuous-deployment)
6. [Monitoring](#monitoring)
7. [Rollback Procedure](#rollback-procedure)
8. [Troubleshooting](#troubleshooting)

## Prerequisites

- AWS CLI configured with appropriate permissions
- Docker installed (for local testing)
- Python 3.8+
- Trained model artifact in MLflow
- `boto3` and `sagemaker` Python packages installed

## Packaging the Model

1. Retrieve the model from MLflow:

```python
import mlflow
import json

# Load configuration
with open('config.json', 'r') as f:
    config = json.load(f)

mlflow.set_tracking_uri(config['mlflow_tracking_uri'])

model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
```

2. Create a `model.py` file with a predict function:

```python
import pandas as pd

def model_fn(model_dir):
    return mlflow.pyfunc.load_model(model_dir)

def predict_fn(model, input_data):
    return model.predict(pd.DataFrame(input_data))
```

3. Package the model and dependencies:

```bash
mlflow models build-docker -m runs:/{run_id}/model -n million-song-model
```

## Deploying to SageMaker

1. Push the Docker image to Amazon ECR:

```python
import boto3

ecr_client = boto3.client('ecr')
repository_name = 'million-song-model'
response = ecr_client.create_repository(repositoryName=repository_name)
repository_uri = response['repository']['repositoryUri']

!docker tag million-song-model {repository_uri}:latest
!docker push {repository_uri}:latest
```

2. Create a SageMaker model:

```python
import sagemaker
from sagemaker.model import Model

sagemaker_model = Model(
    image_uri=repository_uri,
    model_data=f's3://{config["s3_bucket_name"]}/models/{run_id}/model.tar.gz',
    role=sagemaker.get_execution_role(),
    name='million-song-model'
)
```

3. Deploy the model to a SageMaker endpoint:

```python
predictor = sagemaker_model.deploy(
    initial_instance_count=1,
    instance_type='ml.t2.medium',
    endpoint_name='million-song-endpoint'
)
```

## Testing the Endpoint

1. Test the deployed model:

```python
import boto3

runtime = boto3.client('runtime.sagemaker')

# Prepare sample input
sample_input = {...}  # Your input data structure

response = runtime.invoke_endpoint(
    EndpointName='million-song-endpoint',
    ContentType='application/json',
    Body=json.dumps(sample_input)
)

result = json.loads(response['Body'].read().decode())
print(result)
```

## Continuous Deployment

1. Set up a CI/CD pipeline (e.g., using GitHub Actions) that:
   - Trains the model on new data
   - Evaluates model performance
   - If performance improves, packages and deploys the new model

2. Example GitHub Actions workflow:

```yaml
name: Model CI/CD

on:
  schedule:
    - cron: '0 0 * * 0'  # Weekly on Sundays

jobs:
  train_and_deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.8'
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    - name: Train model
      run: python src/models/train_model.py
    - name: Deploy model
      run: python src/models/deploy_model.py
```

## Monitoring

1. Set up CloudWatch alarms for the SageMaker endpoint:
   - Invocation errors
   - Latency
   - Model input/output metrics

2. Create a dashboard in CloudWatch to visualize these metrics.

3. Set up alerts to notify the team of any issues.

## Rollback Procedure

In case of issues with the newly deployed model:

1. Identify the previous stable model version.
2. Update the SageMaker endpoint to use the previous model version:

```python
sagemaker_client = boto3.client('sagemaker')

sagemaker_client.update_endpoint(
    EndpointName='million-song-endpoint',
    EndpointConfigName='previous-stable-config'
)
```

## Troubleshooting

1. **Deployment Failures**:
   - Check CloudWatch logs for the SageMaker endpoint.
   - Verify that the ECR image was pushed successfully.
   - Ensure IAM roles have necessary permissions.

2. **Performance Issues**:
   - Monitor CloudWatch metrics for unusual patterns.
   - Check if the instance type is appropriate for the workload.

3. **Prediction Errors**:
   - Verify input data format matches the expected schema.
   - Check for any data preprocessing steps that might be missing.

4. **Scaling Issues**:
   - Consider using SageMaker automatic scaling for the endpoint.

For further assistance, consult the [AWS SageMaker documentation](https://docs.aws.amazon.com/sagemaker/) or contact your DevOps team.
