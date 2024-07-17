# Model Monitoring

This document outlines the process of setting up and maintaining monitoring for the Million Song Dataset MLOps project.

## Table of Contents
1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Setting Up CloudWatch](#setting-up-cloudwatch)
4. [Monitoring Model Performance](#monitoring-model-performance)
5. [Data Drift Detection](#data-drift-detection)
6. [Alerting](#alerting)
7. [Dashboard Creation](#dashboard-creation)
8. [Automated Retraining](#automated-retraining)
9. [Best Practices](#best-practices)
10. [Troubleshooting](#troubleshooting)

## Introduction

Effective monitoring is crucial for maintaining the performance and reliability of our machine learning model in production. This guide covers setting up comprehensive monitoring using AWS CloudWatch, custom metrics, and automated alerts.

## Prerequisites

- AWS CLI configured with appropriate permissions
- Python 3.8+
- Boto3 library installed
- Deployed model on AWS SageMaker

## Setting Up CloudWatch

1. Enable detailed CloudWatch metrics for your SageMaker endpoint:

```python
import boto3

client = boto3.client('sagemaker')

client.update_endpoint(
    EndpointName='million-song-endpoint',
    EndpointConfigName='million-song-endpoint-config',
    DeploymentConfig={
        'BlueGreenUpdatePolicy': {
            'TrafficRoutingConfiguration': {
                'Type': 'ALL_AT_ONCE'
            }
        },
        'AutoRollbackConfiguration': {
            'Alarms': [
                {'AlarmName': 'million-song-model-error-alarm'}
            ]
        }
    }
)
```

2. Create a CloudWatch Log Group for custom metrics:

```python
logs_client = boto3.client('logs')

logs_client.create_log_group(
    logGroupName='/aws/sagemaker/million-song-endpoint'
)
```

## Monitoring Model Performance

1. Set up a Lambda function to calculate and log model performance metrics:

```python
import boto3
import json
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def lambda_handler(event, context):
    # Retrieve predictions and actual values
    predictions = json.loads(event['predictions'])
    actuals = json.loads(event['actuals'])
    
    # Calculate metrics
    mse = mean_squared_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)
    
    # Log metrics to CloudWatch
    cloudwatch = boto3.client('cloudwatch')
    cloudwatch.put_metric_data(
        Namespace='MillionSongModel',
        MetricData=[
            {
                'MetricName': 'MSE',
                'Value': mse,
                'Unit': 'None'
            },
            {
                'MetricName': 'R2',
                'Value': r2,
                'Unit': 'None'
            }
        ]
    )
    
    return {
        'statusCode': 200,
        'body': json.dumps('Metrics logged successfully')
    }
```

2. Schedule this Lambda function to run periodically using CloudWatch Events.

## Data Drift Detection

1. Implement a data drift detection function:

```python
import scipy.stats as stats

def detect_drift(reference_data, current_data, threshold=0.05):
    drift_detected = False
    for column in reference_data.columns:
        ref_col = reference_data[column]
        cur_col = current_data[column]
        
        # Perform Kolmogorov-Smirnov test
        ks_statistic, p_value = stats.ks_2samp(ref_col, cur_col)
        
        if p_value < threshold:
            drift_detected = True
            print(f"Drift detected in column {column}")
    
    return drift_detected
```

2. Set up a Lambda function to periodically check for data drift and log results to CloudWatch.

## Alerting

1. Create CloudWatch Alarms for critical metrics:

```python
cloudwatch = boto3.client('cloudwatch')

# Alarm for high error rate
cloudwatch.put_metric_alarm(
    AlarmName='HighErrorRate',
    ComparisonOperator='GreaterThanThreshold',
    EvaluationPeriods=1,
    MetricName='ModelErrorRate',
    Namespace='AWS/SageMaker',
    Period=300,
    Statistic='Average',
    Threshold=0.1,
    ActionsEnabled=True,
    AlarmActions=[
        'arn:aws:sns:region:account-id:million-song-alerts'
    ]
)

# Alarm for data drift
cloudwatch.put_metric_alarm(
    AlarmName='DataDriftDetected',
    ComparisonOperator='GreaterThanThreshold',
    EvaluationPeriods=1,
    MetricName='DataDrift',
    Namespace='MillionSongModel',
    Period=3600,
    Statistic='Maximum',
    Threshold=0,
    ActionsEnabled=True,
    AlarmActions=[
        'arn:aws:sns:region:account-id:million-song-alerts'
    ]
)
```

2. Set up an SNS topic to receive these alerts and configure email notifications.

## Dashboard Creation

1. Create a CloudWatch Dashboard to visualize key metrics:

```python
dashboard_body = {
    "widgets": [
        {
            "type": "metric",
            "x": 0,
            "y": 0,
            "width": 12,
            "height": 6,
            "properties": {
                "metrics": [
                    [ "MillionSongModel", "MSE" ],
                    [ ".", "R2" ]
                ],
                "view": "timeSeries",
                "stacked": false,
                "region": "us-west-2",
                "title": "Model Performance Metrics"
            }
        },
        {
            "type": "metric",
            "x": 0,
            "y": 6,
            "width": 12,
            "height": 6,
            "properties": {
                "metrics": [
                    [ "AWS/SageMaker", "ModelLatency", "EndpointName", "million-song-endpoint" ],
                    [ ".", "Invocations", ".", "." ]
                ],
                "view": "timeSeries",
                "stacked": false,
                "region": "us-west-2",
                "title": "Endpoint Metrics"
            }
        }
    ]
}

cloudwatch.put_dashboard(
    DashboardName='MillionSongModelDashboard',
    DashboardBody=json.dumps(dashboard_body)
)
```

## Automated Retraining

1. Set up a Step Functions workflow to automate model retraining when performance degrades:

```python
import stepfunctions
from stepfunctions.steps import LambdaStep, ChoiceRule, Choice
from stepfunctions.workflow import Workflow

# Define Lambda functions for each step
check_performance = LambdaStep(
    state_id="Check Model Performance",
    lambda_function=lambda_function_arn
)

retrain_model = LambdaStep(
    state_id="Retrain Model",
    lambda_function=retrain_lambda_arn
)

deploy_model = LambdaStep(
    state_id="Deploy Model",
    lambda_function=deploy_lambda_arn
)

# Define the workflow
definition = Choice(state_id="Evaluate Performance")
definition.add_choice(
    ChoiceRule.is_present("needsRetraining"),
    retrain_model
)
definition.default_choice(deploy_model)

workflow = Workflow(
    name="ModelRetrainingWorkflow",
    definition=check_performance.next(definition),
    role=role_arn
)

workflow.create()
```

2. Schedule this workflow to run periodically using CloudWatch Events.

## Best Practices

1. **Holistic Monitoring**: Monitor not just model performance, but also system health, data quality, and business impact.
2. **Automated Responses**: Set up automated responses to common issues, such as model retraining or rollbacks.
3. **Regular Reviews**: Conduct regular reviews of monitoring data to identify long-term trends and potential improvements.
4. **Documentation**: Keep monitoring setup and alert responses well-documented and up-to-date.
5. **Testing**: Regularly test your monitoring and alerting system to ensure it's functioning correctly.

## Troubleshooting

1. **Missing Metrics**:
   - Check if the IAM roles have necessary permissions to publish metrics.
   - Verify that the metric namespace and names are correctly specified.

2. **False Alarms**:
   - Review and adjust alarm thresholds based on normal operating conditions.
   - Consider using anomaly detection alarms for metrics with high variability.

3. **Delayed Alerts**:
   - Check the configuration of your SNS topic and subscriptions.
   - Verify that CloudWatch Alarms are configured with appropriate evaluation periods.

4. **Data Drift False Positives**:
   - Refine your drift detection algorithm or adjust thresholds.
   - Consider using more sophisticated drift detection methods if simple statistical tests are insufficient.

For further assistance, consult the [AWS CloudWatch documentation](https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/WhatIsCloudWatch.html) or contact your DevOps team.