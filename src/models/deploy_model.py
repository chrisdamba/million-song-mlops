import os
import json
import boto3
import mlflow
import sagemaker
from sagemaker.mlflow import MLflowModel


def load_config():
    with open('config.json', 'r') as f:
        return json.load(f)


def deploy_model_to_sagemaker(run_id, config):
    # Set up MLflow tracking URI
    mlflow.set_tracking_uri(config['mlflow_tracking_uri'])

    # Load the model from MLflow
    model_uri = f"runs:/{run_id}/model"

    # Set up SageMaker session
    sagemaker_session = sagemaker.Session()

    # Create SageMaker-compatible model
    mlflow_model = MLflowModel(
        model_uri=model_uri,
        role=config['sagemaker_role_arn'],
        image_uri=config.get('sagemaker_container_image_uri')  # Optional: Use a custom container if needed
    )

    # Deploy the model to SageMaker
    predictor = mlflow_model.deploy(
        initial_instance_count=1,
        instance_type="ml.m5.large",
        endpoint_name=config['sagemaker_endpoint_name']
    )

    print(f"Model deployed successfully to endpoint: {config['sagemaker_endpoint_name']}")
    return predictor


def test_endpoint(predictor, test_data):
    # Assuming test_data is a list of dictionaries, each representing a song
    response = predictor.predict(test_data)
    print("Test prediction results:", response)


def main():
    config = load_config()

    # Assuming the latest run ID is stored somewhere, or you can pass it as an argument
    latest_run_id = config.get('latest_mlflow_run_id')

    if not latest_run_id:
        print("No MLflow run ID found. Please provide a valid run ID.")
        return

    predictor = deploy_model_to_sagemaker(latest_run_id, config)

    # Optional: Test the deployed endpoint
    test_data = [
        {"duration": 230, "tempo": 120, "loudness": -5, "year": 2010}
    ]
    test_endpoint(predictor, test_data)


if __name__ == "__main__":
    main()
