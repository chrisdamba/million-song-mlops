import json

import boto3
import mlflow
import mlflow.sklearn

from src.data.load_data import load_data_from_s3

# Load configuration
with open('config.json', 'r') as f:
    config = json.load(f)


def load_model(run_id):
    mlflow.set_tracking_uri(config['mlflow_tracking_uri'])
    return mlflow.sklearn.load_model(f"runs:/{run_id}/model")


def make_predictions(model, X):
    return model.predict(X)


def main():
    # Load the model from MLflow
    run_id = "your_mlflow_run_id"  # Replace with actual run ID
    model = load_model(run_id)

    # Load new data for predictions
    new_data = load_data_from_s3(config['prepared_data_key'])

    # Ensure the new data has the same features as the training data
    required_features = model.feature_names_in_
    new_data = new_data[required_features]

    # Make predictions
    predictions = make_predictions(model, new_data)

    # Add predictions to the dataframe
    new_data['predicted_popularity'] = predictions

    # Save predictions to S3
    output_key = "predictions.parquet"
    s3 = boto3.client('s3')
    buffer = BytesIO()
    new_data.to_parquet(buffer)
    s3.put_object(Bucket=config['s3_bucket_name'], Key=output_key, Body=buffer.getvalue())

    print(f"Predictions saved to S3://{config['s3_bucket_name']}/{output_key}")


if __name__ == "__main__":
    main()
