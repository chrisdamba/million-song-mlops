import json
from io import BytesIO

import mlflow
import mlflow.sklearn
import pandas as pd
import boto3

from src.data.load_data import load_data_from_s3, prepare_data
from src.models.train_model import train_model


def load_model(run_id, config):
    mlflow.set_tracking_uri(config['mlflow_tracking_uri'])
    return mlflow.sklearn.load_model(f"runs:/{run_id}/model")


def make_predictions(model, X):
    return model.predict(X)


def recommend_songs(predictions, song_ids, popular_songs, n=10):
    # Combine predictions with song IDs
    df = pd.DataFrame({'song_id': song_ids, 'predicted_popularity': predictions})

    # Sort by predicted popularity
    df_sorted = df.sort_values('predicted_popularity', ascending=False)

    # Get top N recommendations
    recommendations = df_sorted['song_id'].tolist()[:n]

    # If we have less than N recommendations, fill with popular songs
    if len(recommendations) < n:
        recommendations.extend([s for s in popular_songs if s not in recommendations])
        recommendations = recommendations[:n]

    return recommendations


def main():
    # Load configuration
    with open('config.json', 'r') as f:
        config = json.load(f)

    X, y, feature_names, popular_songs = prepare_data(config['prepared_data_key'])
    run_id = train_model(X, y, feature_names, popular_songs)

    model = load_model(run_id, config)

    # Load new data for predictions
    X_new = load_data_from_s3(config['prepared_data_key'], config['s3_bucket_name'])

    # Ensure the new data has the same features as the training data
    required_features = model.feature_names_in_
    X_new = X_new[required_features]

    # Make predictions
    predictions = make_predictions(model, X_new)

    # Load popular songs (you might want to store this in MLflow or elsewhere)
    popular_songs = mlflow.get_run(run_id).data.params['top_10_popular_songs'].split(',')

    # Get recommendations
    recommendations = recommend_songs(predictions, X_new.index, popular_songs)

    # Save predictions to S3
    predicition_output_key = "predictions.parquet"
    s3 = boto3.client('s3')
    buffer = BytesIO()
    X_new.to_parquet(buffer)
    s3.put_object(Bucket=config['s3_bucket_name'], Key=predicition_output_key, Body=buffer.getvalue())

    print(f"Predictions saved to S3://{config['s3_bucket_name']}/{predicition_output_key}")

    # Save recommendations to S3
    recommendations_output_key = "recommendations.csv"
    csv_buffer = pd.DataFrame({'recommendations': recommendations}).to_csv(index=False)
    s3.put_object(Bucket=config['s3_bucket_name'], Key=recommendations_output_key, Body=csv_buffer)

    print(f"Recommendations saved to S3://{config['s3_bucket_name']}/{recommendations_output_key}")


if __name__ == "__main__":
    main()
