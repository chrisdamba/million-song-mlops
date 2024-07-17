import json

import boto3
import mlflow
import pandas as pd
from mage_ai.data_preparation.decorators import data_loader, transformer

from src.data.load_data import prepare_data
from src.models.predict_model import make_predictions, recommend_songs
from src.models.train_model import train_model


@data_loader
def load_data_from_s3(*args, **kwargs):
    """
    Load data from S3 using the configuration
    """
    with open('config.json', 'r') as f:
        config = json.load(f)

    X, y, feature_names, popular_songs = prepare_data(config['s3_bucket_name'], config['prepared_data_key'])
    return X, y, feature_names, popular_songs


@transformer
def train_and_evaluate_model(data, *args, **kwargs):
    """
    Train and evaluate the model
    """
    X, y, feature_names, popular_songs = data
    run_id = train_model(X, y, feature_names, popular_songs)
    return run_id


@transformer
def generate_recommendations(run_id, *args, **kwargs):
    """
    Generate recommendations using the trained model
    """
    with open('config.json', 'r') as f:
        config = json.load(f)

    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    X_new = load_data_from_s3(config['s3_bucket_name'], config['new_data_key'])
    predictions = make_predictions(model, X_new)

    popular_songs = mlflow.get_run(run_id).data.params['top_10_popular_songs'].split(',')
    recommendations = recommend_songs(predictions, X_new.index, popular_songs)

    return recommendations


@transformer
def save_recommendations(recommendations, *args, **kwargs):
    """
    Save recommendations to S3
    """
    with open('config.json', 'r') as f:
        config = json.load(f)

    output_key = "recommendations.csv"
    s3 = boto3.client('s3')
    csv_buffer = pd.DataFrame({'recommendations': recommendations}).to_csv(index=False)
    s3.put_object(Bucket=config['s3_bucket_name'], Key=output_key, Body=csv_buffer)

    return f"Recommendations saved to S3://{config['s3_bucket_name']}/{output_key}"