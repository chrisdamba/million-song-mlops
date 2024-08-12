from __future__ import annotations
import json
import os
from io import BytesIO
from typing import Any

import boto3
import h5py
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def extract_song_data(h5_file):
    with h5py.File(h5_file, 'r') as f:
        song_data = {
            'song_id': f['metadata']['songs']['song_id'][0].decode('utf-8'),
            'title': f['metadata']['songs']['title'][0].decode('utf-8'),
            'artist_name': f['metadata']['songs']['artist_name'][0].decode('utf-8'),
            'duration': f['analysis']['songs']['duration'][0],
            'tempo': f['analysis']['songs']['tempo'][0],
            'loudness': f['analysis']['songs']['loudness'][0],
            'year': f['musicbrainz']['songs']['year'][0],
            'song_hotttnesss': f['metadata']['songs']['song_hotttnesss'][0]
        }

        # Extract timbre and chroma features
        segments_timbre = f['analysis']['segments_timbre'][:]
        segments_pitches = f['analysis']['segments_pitches'][:]

        song_data.update({
            f'timbre_{i}': segments_timbre[:, i].mean() for i in range(12)
        })
        song_data.update({
            f'chroma_{i}': segments_pitches[:, i].mean() for i in range(12)
        })

    return song_data


def process_dataset(root_dir):
    all_songs = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.h5'):
                file_path = os.path.join(root, file)
                song_data = extract_song_data(file_path)
                all_songs.append(song_data)

    return pd.DataFrame(all_songs)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Handle missing values
    numerical_cols = ['duration', 'tempo', 'loudness', 'year', 'song_hotttnesss']
    df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())

    string_cols = ['title', 'artist_name']
    df[string_cols] = df[string_cols].fillna('Unknown')

    # Remove duplicates
    df.drop_duplicates(subset='song_id', keep='first', inplace=True)

    # Handle outliers
    def remove_outliers(df, column, lower_percentile=0.01, upper_percentile=0.99):
        lower = df[column].quantile(lower_percentile)
        upper = df[column].quantile(upper_percentile)
        return df[(df[column] >= lower) & (df[column] <= upper)]

    for col in ['duration', 'tempo', 'loudness', 'song_hotttnesss']:
        df = remove_outliers(df, col)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    # Create a 'decade' feature
    df['decade'] = (df['year'] // 10) * 10

    # Create a 'tempo_category' feature
    df['tempo_category'] = pd.cut(df['tempo'],
                                  bins=[0, 60, 90, 120, 150, float('inf')],
                                  labels=['Very Slow', 'Slow', 'Moderate', 'Fast', 'Very Fast'])

    # Create a 'loudness_category' feature
    df['loudness_category'] = pd.cut(df['loudness'],
                                     bins=[-float('inf'), -20, -10, 0, float('inf')],
                                     labels=['Very Quiet', 'Quiet', 'Moderate', 'Loud'])

    return df


def load_data_from_s3(file_key: str, bucket_name: str) -> pd.DataFrame:
    """Load a Parquet file from S3 and return as a pandas DataFrame."""
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket=bucket_name, Key=file_key)
    parquet_file = BytesIO(obj['Body'].read())
    return pq.read_table(parquet_file).to_pandas()


def prepare_data(local_data_path: str = None) -> tuple[Any, Any, list[Any], Any]:
    """Prepare the dataset for machine learning."""
    if local_data_path and os.path.exists(local_data_path):
        print(f"Loading data from local Parquet file: {local_data_path}")
        df = pq.read_table(local_data_path).to_pandas()
    else:
        print(f"Processing raw dataset from: {config['raw_data_key']}")
        df = load_data_from_s3(config['raw_data_key'], config['s3_bucket_name'])
        df = clean_data(df)
        df = engineer_features(df)

        # Save processed data locally
        if local_data_path:
            table = pa.Table.from_pandas(df)
            pq.write_table(table, local_data_path)

        # Upload to S3
        print(f"Uploading processed data to S3: s3://{config['s3_bucket_name']}/{config['prepared_data_key']}")
        s3 = boto3.client('s3')
        buffer = BytesIO()
        df.to_parquet(buffer)
        s3.put_object(Bucket=config['s3_bucket_name'], Key=config['prepared_data_key'], Body=buffer.getvalue())

    # Prepare features and target
    target = 'song_hotttnesss'
    features = [col for col in df.columns if col != target]

    X = df[features]
    y = df[target]

    # Calculate song popularity
    song_popularity = df.groupby('song_id').size().sort_values(ascending=False)
    popular_songs = song_popularity.index.tolist()

    return X, y, features, popular_songs


if __name__ == "__main__":
    # Example usage
    with open('config.json', 'r') as f:
        config = json.load(f)

    X, y, feature_names, popular_songs = prepare_data(config['prepared_data_key'])
    print(f"Loaded {len(X)} samples with {len(feature_names)} features")
    print(f"Top 10 popular songs: {popular_songs[:10]}")
    print("Feature names:", feature_names)
    print("X shape:", X.shape)
    print("y shape:", y.shape)
