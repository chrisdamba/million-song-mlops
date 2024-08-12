from io import BytesIO

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from src.data.load_data import clean_data, engineer_features, load_data_from_s3, prepare_data


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'song_id': ['1', '2', '3'],
        'title': ['Song 1', 'Song 2', np.nan],
        'artist_name': ['Artist 1', np.nan, 'Artist 3'],
        'duration': [200, 180, np.nan],
        'tempo': [120, np.nan, 100],
        'loudness': [-5, -8, np.nan],
        'year': [2000, 2010, np.nan],
        'song_hotttnesss': [0.8, np.nan, 0.6]
    })


def test_clean_data(sample_df):
    cleaned_df = clean_data(sample_df)
    assert cleaned_df['title'].isnull().sum() == 0
    assert cleaned_df['artist_name'].isnull().sum() == 0
    assert cleaned_df['duration'].isnull().sum() == 0
    assert cleaned_df['tempo'].isnull().sum() == 0
    assert cleaned_df['loudness'].isnull().sum() == 0
    assert cleaned_df['year'].isnull().sum() == 0
    assert cleaned_df['song_hotttnesss'].isnull().sum() == 0
    assert len(cleaned_df) == len(sample_df)


def test_engineer_features(sample_df):
    engineered_df = engineer_features(sample_df)
    assert 'decade' in engineered_df.columns
    assert 'tempo_category' in engineered_df.columns
    assert 'loudness_category' in engineered_df.columns
    assert engineered_df['decade'].dtype == np.dtype('float64')
    assert engineered_df['tempo_category'].dtype == 'category'
    assert engineered_df['loudness_category'].dtype == 'category'


@patch('boto3.client')
def test_load_data_from_s3(mock_boto3):
    # Create a mock S3 client
    mock_s3 = MagicMock()
    mock_boto3.return_value = mock_s3

    # Create a mock for the S3 object body
    mock_body = MagicMock()
    mock_body.read.return_value = b'mock parquet data'

    # Set up the mock S3 get_object response
    mock_s3.get_object.return_value = {'Body': mock_body}

    # Mock the pyarrow read_table function
    with patch('pyarrow.parquet.read_table') as mock_read_table:
        mock_read_table.return_value.to_pandas.return_value = pd.DataFrame({'col1': [1, 2, 3]})

        # Call the function under test
        result = load_data_from_s3('mock_key', 'mock_bucket')

    # Assertions
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3
    mock_s3.get_object.assert_called_once_with(Bucket='mock_bucket', Key='mock_key')
    mock_body.read.assert_called_once()
    mock_read_table.assert_called_once()

    # Check that BytesIO was called with the correct data
    mock_read_table.assert_called_once_with(BytesIO(b'mock parquet data'))


@patch('src.data.load_data.load_data_from_s3')
@patch('src.data.load_data.clean_data')
@patch('src.data.load_data.engineer_features')
@patch('boto3.client')
def test_prepare_data(mock_boto3, mock_engineer, mock_clean, mock_load, sample_df):
    mock_load.return_value = sample_df
    mock_clean.return_value = sample_df
    mock_engineer.return_value = sample_df

    with patch('builtins.open', create=True) as mock_open:
        mock_open.return_value.__enter__.return_value.read.return_value = ('{"raw_data_key": "mock_key", '
                                                                           '"s3_bucket_name": "mock_bucket", '
                                                                           '"prepared_data_key": "mock_prepared_key"}')
        X, y, feature_names, popular_songs = prepare_data()

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert isinstance(feature_names, list)
    assert isinstance(popular_songs, list)
    mock_load.assert_called_once()
    mock_clean.assert_called_once()
    mock_engineer.assert_called_once()
    mock_boto3.assert_called_once_with('s3')
