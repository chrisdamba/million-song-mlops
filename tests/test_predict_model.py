import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from src.models.predict_model import load_model, make_predictions, recommend_songs, main


@pytest.fixture
def mock_config():
    return {
        'mlflow_tracking_uri': 'http://mock-mlflow-server',
        's3_bucket_name': 'mock-bucket',
        'prepared_data_key': 'mock-data.parquet'
    }


@pytest.fixture
def mock_model():
    model = MagicMock()
    model.predict.return_value = [0.5, 0.7, 0.3]
    return model


def test_load_model(mock_config):
    with patch('mlflow.sklearn.load_model') as mock_load:
        mock_load.return_value = 'mock_model'
        model = load_model('mock_run_id', mock_config)
        assert model == 'mock_model'
        mock_load.assert_called_once_with('runs:/mock_run_id/model')


def test_make_predictions(mock_model):
    X = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
    predictions = make_predictions(mock_model, X)
    assert len(predictions) == 3
    mock_model.predict.assert_called_once_with(X)


def test_recommend_songs():
    predictions = [0.5, 0.7, 0.3]
    song_ids = ['song1', 'song2', 'song3']
    popular_songs = ['song4', 'song5']
    recommendations = recommend_songs(predictions, song_ids, popular_songs, n=2)
    assert recommendations == ['song2', 'song1']


@patch('src.models.predict_model.prepare_data')
@patch('src.models.predict_model.train_model')
@patch('src.models.predict_model.load_model')
@patch('src.models.predict_model.load_data_from_s3')
@patch('boto3.client')
def test_main(mock_boto3, mock_load_data, mock_load_model, mock_train, mock_prepare, mock_config):
    mock_prepare.return_value = (pd.DataFrame(), pd.Series(), ['feature1'], ['song1'])
    mock_train.return_value = 'mock_run_id'
    mock_load_model.return_value = MagicMock()
    mock_load_data.return_value = pd.DataFrame({'feature1': [1, 2, 3]})

    with patch('builtins.open', create=True) as mock_open:
        mock_open.return_value.__enter__.return_value.read.return_value = '{"key": "value"}'
        main()

    mock_prepare.assert_called_once()
    mock_train.assert_called_once()
    mock_load_model.assert_called_once()
    mock_load_data.assert_called_once()
    mock_boto3.assert_called()
