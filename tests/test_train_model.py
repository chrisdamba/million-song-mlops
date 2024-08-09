import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from src.models.train_model import train_model


@pytest.fixture
def mock_data():
    X = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
    y = pd.Series([0, 1, 0])
    feature_names = ['feature1', 'feature2']
    popular_songs = ['song1', 'song2']
    return X, y, feature_names, popular_songs


@patch('mlflow.set_tracking_uri')
@patch('mlflow.set_experiment')
@patch('mlflow.start_run')
@patch('mlflow.sklearn.log_model')
@patch('mlflow.log_param')
@patch('mlflow.log_metric')
def test_train_model(mock_log_metric, mock_log_param, mock_log_model, mock_start_run,
                     mock_set_experiment, mock_set_tracking_uri, mock_data):
    X, y, feature_names, popular_songs = mock_data
    mock_start_run.return_value.__enter__.return_value = MagicMock()
    mock_start_run.return_value.__enter__.return_value.info.run_id = 'mock_run_id'

    run_id = train_model(X, y, feature_names, popular_songs)

    assert run_id == 'mock_run_id'
    mock_set_tracking_uri.assert_called_once()
    mock_set_experiment.assert_called_once_with("million_song_popularity_prediction")
    mock_start_run.assert_called_once()
    mock_log_model.assert_called_once()
    assert mock_log_param.call_count > 0
    assert mock_log_metric.call_count > 0
