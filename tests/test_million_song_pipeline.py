import pytest
from unittest.mock import patch, MagicMock
from src.mage_ai_pipelines.million_song_pipeline import load_data_from_s3, train_and_evaluate_model, generate_recommendations, \
    save_recommendations


@pytest.fixture
def mock_config():
    return {
        's3_bucket_name': 'mock-bucket',
        'prepared_data_key': 'mock-data.parquet',
        'new_data_key': 'mock-new-data.parquet'
    }


@patch('src.data.load_data.prepare_data')
def test_load_data_from_s3(mock_prepare_data, mock_config):
    mock_prepare_data.return_value = ('X', 'y', 'feature_names', 'popular_songs')

    with patch('builtins.open', create=True) as mock_open:
        mock_open.return_value.__enter__.return_value.read.return_value = '{"key": "value"}'
        result = load_data_from_s3()

    assert result == ('X', 'y', 'feature_names', 'popular_songs')
    mock_prepare_data.assert_called_once_with(mock_config['s3_bucket_name'], mock_config['prepared_data_key'])


@patch('src.models.train_model.train_model')
def test_train_and_evaluate_model(mock_train_model):
    mock_train_model.return_value = 'mock_run_id'
    data = ('X', 'y', 'feature_names', 'popular_songs')

    result = train_and_evaluate_model(data)

    assert result == 'mock_run_id'
    mock_train_model.assert_called_once_with('X', 'y', 'feature_names', 'popular_songs')


@patch('mlflow.sklearn.load_model')
@patch('src.models.predict_model.make_predictions')
@patch('src.models.predict_model.recommend_songs')
@patch('src.pipelines.million_song_pipeline.load_data_from_s3')
def test_generate_recommendations(mock_load_data, mock_recommend, mock_predict, mock_load_model, mock_config):
    mock_load_model.return_value = MagicMock()
    mock_load_data.return_value = 'X_new'
    mock_predict.return_value = 'predictions'
    mock_recommend.return_value = 'recommendations'

    with patch('builtins.open', create=True) as mock_open:
        mock_open.return_value.__enter__.return_value.read.return_value = '{"key": "value"}'
        with patch('mlflow.get_run') as mock_get_run:
            mock_get_run.return_value.data.params = {'top_10_popular_songs': 'song1,song2'}
            result = generate_recommendations('mock_run_id')

    assert result == 'recommendations'
    mock_load_model.assert_called_once_with('runs:/mock_run_id/model')
    mock_load_data.assert_called_once_with(mock_config['s3_bucket_name'], mock_config['new_data_key'])
    mock_predict.assert_called_once()
    mock_recommend.assert_called_once()


@patch('boto3.client')
def test_save_recommendations(mock_boto3, mock_config):
    recommendations = ['song1', 'song2']

    with patch('builtins.open', create=True) as mock_open:
        mock_open.return_value.__enter__.return_value.read.return_value = '{"key": "value"}'
        result = save_recommendations(recommendations)

    assert 'Recommendations saved to S3' in result
    mock_boto3.assert_called_once_with('s3')
    mock_boto3.return_value.put_object.assert_called_once()