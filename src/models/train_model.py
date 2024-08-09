import json
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from src.data.load_data import prepare_data


def train_model(X, y, feature_names, popular_songs) -> str:
    mlflow.set_tracking_uri(config['mlflow_tracking_uri'])
    mlflow.set_experiment("million_song_popularity_prediction")
    # Train the model
    with mlflow.start_run():
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define the model
        model = RandomForestRegressor(n_estimators=100, random_state=42)

        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Evaluate the model
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Log parameters
        mlflow.log_param("model_type", "RandomForestRegressor")
        mlflow.log_param("n_estimators", 100)

        # Log metrics
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)

        # Log model
        mlflow.sklearn.log_model(model, "model")

        # Log feature importance
        for name, importance in zip(feature_names, model.feature_importances_):
            mlflow.log_metric(f"feature_importance_{name}", importance)

        mlflow.log_param("top_10_popular_songs", ",".join(popular_songs[:10]))

    print(f"Model training completed. MSE: {mse}, R2: {r2}")
    return mlflow.active_run().info.run_id


if __name__ == "__main__":
    with open('config.json', 'r') as f:
        config = json.load(f)

    X, y, feature_names, popular_songs = prepare_data(config['prepared_data_key'])
    run_id = train_model(X, y, feature_names, popular_songs)
    print(f"MLflow run ID: {run_id}")
