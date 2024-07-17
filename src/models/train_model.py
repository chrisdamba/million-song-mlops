import json
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from src.data.load_data import prepare_data

# Load configuration
with open('config.json', 'r') as f:
    config = json.load(f)


def train_model():
    mlflow.set_tracking_uri(config['mlflow_tracking_uri'])
    mlflow.set_experiment("million_song_popularity_prediction")

    # Load and prepare data
    X, y, feature_names = prepare_data()

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Train the model
    with mlflow.start_run():
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

    print(f"Model training completed. MSE: {mse}, R2: {r2}")
    return mlflow.active_run().info.run_id


if __name__ == "__main__":
    train_model()