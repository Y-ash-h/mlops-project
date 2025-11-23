import os

import mlflow
import mlflow.sklearn
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

def train_tabular_model(data_dir: str):
    print("[Train-Tabular] Training LightGBM + XGBoost with MLflow competition...")

    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("tabular_model")

    # Load CSV file
    csv_files = list(Path(data_dir).glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError("No CSV files found for tabular training")

    df = pd.read_csv(csv_files[0])

    target = df.columns[-1]
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "lightgbm": LGBMClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=-1
        ),
        "xgboost": XGBClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=6, eval_metric="logloss"
        )
    }

    results = {}

    with mlflow.start_run() as parent_run:
        for model_name, model in models.items():
            print(f"[Train-Tabular] Training {model_name}...")

            with mlflow.start_run(run_name=model_name, nested=True):
                mlflow.log_param("model_type", model_name)

                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                acc = accuracy_score(y_test, preds)

                mlflow.log_metric("accuracy", acc)
                mlflow.sklearn.log_model(model, artifact_path=f"{model_name}_model")

                results[model_name] = {"model": model, "accuracy": acc}

        # Compare models
        best_model_name = max(results, key=lambda x: results[x]["accuracy"])
        best_accuracy = results[best_model_name]["accuracy"]
        best_model = results[best_model_name]["model"]

        print(f"[Train-Tabular] Winner: {best_model_name} with accuracy {best_accuracy}")

        # Log the best model in parent run
        mlflow.log_metric("best_accuracy", best_accuracy)
        mlflow.set_tag("best_model", best_model_name)
        mlflow.sklearn.log_model(best_model, artifact_path="best_model")

        model_uri = f"runs:/{parent_run.info.run_id}/best_model"

        registered_model = mlflow.register_model(model_uri, "tabular_model")

        return {
            "status": "trained",
            "type": "tabular",
            "best_model": best_model_name,
            "accuracy": best_accuracy,
            "model_path": mlflow.get_artifact_uri("best_model"),
            "model_version": registered_model.version,
        }
