import mlflow

def train_text_model(data_dir: str):
    print("[Train-Text] Training text model (stub with MLflow)...")

    with mlflow.start_run():
        mlflow.set_tag("model_type", "text")
        mlflow.log_metric("placeholder_metric", 0.0)  # replace later
        mlflow.log_param("training_mode", "text-model")

    return {"status": "trained", "type": "text"}
