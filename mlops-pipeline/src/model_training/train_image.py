import mlflow

def train_image_model(data_dir: str):
    print("[Train-Image] Training image model (stub with MLflow)...")

    with mlflow.start_run():
        mlflow.set_tag("model_type", "image")
        mlflow.log_metric("placeholder_metric", 0.0)
        mlflow.log_param("training_mode", "image-model")

    return {"status": "trained", "type": "image"}
