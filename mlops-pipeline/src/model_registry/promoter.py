import mlflow
from mlflow.tracking import MlflowClient

def promote_if_better(model_name: str):
    client = MlflowClient()

    # Get experiment
    experiment = client.get_experiment_by_name(model_name)
    if experiment is None:
        raise ValueError(f"No experiment named {model_name} found in MLflow.")

    # Get latest run for that experiment
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=1
    )
    if not runs:
        raise ValueError(f"No runs found for model: {model_name}")

    latest_run = runs[0]
    new_accuracy = latest_run.data.metrics.get("best_accuracy")

    if new_accuracy is None:
        raise ValueError("Latest run has no 'best_accuracy' metric logged.")

    print(f"[Promotion] Latest model accuracy = {new_accuracy}")

    # Get current Production model
    try:
        prod_versions = client.get_latest_versions(model_name, stages=["Production"])
        if prod_versions:
            prod_version = prod_versions[0]
            prod_accuracy = float(prod_version.tags.get("accuracy", "0"))
            print(f"[Promotion] Production model accuracy = {prod_accuracy}")
        else:
            prod_accuracy = -1
            print("[Promotion] No model currently in Production.")
    except Exception as e:
        prod_accuracy = -1
        print("[Promotion] Error fetching Production model:", e)

    # Compare
    if new_accuracy > prod_accuracy:
        print("[Promotion] Promoting new model to Production...")

        new_version = client.create_model_version(
            name=model_name,
            source=f"{latest_run.info.artifact_uri}/best_model",
            run_id=latest_run.info.run_id
        )

        client.transition_model_version_stage(
            name=model_name,
            version=new_version.version,
            stage="Production"
        )

        client.set_model_version_tag(
            name=model_name,
            version=new_version.version,
            key="accuracy",
            value=str(new_accuracy)
        )

        print(f"[Promotion] Model version {new_version.version} promoted to Production.")

        return {
            "status": "promoted",
            "version": new_version.version,
            "accuracy": new_accuracy
        }
    else:
        print("[Promotion] New model NOT better. Keeping existing Production model.")
        return {
            "status": "rejected",
            "existing_accuracy": prod_accuracy,
            "new_accuracy": new_accuracy
        }
