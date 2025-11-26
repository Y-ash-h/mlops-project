import mlflow
from mlflow.tracking import MlflowClient
import os

def promote_if_better(model_name: str):
    client = MlflowClient()

    # Get experiment name from environment or use a default
    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "demo_experiment")
    
    # Get experiment
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"No experiment named {experiment_name} found in MLflow.")

    # Get latest parent run (non-nested) for that experiment
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.mlflow.parentRunId = ''",
        order_by=["attributes.start_time DESC"],
        max_results=1
    )
    if not runs:
        # Fallback to any run if no parent runs found
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["attributes.start_time DESC"],
            max_results=1
        )
    
    if not runs:
        raise ValueError(f"No runs found for experiment: {experiment_name}")

    latest_run = runs[0]
    print(f"[Promotion] Latest run ID: {latest_run.info.run_id}")
    print(f"[Promotion] Available metrics: {list(latest_run.data.metrics.keys())}")
    
    # Try multiple metric names: rmse, best_accuracy, accuracy
    new_metric = (latest_run.data.metrics.get("rmse") or 
                  latest_run.data.metrics.get("best_accuracy") or 
                  latest_run.data.metrics.get("accuracy"))

    if new_metric is None:
        raise ValueError(f"Latest run has no 'rmse', 'best_accuracy', or 'accuracy' metric logged. Available metrics: {list(latest_run.data.metrics.keys())}")

    # Determine metric name and comparison direction
    if "rmse" in latest_run.data.metrics:
        metric_name = "rmse"
        is_lower_better = True
    elif "best_accuracy" in latest_run.data.metrics:
        metric_name = "best_accuracy"
        is_lower_better = False
    else:
        metric_name = "accuracy"
        is_lower_better = False
    
    print(f"[Promotion] Latest model {metric_name} = {new_metric}")

    # Get current Production model
    try:
        prod_versions = client.get_latest_versions(model_name, stages=["Production"])
        if prod_versions:
            prod_version = prod_versions[0]
            prod_metric = float(prod_version.tags.get(metric_name, "999" if is_lower_better else "0"))
            print(f"[Promotion] Production model {metric_name} = {prod_metric}")
        else:
            prod_metric = 999 if is_lower_better else -1
            print("[Promotion] No model currently in Production.")
    except Exception as e:
        prod_metric = 999 if is_lower_better else -1
        print("[Promotion] Error fetching Production model:", e)

    # Compare - for RMSE lower is better, for accuracy higher is better
    is_better = (new_metric < prod_metric) if is_lower_better else (new_metric > prod_metric)
    
    if is_better:
        print("[Promotion] Promoting new model to Production...")

        # Use 'model' artifact path (what train.py logs)
        model_uri = f"{latest_run.info.artifact_uri}/model"
        
        new_version = client.create_model_version(
            name=model_name,
            source=model_uri,
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
            key=metric_name,
            value=str(new_metric)
        )

        print(f"[Promotion] Model version {new_version.version} promoted to Production.")

        return {
            "status": "promoted",
            "version": new_version.version,
            metric_name: new_metric
        }
    else:
        print("[Promotion] New model NOT better. Keeping existing Production model.")
        return {
            "status": "rejected",
            f"existing_{metric_name}": prod_metric,
            f"new_{metric_name}": new_metric
        }
