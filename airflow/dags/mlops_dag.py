from datetime import datetime
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.dummy_operator import DummyOperator
from airflow.operators.bash import BashOperator

import sys
import os

sys.path.insert(0, "/opt/airflow/mlops-pipeline")

# Import your pipeline modules from mlops-pipeline
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../mlops-pipeline")))

from src.utils.data_router import detect_data_type
from src.data_preprocessing.preprocess_router import route_preprocessing
from src.model_training.model_router import route_training
from src.model_registry.promoter import promote_if_better

# Dummy ingestion & validation placeholders
from src.data_ingestion.ingest import DataIngestion
from src.data_validation.validate import DataValidation

# Import alerting and model card generation
from src.alerts.slack_notifier import (
    alert_validation_fail,
    alert_drift_detected,
    alert_promotion_success,
    alert_promotion_blocked,
)
from src.model_card.generate_model_card import generate_model_card


# DAG definition
default_args = {
    "owner": "yashvardhan",
    "start_date": datetime(2024, 1, 1),
}

# Use absolute paths for data directories inside the container
DATA_BASE_DIR = os.environ.get("DATA_DIR", "/opt/airflow/mlops-pipeline/data")
RAW_DATA_DIR = str(Path(DATA_BASE_DIR) / "raw")
TRAIN_DATA_PATH = str(Path(DATA_BASE_DIR) / "train.csv")
VALIDATION_DATA_PATH = str(Path(DATA_BASE_DIR) / "validation.csv")
MONITORING_REPORT_PATH = str(Path(DATA_BASE_DIR) / "monitoring" / "drift_report.html")
LABEL_COLUMN = os.environ.get("LABEL_COLUMN", "target")

with DAG(
    dag_id="mlops_full_pipeline",
    default_args=default_args,
    schedule_interval="@once",
    catchup=False,
) as dag:

    # 1) Detect data type
    def detect(**kwargs):
        # Use absolute path
        dtype = detect_data_type(RAW_DATA_DIR)
        print("Detected data type:", dtype)
        return dtype

    detect_task = PythonOperator(
        task_id="detect_data_type",
        python_callable=detect
    )


    # 2) Ingest Data
    def ingest(**kwargs):
        ing = DataIngestion()
        files = ing.load_and_save()
        print("Ingested files:", files)
        return "ok"

    ingest_task = PythonOperator(
        task_id="ingest_data",
        python_callable=ingest
    )


    # 3) Validate Data
    def validate(**kwargs):
        val = DataValidation()
        # load the data again for now
        import pandas as pd
        import os
        # Use absolute path
        try:
            df = pd.read_csv(os.path.join(RAW_DATA_DIR, "data.csv"))
            val.validate(df)
            return "ok"
        except Exception as e:
            alert_validation_fail(str(e))
            raise

    validate_task = PythonOperator(
        task_id="validate_data",
        python_callable=validate
    )


    # 4) Preprocessing Router
    def preprocess(**kwargs):
        # Use absolute path
        result = route_preprocessing(RAW_DATA_DIR)
        print("Preprocessing result:", result)
        return result

    preprocess_task = PythonOperator(
        task_id="preprocess_data",
        python_callable=preprocess
    )


    # NEW: Monitoring Task (Data/Target drift)
    monitor_data = BashOperator(
        task_id="monitor_data_drift",
        bash_command=(
            "python /opt/airflow/mlops-pipeline/src/model_monitoring/monitor_drift.py "
            f"--reference {TRAIN_DATA_PATH} "
            f"--current {VALIDATION_DATA_PATH} "
            f"--output {MONITORING_REPORT_PATH} "
            f"--target-column {LABEL_COLUMN}"
        ),
    )

    # Check drift and send alert
    def check_drift_alert(**kwargs):
        # Parse the drift report to check for significant drift
        # For now, just alert that monitoring completed
        alert_drift_detected(MONITORING_REPORT_PATH)
        return "ok"

    drift_alert_task = PythonOperator(
        task_id="drift_alert_check",
        python_callable=check_drift_alert
    )


    # 5) Training Router
    def train(**kwargs):
        # Use absolute path
        result = route_training(RAW_DATA_DIR)
        print("Training result:", result)
        return result

    train_task = PythonOperator(
        task_id="train_model",
        python_callable=train
    )


    # NEW: Branching Logic
    def check_if_explainable(**kwargs):
        dtype = detect_data_type(RAW_DATA_DIR)
        print(f"Detected data type for explanation check: {dtype}")
        if dtype == 'tabular':
            return 'explain_model'
        else:
            return 'skip_explain'

    check_explainability = BranchPythonOperator(
        task_id='check_explainability',
        python_callable=check_if_explainable,
    )

    skip_explain = DummyOperator(task_id='skip_explain')


    # 6) Explainability (call stub for now)
    def explain(**kwargs):
        print("Explainability stub running...")
        return "explained"

    explain_task = PythonOperator(
        task_id="explain_model",
        python_callable=explain
    )


    # 7) Promotion Step
    def promote(**kwargs):
        print("Attempting model promotion...")
        try:
            # Set experiment name to match training
            import os
            os.environ["MLFLOW_EXPERIMENT_NAME"] = "tabular_model"
            result = promote_if_better("tabular_model")
            print(result)
            if "promoted" in str(result).lower() or "success" in str(result).lower():
                alert_promotion_success("tabular_model", "Better metrics detected")
            else:
                alert_promotion_blocked("tabular_model", "No improvement detected")
        except Exception as e:
            print(f"Promotion error (non-critical): {e}")
            print("Model was trained and registered successfully. Promotion step skipped.")
            alert_promotion_blocked("tabular_model", str(e))
            # Don't raise - allow pipeline to continue
        return "done"

    promote_task = PythonOperator(
        task_id="promote_model",
        python_callable=promote,
        trigger_rule='none_failed_min_one_success'
    )

    # 8) Generate Model Card
    def generate_card(**kwargs):
        # Get the latest run ID from MLflow (simplified - you can enhance this)
        import mlflow
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
        client = mlflow.tracking.MlflowClient()
        
        # Get the latest run from the experiment
        experiment = mlflow.get_experiment_by_name(
            os.environ.get("MLFLOW_EXPERIMENT_NAME", "demo_experiment")
        )
        if experiment:
            runs = client.search_runs([experiment.experiment_id], order_by=["start_time DESC"], max_results=1)
            if runs:
                run_id = runs[0].info.run_id
                print(f"Generating model card for run: {run_id}")
                
                # Check if SHAP artifacts exist
                artifacts = [f.path for f in client.list_artifacts(run_id)]
                shap_path = None
                for art in artifacts:
                    if "shap" in art.lower() and art.endswith(".png"):
                        shap_path = f"/mlflow-artifacts/{experiment.experiment_id}/{run_id}/artifacts/{art}"
                        break
                
                output_path = str(Path(DATA_BASE_DIR) / "model_card.html")
                generate_model_card(
                    run_id=run_id,
                    model_name="tabular_model",
                    shap_path=shap_path,
                    output_path=output_path,
                )
                print(f"Model card generated: {output_path}")
        return "ok"

    model_card_task = PythonOperator(
        task_id="generate_model_card",
        python_callable=generate_card,
        trigger_rule='none_failed_min_one_success'
    )


    # DAG Task Flow
    detect_task >> ingest_task >> validate_task >> preprocess_task >> train_task
    train_task >> monitor_data >> drift_alert_task >> check_explainability
    check_explainability >> explain_task >> promote_task >> model_card_task
    check_explainability >> skip_explain >> promote_task >> model_card_task
