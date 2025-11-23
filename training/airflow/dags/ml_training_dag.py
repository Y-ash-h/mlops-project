from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    "owner": "yash",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="ml_training_pipeline",
    default_args=default_args,
    start_date=datetime(2025, 11, 1),
    schedule_interval="@daily",
    catchup=False,
    tags=["mlops"],
) as dag:

    train_model = BashOperator(
        task_id="train_model",
        bash_command="python /usr/local/airflow/training/train.py",
    )

    train_model
