import os
from pathlib import Path


class Config:
    _repo_root = Path(__file__).resolve().parents[1]

    # Allow overriding data root via DATA_DIR env var (useful inside Airflow)
    data_dir = os.getenv("DATA_DIR", str(_repo_root / "data"))

    # Data paths
    raw_data_path = os.path.join(data_dir, "raw", "data.csv")
    processed_data_path = os.path.join(data_dir, "processed", "processed.csv")

    # Target column
    target = "target"

    # Model hyperparameters
    n_estimators = 100
    max_depth = 5

config = Config()
