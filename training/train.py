import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import math
from pathlib import Path
import glob
import os

# CONNECT TO MLflow INSIDE DOCKER
mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("demo_experiment")

# Prefer training split produced by preprocessing (feature-engineered)
DATA_BASE = os.environ.get("DATA_DIR", "/opt/airflow/mlops-pipeline/data")
TRAIN_CSV = os.path.join(DATA_BASE, "train.csv")
FEATURES_DIR = os.path.join(DATA_BASE, "features")

if os.path.exists(TRAIN_CSV):
    df = pd.read_csv(TRAIN_CSV)
    print(f"[train.py] Loaded training split from {TRAIN_CSV}")
else:
    # Fallback: concatenate all engineered feature CSVs
    feature_files = sorted(glob.glob(os.path.join(FEATURES_DIR, "*.csv")))
    if not feature_files:
        raise FileNotFoundError(
            f"No feature-engineered files found at {FEATURES_DIR} and {TRAIN_CSV} missing. "
            "Run the preprocessing DAG first."
        )
    df_list = [pd.read_csv(p) for p in feature_files]
    df = pd.concat(df_list, ignore_index=True)
    print(f"[train.py] Concatenated feature files: {feature_files} -> {len(df)} rows")

label_col = os.environ.get("LABEL_COLUMN", "target")
if label_col not in df.columns:
    raise ValueError(f"Label column '{label_col}' not found in training data.")
X_train = df.drop(columns=[label_col])
y_train = df[label_col]

X_train, X_test, y_train, y_test = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# End any active run from previous attempts
if mlflow.active_run():
    mlflow.end_run()

with mlflow.start_run(run_name="RandomForest_v1"):
    n_estimators = 100
    max_depth = 5

    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    rmse = math.sqrt(mse)

    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_metric("rmse", rmse)
    
    # Save training feature list for validator to use
    training_features = list(X_train.columns)
    feat_file = "training_features.txt"
    with open(feat_file, "w") as f:
        f.write("\n".join(training_features))
    mlflow.log_artifact(feat_file, artifact_path="training")
    print(f"[train.py] Saved and logged training features ({len(training_features)}): {training_features}")
    
    mlflow.sklearn.log_model(model, "model")

    print(f"Training done. RMSE: {rmse:.4f}")