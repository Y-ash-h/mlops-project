# /opt/airflow/dags/ml_training_dag.py
import os
import sys
import logging
import traceback
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

# allow imports from repo
sys.path.insert(0, "/opt/airflow/mlops-pipeline")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../mlops-pipeline")))

# new ingestion/preprocessing modules
from src.data_ingestion.ingest import DataIngestion
from src.data_preprocessing.preprocess_router import PreprocessRouter

# Standard ML libs
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

# MLflow
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

# SHAP optional
try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ml_training_pipeline")

# DAG defaults
default_args = {
    "owner": "yash",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

# Paths & defaults
DATA_BASE_DIR = os.environ.get("DATA_DIR", "/opt/airflow/mlops-pipeline/data")
RAW_DIR = str(Path(DATA_BASE_DIR) / "raw")
CLEAN_DIR = str(Path(DATA_BASE_DIR) / "clean")
FEATURE_DIR = str(Path(DATA_BASE_DIR) / "features")
INGESTED_DIR = str(Path(DATA_BASE_DIR) / "ingested")
VALIDATION_CSV_DEFAULT = str(Path(DATA_BASE_DIR) / "validation.csv")
TRAIN_CSV_DEFAULT = str(Path(DATA_BASE_DIR) / "train.csv")
LABEL_COLUMN_DEFAULT = os.environ.get("LABEL_COLUMN", "target")
MODEL_REGISTRY_NAME = os.environ.get("MODEL_REGISTRY_NAME", "tabular_model")
EXPLAIN_SAMPLE_SIZE = int(os.environ.get("EXPLAIN_SAMPLE_SIZE", "500"))

# --------------------
# Promotion helper (kept from earlier)
# --------------------
def promote_if_better_mlflow(new_run_id: str, model_name: str = MODEL_REGISTRY_NAME):
    client = MlflowClient(tracking_uri=mlflow.get_tracking_uri())
    logger.info("Checking production model for %s", model_name)
    prod_rmse = None
    try:
        prod_versions = client.get_latest_versions(name=model_name, stages=["Production"])
    except Exception as e:
        logger.warning("Could not fetch model registry versions (may not exist yet): %s", e)
        prod_versions = []

    if prod_versions:
        prod_version = prod_versions[0]
        try:
            prod_run = client.get_run(prod_version.run_id)
            prod_rmse = float(prod_run.data.metrics.get("val_rmse", float("inf")))
            logger.info("Current production RMSE: %s (version %s)", prod_rmse, prod_version.version)
        except Exception:
            logger.exception("Failed to read production run metrics")

    try:
        new_run = client.get_run(new_run_id)
        new_rmse = float(new_run.data.metrics.get("val_rmse", float("inf")))
    except Exception:
        logger.exception("Failed to read new run metrics")
        new_rmse = float("inf")

    logger.info("New run %s RMSE = %s", new_run_id, new_rmse)

    if prod_rmse is None or new_rmse < prod_rmse:
        logger.info("New model is better (or no production model). Registering and promoting...")
        source = f"runs:/{new_run_id}/model"
        try:
            try:
                client.get_registered_model(name=model_name)
                logger.info("Registered model '%s' already exists.", model_name)
            except Exception:
                logger.info("Registered model '%s' not found — creating it.", model_name)
                try:
                    client.create_registered_model(name=model_name)
                    logger.info("Registered model '%s' created.", model_name)
                except Exception as ce:
                    logger.warning("create_registered_model raised: %s — continuing (maybe created concurrently)", ce)

            mv = client.create_model_version(name=model_name, source=source, run_id=new_run_id)
            logger.info("Created model version %s for model %s", mv.version, model_name)

            try:
                client.transition_model_version_stage(
                    name=model_name,
                    version=mv.version,
                    stage="Production",
                    archive_existing_versions=True
                )
                logger.info("Promoted model %s to Production as version %s", model_name, mv.version)
                return {"promoted": True, "new_rmse": new_rmse, "prev_rmse": prod_rmse, "version": mv.version}
            except Exception:
                logger.exception("Failed to transition model version stage; leaving version unpromoted.")
                return {"promoted": False, "new_rmse": new_rmse, "prev_rmse": prod_rmse, "error": traceback.format_exc()}

        except Exception:
            logger.exception("Failed to register/promote model")
            return {"promoted": False, "new_rmse": new_rmse, "prev_rmse": prod_rmse, "error": traceback.format_exc()}
    else:
        logger.info("New model did not beat production. Not promoting.")
        return {"promoted": False, "new_rmse": new_rmse, "prev_rmse": prod_rmse}

# --------------------
# DAG tasks
# --------------------
def ingest_raw_data(**kwargs):
    """Simple ingestion: copy raw files from RAW_DIR into data/raw and return list of files ingested."""
    logger.info("Ingest: reading raw data from %s", RAW_DIR)
    # ensure ingested dir exists and pass it as out_dir so we avoid copying onto raw source
    os.makedirs(INGESTED_DIR, exist_ok=True)
    ing = DataIngestion(raw_dir=RAW_DIR, out_dir=INGESTED_DIR)
    files = ing.load_and_save()
    logger.info("Ingested files: %s", files)
    return {"ingested_files": files}

def clean_data(**kwargs):
    """Basic cleaning: read raw, drop duplicate rows, fill na (simple rules)."""
    logger.info("Cleaning data")
    pr = PreprocessRouter(raw_dir=RAW_DIR, clean_dir=CLEAN_DIR, feature_dir=FEATURE_DIR)
    cleaned_paths = pr.clean_all()
    logger.info("Cleaned and saved: %s", cleaned_paths)
    return {"cleaned": cleaned_paths}

def feature_engineer(**kwargs):
    """Feature engineering: creates feature CSVs in features/ and returns path."""
    logger.info("Feature engineering")
    pr = PreprocessRouter(raw_dir=RAW_DIR, clean_dir=CLEAN_DIR, feature_dir=FEATURE_DIR)
    feat_path = pr.feature_engineer_all()
    logger.info("Feature files saved: %s", feat_path)
    return {"features": feat_path}

def split_data(**kwargs):
    """Split features into train/validation and save to DATA_BASE_DIR/train.csv & validation.csv"""
    logger.info("Splitting into train/validation")
    pr = PreprocessRouter(raw_dir=RAW_DIR, clean_dir=CLEAN_DIR, feature_dir=FEATURE_DIR)
    train_path, val_path = pr.train_val_split(os.path.join(DATA_BASE_DIR, "train.csv"), os.path.join(DATA_BASE_DIR, "validation.csv"))
    logger.info("Train/Validation saved: %s, %s", train_path, val_path)
    return {"train": train_path, "validation": val_path}

def run_training(**kwargs):
    import subprocess
    try:
        logger.info("Starting training via /opt/airflow/training/train.py")
        result = subprocess.run(
            [sys.executable, "/opt/airflow/training/train.py"],
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info("Training stdout:\n%s", result.stdout)
    except subprocess.CalledProcessError as e:
        logger.error("Training failed:\nstdout:\n%s\nstderr:\n%s", e.stdout, e.stderr)
        raise

def validate_model_mlflow(**kwargs):
    logger.info("Starting MLflow-backed validation")
    try:
        mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        mlflow.set_tracking_uri(mlflow_uri)
        experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "demo_experiment")
        artifact_path = os.environ.get("MLFLOW_ARTIFACT_PATH", "model")
        validation_csv = os.environ.get("VALIDATION_CSV", VALIDATION_CSV_DEFAULT)
        label_col = os.environ.get("LABEL_COLUMN", LABEL_COLUMN_DEFAULT)
        model_registry_name = os.environ.get("MODEL_REGISTRY_NAME", MODEL_REGISTRY_NAME)
        promote_on_improvement = os.environ.get("PROMOTE_ON_IMPROVEMENT", "true").lower() in ("1", "true", "yes")

        if not os.path.exists(validation_csv):
            raise FileNotFoundError(f"Validation CSV not found at {validation_csv}")

        df_val = pd.read_csv(validation_csv)
        if label_col not in df_val.columns:
            raise ValueError(f"Label column '{label_col}' not present in validation CSV")

        X_val = df_val.drop(columns=[label_col])
        y_val = df_val[label_col]

        client = MlflowClient(tracking_uri=mlflow_uri)
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            raise RuntimeError(f"Experiment '{experiment_name}' not found")

        runs = client.search_runs([experiment.experiment_id], order_by=["attributes.start_time DESC"], max_results=20)
        chosen_run = None
        for r in runs:
            if r.info.status == "FINISHED":
                chosen_run = r
                break
        if chosen_run is None:
            raise RuntimeError(f"No finished runs found in experiment {experiment_name}")

        run_id = chosen_run.info.run_id
        logger.info("Selected run_id=%s for validation", run_id)

        model_uri = f"runs:/{run_id}/{artifact_path}"
        logger.info("Loading model from %s", model_uri)
        model = mlflow.pyfunc.load_model(model_uri)

        # -- STRICT feature contract enforcement --
        # load training feature list from artifact if possible, else try model signature
        expected_cols = None
        # 1) Try to load training_features.txt from run artifacts (preferred)
        try:
            remote_feat = "training/training_features.txt"
            local_tmp_dir = "/tmp/mlflow_training_artifacts"
            os.makedirs(local_tmp_dir, exist_ok=True)
            client.download_artifacts(run_id=run_id, path=remote_feat, dst_path=local_tmp_dir)
            local_file = os.path.join(local_tmp_dir, "training_features.txt")
            if os.path.exists(local_file):
                with open(local_file, "r") as f:
                    expected_cols = [l.strip() for l in f.readlines() if l.strip()]
                logger.info("Loaded training feature list from artifact: %s", local_file)
        except Exception:
            expected_cols = None

        # 2) Fallback: try model metadata/signature (best-effort)
        if expected_cols is None:
            try:
                md = getattr(model, "metadata", None)
                if md is not None:
                    try:
                        sig = md.get_input_schema()
                        if sig and getattr(sig, "columns", None):
                            expected_cols = [c.name for c in sig.columns]
                    except Exception:
                        try:
                            sig2 = getattr(md, "signature", None)
                            if sig2 and hasattr(sig2, "inputs"):
                                try:
                                    expected_cols = [i['name'] for i in sig2.inputs]
                                except Exception:
                                    expected_cols = [getattr(i, "name") for i in sig2.inputs]
                        except Exception:
                            pass
            except Exception:
                expected_cols = None

        # If we still don't have an expected list, fail loudly — we require an explicit feature contract
        if not expected_cols:
            msg = ("Strict validation requires a saved training feature list (artifact 'training/training_features.txt') "
                   "or a model signature. None found for run_id=%s. Please ensure train.py logs 'training/training_features.txt' "
                   "or trains a model with an input signature." ) % run_id
            logger.error(msg)
            raise RuntimeError(msg)

        # Now enforce exact match — no reordering, no dropping extra columns
        val_cols = list(X_val.columns)
        expected_set = set(expected_cols)
        val_set = set(val_cols)

        missing = [c for c in expected_cols if c not in val_set]
        unexpected = [c for c in val_cols if c not in expected_set]

        if missing or unexpected:
            # helpful diagnostic
            logger.error("Feature contract violation detected for run_id=%s", run_id)
            logger.error("Expected features (%d): %s", len(expected_cols), expected_cols)
            logger.error("Validation features (%d): %s", len(val_cols), val_cols)
            if missing:
                logger.error("Missing required features (present during training but absent in validation): %s", missing)
            if unexpected:
                logger.error("Unexpected features (present in validation but not during training): %s", unexpected)
            err_msg = ("Strict feature contract failed. Missing: %s. Unexpected: %s. "
                       "Regenerate train/validation from the same feature-engineered dataframe.") % (missing, unexpected)
            raise ValueError(err_msg)

        # exact set match — ensure same order as expected
        if val_cols != expected_cols:
            logger.error("Feature order mismatch. Expected order: %s; Validation order: %s", expected_cols, val_cols)
            raise ValueError("Feature order mismatch — validation will not proceed under strict contract.")

        preds = model.predict(X_val)
        if isinstance(preds, (pd.Series, pd.DataFrame)):
            preds_arr = np.array(preds).squeeze()
        else:
            preds_arr = np.array(preds)

        if np.isnan(preds_arr).any():
            raise ValueError("Model produced NaN predictions")
        if preds_arr.shape[0] != X_val.shape[0]:
            raise ValueError("Prediction length mismatch")

        mse = mean_squared_error(y_val, preds_arr)
        rmse = float(np.sqrt(mse))
        logger.info("Validation RMSE: %.4f", rmse)

        mlflow.start_run(run_id=run_id)
        mlflow.log_metric("val_rmse", float(rmse))

        preds_df = X_val.copy()
        preds_df[label_col + "_actual"] = y_val.values
        preds_df[label_col + "_pred"] = preds_arr
        preds_path = "/tmp/val_predictions.csv"
        preds_df.to_csv(preds_path, index=False)
        mlflow.log_artifact(preds_path, artifact_path="validation")
        mlflow.end_run()

        result = {"run_id": run_id, "val_rmse": float(rmse)}
        logger.info("Validation complete: %s", result)

        if promote_on_improvement:
            promo_res = promote_if_better_mlflow(run_id, model_name=model_registry_name)
            logger.info("Promotion result: %s", promo_res)
            result["promotion"] = promo_res

        return result
    except Exception as e:
        logger.exception("Validation failed: %s", e)
        raise

def explain_model(**kwargs):
    """
    Explainability step:
    - load production model (or fall back to latest run)
    - attempt to load training feature list from artifacts or model metadata (with extensive logging)
    - if missing, search other runs in the experiment for the artifact
    - as a last resort, use validation CSV columns as fallback (with WARNING)
    - compute SHAP or permutation importance
    - save artifacts and log to MLflow
    """
    logger.info("Starting explainability")
    
    # FIX: always initialize expected_cols to avoid UnboundLocalError
    expected_cols = None
    
    try:
        mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        mlflow.set_tracking_uri(mlflow_uri)
        client = MlflowClient(tracking_uri=mlflow_uri)
        model_name = os.environ.get("MODEL_REGISTRY_NAME", MODEL_REGISTRY_NAME)

        # 1) Load model (prefer production registry)
        model = None
        model_uri = None
        run_id_for_features = None
        experiment = None
        try:
            registry_uri = f"models:/{model_name}/Production"
            logger.info("Attempting to load production model from %s", registry_uri)
            model = mlflow.pyfunc.load_model(registry_uri)
            model_uri = registry_uri
        except Exception:
            logger.info("Production model not available; falling back to latest run model")
            exp_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "demo_experiment")
            experiment = mlflow.get_experiment_by_name(exp_name)
            if experiment:
                runs = client.search_runs([experiment.experiment_id], order_by=["attributes.start_time DESC"], max_results=20)
                for r in runs:
                    if r.info.status == "FINISHED":
                        model_uri = f"runs:/{r.info.run_id}/model"
                        run_id_for_features = r.info.run_id
                        try:
                            model = mlflow.pyfunc.load_model(model_uri)
                            break
                        except Exception:
                            continue

        if model is None:
            raise RuntimeError("Could not load any model for explanation")

        print(f"DEBUG: model_uri={model_uri}, run_id_for_features={run_id_for_features}")

        # If loaded from registry, try to discover run_id from the model version artifact (best-effort)
        # Fallback logic: if model_uri is runs:/... then extract run_id
        if model_uri and model_uri.startswith("runs:/"):
            try:
                run_id_for_features = model_uri.split("/")[1]
                logger.info("run_id_for_features detected from model_uri: %s", run_id_for_features)
            except Exception:
                run_id_for_features = None
                logger.warning("Could not parse run_id from model_uri: %s", model_uri)

        # Defensive check for model type
        if run_id_for_features:
            try:
                run_obj = client.get_run(run_id_for_features)
                model_type = run_obj.data.tags.get("model_type", "tabular")
                if model_type != "tabular":
                    logger.warning("Skipping SHAP explanation. Model type '%s' is not supported yet.", model_type)
                    return {"method": "skipped", "reason": f"model_type={model_type}"}
            except Exception as e:
                logger.warning("Could not verify model_type tag: %s", e)

        # Try to download training/training_features.txt from the primary run_id (if available)
        if run_id_for_features:
            remote_feat = "training/training_features.txt"
            local_tmp_dir = "/tmp/explain_training_features"
            os.makedirs(local_tmp_dir, exist_ok=True)
            try:
                logger.info("Attempting artifact download: run_id=%s path=%s", run_id_for_features, remote_feat)
                client.download_artifacts(run_id=run_id_for_features, path=remote_feat, dst_path=local_tmp_dir)
                local_file = os.path.join(local_tmp_dir, "training_features.txt")
                if os.path.exists(local_file):
                    with open(local_file, "r") as f:
                        expected_cols = [l.strip() for l in f.readlines() if l.strip()]
                    logger.info("Successfully loaded training feature list from artifact: %s", local_file)
                else:
                    logger.warning("Artifact not present at expected path for run %s: %s", run_id_for_features, local_file)
            except Exception as ex:
                logger.warning("Artifact download failed for run %s: %s", run_id_for_features, ex)
                expected_cols = None

        # If still none, try model metadata (feature_names_in_ or signature)
        if expected_cols is None:
            try:
                wrapped = getattr(model, "model", None) or getattr(model, "_model_impl", None) or getattr(model, "pyfunc", None) or None
                if wrapped is not None and hasattr(wrapped, "feature_names_in_"):
                    expected_cols = list(wrapped.feature_names_in_)
                    logger.info("Loaded expected cols from estimator.feature_names_in_: %s", expected_cols)
            except Exception:
                logger.warning("Could not load feature_names_in_ from wrapped estimator")

        if expected_cols is None:
            try:
                md = getattr(model, "metadata", None)
                if md is not None:
                    sig = None
                    try:
                        sig = md.get_input_schema()
                    except Exception:
                        sig = getattr(md, "signature", None)
                    if sig and getattr(sig, "columns", None):
                        expected_cols = [c.name for c in sig.columns]
                        logger.info("Loaded expected cols from model signature: %s", expected_cols)
            except Exception:
                logger.warning("Could not read model signature metadata")

        # 3) If still none, search other finished runs in the experiment for the artifact (best-effort)
        if expected_cols is None and experiment is not None:
            logger.info("Primary run did not yield features; searching finished runs in experiment '%s' for artifact 'training/training_features.txt'", experiment.name)
            all_runs = client.search_runs([experiment.experiment_id], order_by=["attributes.start_time DESC"], max_results=200)
            found_run = None
            for r in all_runs:
                if r.info.status != "FINISHED":
                    continue
                try:
                    test_run_id = r.info.run_id
                    local_tmp_dir = "/tmp/explain_training_features_search"
                    os.makedirs(local_tmp_dir, exist_ok=True)
                    remote_feat = "training/training_features.txt"
                    client.download_artifacts(run_id=test_run_id, path=remote_feat, dst_path=local_tmp_dir)
                    candidate = os.path.join(local_tmp_dir, "training_features.txt")
                    if os.path.exists(candidate):
                        with open(candidate, "r") as f:
                            expected_cols = [l.strip() for l in f.readlines() if l.strip()]
                        found_run = test_run_id
                        logger.info("Found training features in run %s (artifact path %s)", test_run_id, candidate)
                        break
                except Exception:
                    # ignore and keep searching
                    continue
            if found_run is None:
                logger.warning("No training_features.txt found in any finished runs for experiment '%s'", experiment.name)

        # Last resort: use validation CSV columns as expected features
        # (they should match training features due to strict validation)
        if expected_cols is None:
            logger.warning("Could not load training feature list from artifacts or model metadata; using validation CSV columns as fallback")
            try:
                sample_path = os.environ.get("EXPLAIN_SAMPLE_PATH", VALIDATION_CSV_DEFAULT)
                if os.path.exists(sample_path):
                    df_sample = pd.read_csv(sample_path)
                    label_col = os.environ.get("LABEL_COLUMN", LABEL_COLUMN_DEFAULT)
                    expected_cols = [c for c in df_sample.columns if c != label_col]
                    logger.info("Using validation CSV columns as expected features: %s", expected_cols)
            except Exception as e:
                logger.warning("Failed to use validation CSV as fallback: %s", e)

        if not expected_cols:
            raise RuntimeError("Explainability requires training feature list artifact 'training/training_features.txt', model signature/feature_names_in_, or validation CSV. None found.")

        # 4) Load explain sample (validation or provided sample)
        sample_path = os.environ.get("EXPLAIN_SAMPLE_PATH", VALIDATION_CSV_DEFAULT)
        if not os.path.exists(sample_path):
            raise FileNotFoundError(f"Explain sample data not found at {sample_path}")
        df_sample = pd.read_csv(sample_path)
        label_col = os.environ.get("LABEL_COLUMN", LABEL_COLUMN_DEFAULT)
        X = df_sample.drop(columns=[label_col]) if label_col in df_sample.columns else df_sample

        # 5) If no expected columns were discovered from artifacts/metadata/search, fallback to using validation CSV columns
        if expected_cols is None:
            logger.warning("Could not load training features from artifacts or model metadata; using validation CSV columns as fallback (explainability may fail if model expects different features).")
            expected_cols = list(X.columns)
            logger.info("Using validation CSV columns as expected features: %s", expected_cols)

        # 6) Strict set match: must have exactly same set of columns (order can differ -> we'll reorder)
        val_cols = list(X.columns)
        expected_set = set(expected_cols)
        val_set = set(val_cols)
        missing = [c for c in expected_cols if c not in val_set]
        unexpected = [c for c in val_cols if c not in expected_set]
        if missing or unexpected:
            logger.error("Explainability feature contract violation")
            logger.error("Expected features (%d): %s", len(expected_cols), expected_cols)
            logger.error("Sample features (%d): %s", len(val_cols), val_cols)
            if missing:
                logger.error("Missing required features: %s", missing)
            if unexpected:
                logger.error("Unexpected features: %s", unexpected)
            raise ValueError(f"Explainability aborted: feature set mismatch. Missing: {missing}, Unexpected: {unexpected}")

        # 7) Reorder sample to expected order (if needed)
        if val_cols != expected_cols:
            logger.info("Reordering explain sample columns to match training feature order.")
            X = X[expected_cols]

        # limit sample size
        if len(X) > EXPLAIN_SAMPLE_SIZE:
            X = X.sample(EXPLAIN_SAMPLE_SIZE, random_state=42)

        # 6) compute SHAP or permutation importances
        explain_artifact_dir = "/tmp/explain_artifacts"
        os.makedirs(explain_artifact_dir, exist_ok=True)

        if HAS_SHAP:
            logger.info("SHAP available — computing SHAP values")
            try:
                # Create SHAP explainer (wrap model.predict). Note: some SHAP versions differ in API.
                explainer = shap.Explainer(model.predict, X)
                shap_values = explainer(X)

                # Save summary plot PNG (most robust across SHAP versions)
                shap_png = os.path.join(explain_artifact_dir, "shap_summary.png")
                shap.summary_plot(shap_values, X, show=False)
                plt.savefig(shap_png, bbox_inches="tight")
                plt.close()

                # Save numeric importances as CSV
                try:
                    mean_abs = np.abs(shap_values.values).mean(axis=0)
                except Exception:
                    # fallback if shap_values has different structure
                    mean_abs = np.mean(np.abs(np.array(shap_values).reshape(len(X), -1)), axis=0)
                imp_df = pd.DataFrame({"feature": X.columns, "mean_abs_shap": mean_abs})
                shap_csv = os.path.join(explain_artifact_dir, "shap_importances.csv")
                imp_df.to_csv(shap_csv, index=False)

                # Create a small HTML that embeds the PNG (portable)
                shap_html = os.path.join(explain_artifact_dir, "shap_summary.html")
                with open(shap_html, "w") as fh:
                    fh.write(f"<html><body><h2>SHAP summary</h2><img src='shap_summary.png' /></body></html>")

                mlflow.start_run()
                mlflow.log_artifact(shap_png, artifact_path="explainability")
                mlflow.log_artifact(shap_csv, artifact_path="explainability")
                mlflow.log_artifact(shap_html, artifact_path="explainability")
                mlflow.end_run()
                return {"method": "shap", "artifacts": ["shap_summary.png", "shap_importances.csv", "shap_summary.html"]}
            except Exception:
                logger.exception("SHAP explanation failed; will try permutation importance as fallback")

        # permutation importance fallback (requires an estimator-like wrapper)
        try:
            # Permutation importance expects an estimator implementing fit(). Provide a thin wrapper.
            class EstimatorForPermImp:
                def __init__(self, pyfunc):
                    self._pyfunc = pyfunc
                def fit(self, X_, y_):
                    # sklearn requires fit to exist; we do not need to actually fit
                    return self
                def predict(self, X_):
                    preds = np.array(self._pyfunc.predict(X_))
                    return preds.squeeze()

            estimator = None
            # prefer an actual sklearn estimator if present
            wrapped_estimator = getattr(model, "model", None) or getattr(model, "_model_impl", None) or None
            if wrapped_estimator is not None and hasattr(wrapped_estimator, "fit"):
                estimator = wrapped_estimator
            else:
                estimator = EstimatorForPermImp(model)

            # Use true labels if available in the sample
            if label_col in df_sample.columns:
                y_true = df_sample[label_col].values
            else:
                raise RuntimeError("Permutation importance requires labeled sample. Provide a sample with the label column.")

            r = permutation_importance(estimator, X, y_true, n_repeats=10, random_state=42, n_jobs=1)
            imp_df = pd.DataFrame({"feature": X.columns, "importance_mean": r.importances_mean, "importance_std": r.importances_std})
            imp_csv = os.path.join(explain_artifact_dir, "perm_importances.csv")
            imp_df.to_csv(imp_csv, index=False)
            imp_png = os.path.join(explain_artifact_dir, "perm_importance.png")
            imp_df_sorted = imp_df.sort_values("importance_mean", ascending=False).head(30)
            plt.figure(figsize=(8, max(4, 0.25*len(imp_df_sorted))))
            plt.barh(imp_df_sorted["feature"], imp_df_sorted["importance_mean"])
            plt.xlabel("Importance (mean)")
            plt.tight_layout()
            plt.savefig(imp_png, bbox_inches="tight")
            plt.close()

            mlflow.start_run()
            mlflow.log_artifact(imp_csv, artifact_path="explainability")
            mlflow.log_artifact(imp_png, artifact_path="explainability")
            mlflow.end_run()

            return {"method": "permutation", "artifacts": ["perm_importances.csv", "perm_importance.png"]}
        except Exception:
            logger.exception("Permutation importance failed")
            raise

    except Exception as e:
        logger.exception("Explainability failed: %s", e)
        raise

# --------------------
# DAG wiring
# --------------------
with DAG(
    dag_id="ml_training_pipeline",
    default_args=default_args,
    start_date=datetime(2025, 11, 1),
    schedule_interval=None,
    catchup=False,
    max_active_runs=1,
    tags=["mlops"],
) as dag:

    t_ingest = PythonOperator(
        task_id="ingest_raw_data",
        python_callable=ingest_raw_data,
    )

    t_clean = PythonOperator(
        task_id="clean_data",
        python_callable=clean_data,
    )

    t_features = PythonOperator(
        task_id="feature_engineer",
        python_callable=feature_engineer,
    )

    t_split = PythonOperator(
        task_id="split_data",
        python_callable=split_data,
    )

    t_train = PythonOperator(
        task_id="train_model",
        python_callable=run_training,
    )

    t_validate = PythonOperator(
        task_id="validate_model",
        python_callable=validate_model_mlflow,
    )

    t_explain = PythonOperator(
        task_id="explain_model",
        python_callable=explain_model,
    )

    # DAG order: ingest -> clean -> features -> split -> train -> validate -> explain
    t_ingest >> t_clean >> t_features >> t_split >> t_train >> t_validate >> t_explain
