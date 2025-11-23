# /opt/airflow/mlops-pipeline/src/data_preprocessing/preprocess_router.py
import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class PreprocessRouter:
    """
    Simple preprocessing router:
    - clean_all: reads all raw CSVs, drops duplicates and simple NA handling, saves to clean/
    - feature_engineer_all: placeholder to create features (for now: copy clean -> features)
    - train_val_split: concatenate features and split into train/validation CSVs
    """
    def __init__(self, raw_dir: str, clean_dir: str, feature_dir: str):
        self.raw_dir = raw_dir
        self.clean_dir = clean_dir
        self.feature_dir = feature_dir
        os.makedirs(self.clean_dir, exist_ok=True)
        os.makedirs(self.feature_dir, exist_ok=True)

    def clean_all(self):
        saved = []
        csvs = glob.glob(os.path.join(self.raw_dir, "*.csv"))
        for path in csvs:
            df = pd.read_csv(path)
            # simple cleaning rules:
            df = df.drop_duplicates().reset_index(drop=True)
            # fill numeric nans with median, categorical with mode
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else "")
            fname = Path(path).name
            out_path = os.path.join(self.clean_dir, fname)
            df.to_csv(out_path, index=False)
            saved.append(out_path)
            logger.info("Cleaned %s -> %s", path, out_path)
        return saved

    def feature_engineer_all(self):
        # placeholder: copy clean files to features dir, real logic should go here
        saved = []
        csvs = glob.glob(os.path.join(self.clean_dir, "*.csv"))
        for path in csvs:
            df = pd.read_csv(path)
            # Example feature engineering: create interaction if numeric columns exist
            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if len(num_cols) >= 2:
                a, b = num_cols[:2]
                df[f"{a}_x_{b}"] = df[a] * df[b]
            out_path = os.path.join(self.feature_dir, Path(path).name)
            df.to_csv(out_path, index=False)
            saved.append(out_path)
            logger.info("Features written %s", out_path)
        return saved

    def train_val_split(self, train_out_path: str, val_out_path: str, label_col: str = "target", test_size: float = 0.2, random_state: int = 42):
        # read all feature CSVs and concatenate
        csvs = glob.glob(os.path.join(self.feature_dir, "*.csv"))
        if not csvs:
            raise FileNotFoundError(f"No feature files found in {self.feature_dir}")
        dfs = [pd.read_csv(p) for p in csvs]
        df = pd.concat(dfs, ignore_index=True)
        if label_col not in df.columns:
            # if label not found, try to infer last column as label (warning)
            logger.warning("Label column '%s' not found in features; attempting to use last column as label", label_col)
            label_col = df.columns[-1]
        train, val = train_test_split(df, test_size=test_size, random_state=random_state)
        os.makedirs(os.path.dirname(train_out_path), exist_ok=True)
        train.to_csv(train_out_path, index=False)
        val.to_csv(val_out_path, index=False)
        logger.info("Train/val saved to %s and %s", train_out_path, val_out_path)
        return train_out_path, val_out_path


def route_preprocessing(raw_dir: str,
                        clean_dir: str = None,
                        feature_dir: str = None,
                        action: str = "all",
                        train_out_path: str = None,
                        val_out_path: str = None,
                        label_col: str = "target",
                        test_size: float = 0.2,
                        random_state: int = 42):
    """
    Backwards-compatible helper expected by the DAG.
    action:
      - "all": run clean_all -> feature_engineer_all -> train_val_split (if train/val paths provided or default)
      - "clean": run clean_all()
      - "features": run feature_engineer_all()
      - "split": run train_val_split()
    Returns the outputs of the steps performed.
    """
    if clean_dir is None:
        # infer from raw_dir parent
        base = os.path.abspath(os.path.join(raw_dir, ".."))
        clean_dir = os.path.join(base, "clean")
    if feature_dir is None:
        base = os.path.abspath(os.path.join(raw_dir, ".."))
        feature_dir = os.path.join(base, "features")

    pr = PreprocessRouter(raw_dir=raw_dir, clean_dir=clean_dir, feature_dir=feature_dir)

    if action == "clean":
        return pr.clean_all()

    if action == "features":
        return pr.feature_engineer_all()

    if action == "split":
        if train_out_path is None or val_out_path is None:
            # default output into parent data directory if not provided
            base = os.path.abspath(os.path.join(feature_dir, ".."))
            train_out_path = train_out_path or os.path.join(base, "train.csv")
            val_out_path = val_out_path or os.path.join(base, "validation.csv")
        return pr.train_val_split(train_out_path, val_out_path, label_col=label_col, test_size=test_size, random_state=random_state)

    # action == "all"
    result = {}
    result["cleaned"] = pr.clean_all()
    result["features"] = pr.feature_engineer_all()
    # do split only if there are feature files
    try:
        if train_out_path is None or val_out_path is None:
            base = os.path.abspath(os.path.join(feature_dir, ".."))
            train_out_path = train_out_path or os.path.join(base, "train.csv")
            val_out_path = val_out_path or os.path.join(base, "validation.csv")
        train_path, val_path = pr.train_val_split(train_out_path, val_out_path, label_col=label_col, test_size=test_size, random_state=random_state)
        result["train"] = train_path
        result["validation"] = val_path
    except Exception as e:
        # don't fail the whole wrapper if split can't run; return what we have and the error
        result["split_error"] = str(e)
    return result
