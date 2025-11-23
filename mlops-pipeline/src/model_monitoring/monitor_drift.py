import argparse
import os

import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report


def load_dataset(path: str) -> pd.DataFrame:
    """Read a CSV dataset, ensuring the file exists."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Evidently data drift report")
    parser.add_argument("--reference", required=True, help="Path to reference data (e.g. train.csv)")
    parser.add_argument("--current", required=True, help="Path to current data (e.g. validation.csv)")
    parser.add_argument("--output", required=True, help="Path to save HTML report")
    parser.add_argument("--target-column", required=True, help="Target column to drop before drift analysis")
    args = parser.parse_args()

    print(f"[INFO] Loading datasets:\n  reference={args.reference}\n  current={args.current}")
    reference_df = load_dataset(args.reference)
    current_df = load_dataset(args.current)

    # Drop target column if present so we only compare features
    for df_name, df in ("reference", reference_df), ("current", current_df):
        if args.target_column in df.columns:
            df.drop(columns=[args.target_column], inplace=True)
            print(f"[INFO] Removed target column '{args.target_column}' from {df_name} dataset")

    print("[INFO] Running Evidently DataDriftPreset ...")
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_df, current_data=current_df)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    report.save_html(args.output)
    print(f"[INFO] Drift report saved to {args.output}")


if __name__ == "__main__":
    main()
