import pandas as pd
from config.config import config

class DataPreprocessing:
    def __init__(self):
        self.output_path = config.processed_data_path

    def transform(self, df: pd.DataFrame):
        print("[DataPreprocessing] Starting preprocessing...")

        # Fill missing values (simple example)
        df = df.fillna(method='ffill').fillna(method='bfill')

        # Encode categorical columns
        cat_cols = df.select_dtypes(include=['object']).columns
        if len(cat_cols) > 0:
            print(f"[DataPreprocessing] Encoding categorical columns: {list(cat_cols)}")
            df = pd.get_dummies(df, drop_first=True)

        # Save processed data
        df.to_csv(self.output_path, index=False)
        print(f"[DataPreprocessing] Processed data saved to {self.output_path}")

        return df
