import pandas as pd


class DataValidation:
    def __init__(self):
        pass

    def validate(self, df: pd.DataFrame):
        if df is None or df.empty:
            raise ValueError("[DataValidation] ERROR: Dataframe is empty.")
        missing = df.isnull().sum().sum()
        if missing > 0:
            print(f"[DataValidation] Warning: Dataset contains {missing} missing values.")
        print("[DataValidation] Validation complete.")
        return df

    def generate_evidently_report(self, df: pd.DataFrame, output_path: str, report_type: str = "data_quality"):
        """Generates an Evidently HTML report.
        
        report_type: "data_quality" or "data_drift"
        """
        # Defer imports to function execution time (lazy loading)
        try:
            from evidently.test_suite import TestSuite
            from evidently.test_preset import DataQualityTestPreset, DataDriftTestPreset
            
            if report_type == "data_drift":
                suite = TestSuite(tests=[DataDriftTestPreset()])
            else:
                suite = TestSuite(tests=[DataQualityTestPreset()])
            
            suite.run(current_data=df)
            suite.save_html(output_path)
            print(f"[DataValidation] Evidently TestSuite report saved to {output_path}")
            
        except ImportError as e:
            print(f"[DataValidation] Warning: Evidently not available ({e}). Skipping report generation.")
        except Exception as e:
            print(f"[DataValidation] Error generating Evidently report: {e}")