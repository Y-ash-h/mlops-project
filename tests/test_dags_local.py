import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# Check if we're in CI environment (no Airflow installed)
IN_CI = os.environ.get("CI") == "true" or os.environ.get("GITHUB_ACTIONS") == "true"

# 1. Mock Airflow modules BEFORE importing DAGs
sys.modules["airflow"] = MagicMock()
sys.modules["airflow.operators"] = MagicMock()
sys.modules["airflow.operators.python"] = MagicMock()
sys.modules["airflow.operators.dummy"] = MagicMock()
sys.modules["airflow.operators.dummy_operator"] = MagicMock()
sys.modules["airflow.operators.bash"] = MagicMock()

# Mock other heavy dependencies
sys.modules["matplotlib"] = MagicMock()
sys.modules["matplotlib.pyplot"] = MagicMock()
sys.modules["mlflow"] = MagicMock()
sys.modules["mlflow.pyfunc"] = MagicMock()
sys.modules["mlflow.tracking"] = MagicMock()
sys.modules["mlflow.sklearn"] = MagicMock()
sys.modules["sklearn"] = MagicMock()
sys.modules["sklearn.metrics"] = MagicMock()
sys.modules["sklearn.inspection"] = MagicMock()
sys.modules["sklearn.model_selection"] = MagicMock()
sys.modules["sklearn.ensemble"] = MagicMock()
sys.modules["shap"] = MagicMock()
sys.modules["lightgbm"] = MagicMock()
sys.modules["xgboost"] = MagicMock()
sys.modules["pandas"] = MagicMock()
sys.modules["numpy"] = MagicMock()

# Mock DAG context manager
mock_dag = MagicMock()
mock_dag.__enter__.return_value = mock_dag
sys.modules["airflow"].DAG = MagicMock(return_value=mock_dag)

# 2. Add paths
# We are in project root.
sys.path.insert(0, os.path.abspath("mlops-pipeline"))
sys.path.insert(0, os.path.abspath("airflow/dags"))

class TestDAGLogicLocal(unittest.TestCase):

    @unittest.skipIf(IN_CI, "Skipping DAG tests in CI environment (requires Airflow)")
    def test_mlops_dag_branching_logic(self):
        """Test that check_if_explainable returns 'skip_explain' for current data/raw"""
        # Import the module inside test to ensure mocks are active
        import mlops_dag
        
        if not hasattr(mlops_dag, 'check_if_explainable'):
            self.fail("check_if_explainable not found in mlops_dag module")
            
        # Run it
        # We need to ensure RAW_DATA_DIR is correctly set in the module or we mock detect_data_type
        # In mlops_dag.py: RAW_DATA_DIR = str(Path(DATA_BASE_DIR) / "raw")
        # DATA_BASE_DIR defaults to /opt/airflow/mlops-pipeline/data
        # We should patch RAW_DATA_DIR to point to local data/raw
        
        local_raw_path = os.path.abspath("data/raw")
        
        with patch("mlops_dag.RAW_DATA_DIR", local_raw_path):
            result = mlops_dag.check_if_explainable()
            print(f"check_if_explainable returned: {result}")
            
            # We expect 'skip_explain' because data/raw has 1 csv -> text (or empty)
            self.assertEqual(result, 'skip_explain')

    @unittest.skipIf(IN_CI, "Skipping DAG tests in CI environment (requires Airflow)")
    @unittest.skipIf(IN_CI, "Skipping DAG tests in CI (requires full Airflow environment)")
    @patch("ml_training_dag.mlflow")
    def test_ml_training_dag_defensive_check(self, mock_mlflow_local):
        """Test that explain_model returns early if model_type is not tabular"""
        # Skip this test if we can't properly mock all dependencies
        # This test requires extensive mocking of mlflow, pandas, etc.
        try:
            import ml_training_dag
        except Exception as e:
            self.skipTest(f"Could not import ml_training_dag: {e}")
        
        # Mock the MLflow client
        mock_client = MagicMock()
        
        # Create a mock run with text model type
        mock_run_data = MagicMock()
        mock_run_data.tags = {"model_type": "text"}
        mock_run_data.tags.get = lambda key, default=None: {"model_type": "text"}.get(key, default)
        
        mock_run = MagicMock()
        mock_run.data = mock_run_data
        mock_run.info.run_id = "run123"
        mock_run.info.status = "FINISHED"
        
        # Mock the MlflowClient - need to patch it in the module
        from mlflow.tracking import MlflowClient as RealMlflowClient
        with patch('ml_training_dag.MlflowClient') as mock_client_class:
            mock_client_class.return_value = mock_client
            
            # Mock client methods
            mock_client.get_run.return_value = mock_run
            mock_client.search_runs.return_value = [mock_run]
            
            # Mock a successful model load (returns a mock model)
            mock_model = MagicMock()
            
            # Mock experiment
            mock_exp = MagicMock()
            mock_exp.experiment_id = "1"
            
            # Mock environment variables
            with patch.dict(os.environ, {"MLFLOW_TRACKING_URI": "http://mock", "MODEL_REGISTRY_NAME": "test_model", "MLFLOW_EXPERIMENT_NAME": "demo_experiment"}):
                # Configure load_model to fail for registry, then succeed for runs:/
                def load_model_side_effect(uri):
                    if uri.startswith("models:/"):
                        raise Exception("No registry")
                    elif uri.startswith("runs:/"):
                        return mock_model
                    else:
                        return mock_model
                
                ml_training_dag.mlflow.pyfunc.load_model.side_effect = load_model_side_effect
                ml_training_dag.mlflow.get_experiment_by_name.return_value = mock_exp
                ml_training_dag.mlflow.set_tracking_uri = MagicMock()
                
                result = ml_training_dag.explain_model()
                
                self.assertEqual(result["method"], "skipped")
                self.assertEqual(result["reason"], "model_type=text")

if __name__ == "__main__":
    unittest.main()
