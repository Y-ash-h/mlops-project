import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# 1. Mock Airflow modules BEFORE importing DAGs
sys.modules["airflow"] = MagicMock()
sys.modules["airflow.operators"] = MagicMock()
sys.modules["airflow.operators.python"] = MagicMock()
sys.modules["airflow.operators.dummy"] = MagicMock()
sys.modules["airflow.operators.dummy_operator"] = MagicMock()

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

# Mock DAG context manager
mock_dag = MagicMock()
mock_dag.__enter__.return_value = mock_dag
sys.modules["airflow"].DAG = MagicMock(return_value=mock_dag)

# 2. Add paths
# We are in project root.
sys.path.insert(0, os.path.abspath("mlops-pipeline"))
sys.path.insert(0, os.path.abspath("airflow/dags"))

class TestDAGLogicLocal(unittest.TestCase):

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

    @patch("ml_training_dag.mlflow")
    def test_ml_training_dag_defensive_check(self, mock_mlflow_local):
        """Test that explain_model returns early if model_type is not tabular"""
        import ml_training_dag
        import importlib
        importlib.reload(ml_training_dag)
        
        # Now configure the mocks on the reloaded module
        
        # Mock the MLflow client
        mock_client = MagicMock()
        # The module uses `mlflow.tracking.MlflowClient`
        # Since we mocked sys.modules["mlflow.tracking"], we should configure that.
        import mlflow.tracking
        mlflow.tracking.MlflowClient.return_value = mock_client
        
        mock_run = MagicMock()
        mock_run.data.tags.get.return_value = "text" # Simulate text model
        mock_run.info.run_id = "run123"
        mock_run.info.status = "FINISHED"
        
        mock_client.get_run.return_value = mock_run
        mock_client.search_runs.return_value = [mock_run]
        
        # Mock environment variables
        with patch.dict(os.environ, {"MLFLOW_TRACKING_URI": "http://mock", "MODEL_REGISTRY_NAME": "test_model"}):
            
            # Configure load_model to fail first, then succeed
            # We access it via the module to be sure
            ml_training_dag.mlflow.pyfunc.load_model.side_effect = [Exception("No registry"), MagicMock()]
            
            # Mock experiment
            mock_exp = MagicMock()
            mock_exp.experiment_id = "1"
            ml_training_dag.mlflow.get_experiment_by_name.return_value = mock_exp
            
            result = ml_training_dag.explain_model()
            
            self.assertEqual(result["method"], "skipped")
            self.assertEqual(result["reason"], "model_type=text")

if __name__ == "__main__":
    unittest.main()
