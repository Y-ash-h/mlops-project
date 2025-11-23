"""FastAPI serving application for MLflow models."""
import os
from typing import Dict, List, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd


app = FastAPI(title="MLOps Model Serving", version="1.0.0")

# Lazy-loaded model
_model = None
_model_name = os.getenv("MODEL_NAME", "mlops_model")
_model_stage = os.getenv("MODEL_STAGE", "Production")
_mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5050")


class PredictRequest(BaseModel):
    """Request schema for predictions."""
    instances: List[Dict[str, Any]]


def _load_model():
    """Lazy-load the model from MLflow registry."""
    global _model
    if _model is None:
        mlflow.set_tracking_uri(_mlflow_uri)
        model_uri = f"models:/{_model_name}/{_model_stage}"
        print(f"[INFO] Loading model from {model_uri}")
        _model = mlflow.pyfunc.load_model(model_uri)
        print(f"[INFO] Model loaded successfully")
    return _model


@app.get("/health")
def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_name": _model_name,
        "model_stage": _model_stage,
        "mlflow_uri": _mlflow_uri,
    }


@app.post("/predict")
def predict(request: PredictRequest):
    """Prediction endpoint.
    
    Args:
        request: JSON payload with instances list
        
    Returns:
        Dict with predictions
    """
    try:
        model = _load_model()
        df = pd.DataFrame(request.instances)
        predictions = model.predict(df)
        
        # Convert predictions to list (handles numpy arrays)
        if hasattr(predictions, "tolist"):
            predictions = predictions.tolist()
        
        return {
            "predictions": predictions,
            "model_name": _model_name,
            "model_stage": _model_stage,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
