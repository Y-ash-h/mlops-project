import sys
import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

# Initialize FastAPI
app = FastAPI(title="MLOps Model Serving", version="1.0")

# Path to the model (Shared Volume Path)
MODEL_PATH = "/opt/airflow/mlops-pipeline/data/model/model.pkl"
model = None

@app.on_event("startup")
def load_model():
    global model
    print(f"🔄 Loading model from {MODEL_PATH}...")
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            print("✅ Model loaded successfully.")
        else:
            print(f"⚠️ Warning: Model not found at {MODEL_PATH}. Waiting for pipeline to run.")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")

# Define Input Schema
class PredictionRequest(BaseModel):
    features: List[Dict[str, Any]]
    # Example: [{"feature1": 10, "feature2": 20}, ...]

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.post("/predict")
def predict(request: PredictionRequest):
    if not model:
        raise HTTPException(status_code=503, detail="Model not yet trained or loaded")
    
    try:
        # Convert input JSON to DataFrame
        data = pd.DataFrame(request.features)
        
        # Ensure feature engineering happens if needed (Optional: Add logic here)
        if "feature1" in data.columns and "feature2" in data.columns:
             if "feature1_x_feature2" not in data.columns:
                 data["feature1_x_feature2"] = data["feature1"] * data["feature2"]

        # Make Prediction
        predictions = model.predict(data)
        return {"predictions": predictions.tolist()}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

