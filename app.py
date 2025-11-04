# app.py
"""
FastAPI model server. Expects exported_model/rf_churn_pipe.joblib in project root.
Start with:
  uvicorn app:app --host 127.0.0.1 --port 8000
"""

import os
import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

MODEL_PATH = os.path.join("exported_model", "rf_churn_pipe.joblib")
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model not found at {MODEL_PATH}. Run `python train_and_export.py` first.")

model = joblib.load(MODEL_PATH)

app = FastAPI(title="Online Retail Churn Demo")

class RFMRequest(BaseModel):
    Recency: float
    Frequency: float
    Monetary: float
    AvgUnitPrice: float
    TotalQuantity: float

@app.get("/")
def root():
    return {"status": "ok", "model_loaded": True}

@app.post("/predict")
def predict(req: RFMRequest):
    x = np.array([[req.Recency, req.Frequency, req.Monetary, req.AvgUnitPrice, req.TotalQuantity]])
    prob = float(model.predict_proba(x)[0, 1])
    pred = int(prob >= 0.5)
    return {"churn_probability": prob, "churn_prediction": pred}
