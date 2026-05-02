from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(
    title="Smart Irrigation API ",
    description="Prédit si l'irrigation est nécessaire",
    version="1.0.0"
)

# Charger modèle + scaler
model    = joblib.load("models/best_model.pkl")
scaler   = joblib.load("models/scaler.pkl")
features = joblib.load("models/features.pkl")

class SensorData(BaseModel):
    moisture0: float
    moisture1: float
    moisture2: float
    moisture3: float
    moisture4: float
    hour: int

@app.get("/")
def home():
    return {
        "message": "Smart Irrigation API ",
        "status": "running ",
        "model": "RandomForest",
        "features": features
    }

@app.post("/predict")
def predict(data: SensorData):
    X = np.array([[
        data.moisture0,
        data.moisture1,
        data.moisture2,
        data.moisture3,
        data.moisture4,
        data.hour
    ]])

    prediction  = model.predict(X)[0]
    probability = model.predict_proba(X)[0]

    return {
        "irrigation_needed": bool(prediction),
        "recommendation": "💧 Irrigate now!" if prediction == 1
                          else " No irrigation needed",
        "confidence": f"{max(probability):.1%}",
        "moisture_average": float(np.mean(X[0][:5])),
        "sensor_values": {
            "moisture0": data.moisture0,
            "moisture1": data.moisture1,
            "moisture2": data.moisture2,
            "moisture3": data.moisture3,
            "moisture4": data.moisture4,
            "hour": data.hour
        }
    }

@app.get("/health")
def health():
    return {"status": "healthy ", "model_loaded": True}