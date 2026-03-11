from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pickle
import pandas as pd

# Load saved models
with open('bed_model.pkl', 'rb') as f:
    bed_model = pickle.load(f)

with open('alert_model.pkl', 'rb') as f:
    alert_model = pickle.load(f)

app = FastAPI()

# Allow dashboard to talk to this API
app.add_middleware(CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"])

@app.get("/")
def home():
    return {"message": "AI Bed Tracking System is running"}

@app.get("/predict")
def predict(hour: int, day: int, is_weekend: int, is_festival: int):

    # Prepare input
    input_data = pd.DataFrame([{
        'hour': hour,
        'day_of_week': day,
        'is_weekend': is_weekend,
        'is_festival': is_festival
    }])

    # Predict occupied beds
    occupied_pred = int(bed_model.predict(input_data)[0])
    available_pred = 247 - occupied_pred

    # Predict alert
    alert_prob = alert_model.predict_proba(input_data)[0][1]
    alert_status = "CRITICAL" if alert_prob > 0.7 else "NORMAL"

    return {
        "hour": hour,
        "occupied": occupied_pred,
        "available": available_pred,
        "occupancy_pct": round((occupied_pred / 247) * 100, 1),
        "alert": alert_status,
        "alert_probability": round(alert_prob * 100, 1)
    }

@app.get("/forecast")
def forecast(day: int, is_weekend: int, is_festival: int):

    results = []

    for hour in range(6, 24):
        input_data = pd.DataFrame([{
            'hour': hour,
            'day_of_week': day,
            'is_weekend': is_weekend,
            'is_festival': is_festival
        }])

        occupied_pred = int(bed_model.predict(input_data)[0])
        alert_prob = alert_model.predict_proba(input_data)[0][1]

        results.append({
            "hour": hour,
            "occupied": occupied_pred,
            "available": 247 - occupied_pred,
            "alert_probability": round(alert_prob * 100, 1)
        })

    return {"forecast": results} 