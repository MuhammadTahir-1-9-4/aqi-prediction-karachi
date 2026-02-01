import hopsworks
import os
import joblib
import pandas as pd
import numpy as np
from datetime import timedelta
from dotenv import load_dotenv

load_dotenv()

project = hopsworks.login(
    api_key_value=os.getenv("HOPSWORKS_API_KEY")
)
fs = project.get_feature_store()

model = joblib.load("models_artifacts/gradient_boosting_model.pkl")
scaler = joblib.load("models_artifacts/scaler.pkl")
feature_names = joblib.load("models_artifacts/feature_names.pkl")

fg = fs.get_feature_group(
    name="aqi_feature_group",
    version=1
)

latest_df = fg.read().sort_values("datetime").tail(1)
latest_row = latest_df.copy()

predictions = []

current_row = latest_row.copy()

for step in range(1, 4):
    future_time = current_row["datetime"].iloc[0] + timedelta(days=1)

    # update time features
    current_row["datetime"] = future_time
    current_row["hour"] = 12
    current_row["day"] = future_time.day
    current_row["month"] = future_time.month
    current_row["day_of_week"] = future_time.weekday()
    current_row["is_weekend"] = 1 if future_time.weekday() >= 5 else 0

    X = current_row[feature_names]
    X_scaled = scaler.transform(X.values)

    pred_aqi = model.predict(X_scaled)[0]

    predictions.append({
        "date": future_time.date(),
        "predicted_aqi": round(float(pred_aqi), 2)
    })

    # update rolling features (recursive)
    current_row["rolling_aqi_3h"] = pred_aqi
    current_row["rolling_aqi_6h"] = pred_aqi
    current_row["rolling_aqi_24h"] = pred_aqi
    current_row["aqi_change_rate"] = pred_aqi - latest_row["aqi"].iloc[0]

forecast_df = pd.DataFrame(predictions)
print("\n📅 3-Day AQI Forecast for Karachi")
print(forecast_df)
