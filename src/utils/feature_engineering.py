import pandas as pd
import numpy as np
import uuid
from datetime import datetime

def compute_aqi_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    
    df = df.sort_values("datetime").copy()
    # time features
    df["hour"] = df["datetime"].dt.hour
    df["day"] = df["datetime"].dt.day
    df["day_of_week"] = df["datetime"].dt.weekday
    df["month"] = df["datetime"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    # derived features
    df["aqi_change_rate"] = df["aqi"].diff().fillna(0)
    # rolling averages
    df["rolling_aqi_3h"] = df["aqi"].rolling(window=3, min_periods=1).mean()
    df["rolling_aqi_6h"] = df["aqi"].rolling(window=6, min_periods=1).mean()
    df["rolling_aqi_24h"] = df["aqi"].rolling(window=24, min_periods=1).mean()
    
    # generate unique event_id
    df["event_id"] = [str(uuid.uuid4()) for _ in range(len(df))]
    
    df["aqi"] = df["aqi"].astype("float64")
    if "no" in df.columns:
        df["no"] = df["no"].astype("float64")
    
    float_cols = [
        "co", "no2", "o3", "so2", "pm2_5", "pm10", "nh3",
        "aqi_change_rate", "rolling_aqi_3h", "rolling_aqi_6h", "rolling_aqi_24h"
    ]
    
    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype("float64")
    
    return df

def validate_aqi_data(df: pd.DataFrame) -> bool:
    if df.empty:
        return False
    
    required_columns = ['datetime', 'aqi', 'co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']
    
    for col in required_columns:
        if col not in df.columns:
            return False
    
    return True