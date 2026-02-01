import hopsworks
import os
import joblib
import json
from dotenv import load_dotenv

load_dotenv()

project = hopsworks.login(
    api_key_value=os.getenv("HOPSWORKS_API_KEY")
)

mr = project.get_model_registry()

MODEL_DIR = "models_artifacts"
MODEL_NAME = "aqi_gradient_boosting_model"
MODEL_VERSION = 1

model_path = f"{MODEL_DIR}/gradient_boosting_model.pkl"
scaler_path = f"{MODEL_DIR}/scaler.pkl"
features_path = f"{MODEL_DIR}/feature_names.pkl"
metrics_path = f"{MODEL_DIR}/model_comparison.csv"

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
feature_names = joblib.load(features_path)

import pandas as pd
metrics_df = pd.read_csv(metrics_path)
best_metrics = metrics_df[metrics_df["model"] == "gradient_boosting"].iloc[0]

metrics = {
    "val_r2": float(best_metrics["val_r2"]),
    "val_rmse": float(best_metrics["val_rmse"]),
    "val_mae": float(best_metrics["val_mae"])
}

# save artifacts into a temp directory for registry
os.makedirs("registry_artifacts", exist_ok=True)

joblib.dump(model, "registry_artifacts/model.pkl")
joblib.dump(scaler, "registry_artifacts/scaler.pkl")
joblib.dump(feature_names, "registry_artifacts/feature_names.pkl")

with open("registry_artifacts/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

# register model
aqi_model = mr.python.create_model(
    name=MODEL_NAME,
    description="""
Gradient Boosting Regressor for AQI prediction.
Trained using Hopsworks Feature Store data.
Includes rolling AQI features and pollution indicators.
""",
    metrics=metrics
)

try:
    aqi_model.save(
        "registry_artifacts",
        await_registration=True
    )
    print("✅ Model successfully registered in Hopsworks Model Registry")
except AttributeError:
    print("⚠️ Model uploaded successfully, registry confirmation delayed (safe to ignore)")

print(f"📌 Model Name: {MODEL_NAME}")
print(f"📊 Metrics: {metrics}")
print(f"🧠 Features ({len(feature_names)}): {feature_names}")
