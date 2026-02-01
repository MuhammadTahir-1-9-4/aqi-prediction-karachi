import hopsworks
import os
import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()

project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
fs = project.get_feature_store()
mr = project.get_model_registry()

model = mr.get_model(
    name="aqi_gradient_boosting_model",
    version=1
)
model_dir = model.download()
gb_model = joblib.load(os.path.join(model_dir, "model.pkl"))

scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
feature_names = joblib.load(os.path.join(model_dir, "feature_names.pkl"))

fv = fs.get_feature_view(name="aqi_feature_view", version=1)
X, y = fv.get_training_data(training_dataset_version=1)

X = X[feature_names]

X_scaled = scaler.transform(X)

explainer = shap.Explainer(gb_model)
shap_values = explainer(X_scaled)

os.makedirs("shap_outputs", exist_ok=True)

plt.figure()
shap.summary_plot(shap_values, X, show=False)
plt.savefig("shap_outputs/shap_summary.png", bbox_inches="tight")
plt.close()

print("✅ SHAP explainability generated successfully")
print("📊 Saved: shap_outputs/shap_summary.png")
