# flask API for AQI Predictions
from flask import Flask, request, jsonify
import hopsworks
import joblib
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
import logging

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

model = None
scaler = None
feature_names = None
fs = None
project = None

def load_model_components():
    global model, scaler, feature_names, fs, project

    try:
        project = hopsworks.login(
            api_key_value=os.getenv("HOPSWORKS_API_KEY"),
            project="aqi_predicton"
        )

        mr = project.get_model_registry()
        fs = project.get_feature_store()

        aqi_model = mr.get_model(
            name="aqi_gradient_boosting_model"
        )

        model_dir = aqi_model.download()
        model = joblib.load(os.path.join(model_dir, "model.pkl"))
        scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
        feature_names = joblib.load(os.path.join(model_dir, "feature_names.pkl"))

        logger.info("Model components loaded successfully")
        return True

    except Exception as e:
        logger.error(f"Error loading model components: {str(e)}")
        return False

def get_latest_features():
    try:
        fv = fs.get_feature_view(
            name="aqi_feature_view",
            version=1
        )

        end_time = datetime.now()
        start_time = end_time - timedelta(hours=24)
        
        # get batch data last 24 hours
        batch_data = fv.get_batch_data(
            start_time=start_time,
            end_time=end_time
        )

        if isinstance(batch_data, tuple) and len(batch_data) == 2:
            X, y = batch_data
        else:
            X = batch_data

        if X is not None and not X.empty:
            available_features = [f for f in feature_names if f in X.columns]
            X_filtered = X[available_features].copy()

            latest_sample = X_filtered.iloc[-1:].copy()

            for feature in feature_names:
                if feature not in latest_sample.columns:
                    latest_sample[feature] = 0  # Default value

            return latest_sample[feature_names]

    except Exception as e:
        logger.warning(f"Could not fetch live data: {str(e)}")

    return None

def create_sample_features():
    now = datetime.now()

    sample_data = pd.DataFrame({
        'co': [np.random.uniform(200, 400)],
        'no': [np.random.randint(0, 10)],
        'no2': [np.random.uniform(20, 60)],
        'o3': [np.random.uniform(20, 80)],
        'so2': [np.random.uniform(5, 30)],
        'pm2_5': [np.random.uniform(40, 120)],
        'pm10': [np.random.uniform(80, 200)],
        'nh3': [np.random.uniform(5, 25)],
        'hour': [now.hour],
        'day': [now.day],
        'day_of_week': [now.weekday()],
        'month': [now.month],
        'is_weekend': [1 if now.weekday() >= 5 else 0],
        'aqi_change_rate': [np.random.uniform(-0.3, 0.3)],
        'rolling_aqi_3h': [np.random.uniform(2.0, 4.0)],
        'rolling_aqi_6h': [np.random.uniform(2.0, 4.0)],
        'rolling_aqi_24h': [np.random.uniform(2.0, 4.0)]
    })

    for feature in feature_names:
        if feature not in sample_data.columns:
            sample_data[feature] = 0

    return sample_data[feature_names]

def predict_aqi(features_df):
    try:
        features_scaled = scaler.transform(features_df.values)
        prediction = model.predict(features_scaled)[0]

        confidence = None
        if hasattr(model, 'predict_proba'):
            # For classification-like interpretation
            confidence = 0.8  # placeholder
        else:
            # For regression, use a simple confidence measure
            confidence = 0.85  # placeholder

        return float(prediction), confidence

    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        return None, None

def get_aqi_status(aqi):
    if aqi <= 1:
        return {
            "status": "Good",
            "color": "green",
            "description": "Air quality is satisfactory, and air pollution poses little or no risk.",
            "recommendations": ["Enjoy outdoor activities", "No special precautions needed"]
        }
    elif aqi <= 2:
        return {
            "status": "Moderate",
            "color": "yellow",
            "description": "Air quality is acceptable. However, there may be a risk for some people.",
            "recommendations": ["Sensitive individuals should consider limiting prolonged outdoor exertion"]
        }
    elif aqi <= 3:
        return {
            "status": "Unhealthy for Sensitive Groups",
            "color": "orange",
            "description": "Members of sensitive groups may experience health effects.",
            "recommendations": [
                "Children and the elderly should limit prolonged outdoor exertion",
                "People with respiratory disease should avoid prolonged outdoor exertion"
            ]
        }
    elif aqi <= 4:
        return {
            "status": "Unhealthy",
            "color": "red",
            "description": "Everyone may begin to experience health effects.",
            "recommendations": [
                "Avoid prolonged outdoor exertion",
                "Consider rescheduling outdoor activities",
                "People with respiratory disease should avoid outdoor exertion"
            ]
        }
    else:
        return {
            "status": "Hazardous",
            "color": "darkred",
            "description": "Emergency conditions. The entire population is more likely to be affected.",
            "recommendations": [
                "Avoid all outdoor activities",
                "Stay indoors with windows closed",
                "Use air purifiers if available",
                "Wear N95 masks if going outside is necessary"
            ]
        }

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model is not None
    })

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    try:
        features_df = get_latest_features()

        if features_df is None:
            logger.warning("Using sample data for prediction")
            features_df = create_sample_features()

        prediction, confidence = predict_aqi(features_df)

        if prediction is None:
            return jsonify({
                "error": "Prediction failed",
                "timestamp": datetime.now().isoformat()
            }), 500

        aqi_info = get_aqi_status(prediction)

        response = {
            "prediction": round(prediction, 2),
            "confidence": confidence,
            "status": aqi_info["status"],
            "color": aqi_info["color"],
            "description": aqi_info["description"],
            "recommendations": aqi_info["recommendations"],
            "timestamp": datetime.now().isoformat(),
            "location": "Karachi, Pakistan",
            "data_source": "live" if get_latest_features() is not None else "sample"
        }

        return jsonify(response)

    except Exception as e:
        logger.error(f"Error in predict endpoint: {str(e)}")
        return jsonify({
            "error": "Internal server error",
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/forecast', methods=['GET'])
def forecast():
    try:
        current_features = get_latest_features()
        if current_features is None:
            current_features = create_sample_features()

        forecast_data = []
        current_row = current_features.copy()

        for day in range(1, 4):
            future_date = datetime.now() + timedelta(days=day)
            current_row_copy = current_row.copy()

            current_row_copy['hour'] = 12  # Noon
            current_row_copy['day'] = future_date.day
            current_row_copy['day_of_week'] = future_date.weekday()
            current_row_copy['month'] = future_date.month
            current_row_copy['is_weekend'] = 1 if future_date.weekday() >= 5 else 0

            prediction, confidence = predict_aqi(current_row_copy)

            if prediction is not None:
                aqi_info = get_aqi_status(prediction)

                forecast_data.append({
                    "date": future_date.strftime("%Y-%m-%d"),
                    "day": future_date.strftime("%A"),
                    "prediction": round(prediction, 2),
                    "status": aqi_info["status"],
                    "color": aqi_info["color"],
                    "description": aqi_info["description"]
                })

                current_row['rolling_aqi_3h'] = prediction
                current_row['rolling_aqi_6h'] = prediction
                current_row['rolling_aqi_24h'] = prediction
                current_row['aqi_change_rate'] = prediction - (forecast_data[-1]['prediction'] if len(forecast_data) > 1 else prediction)

        return jsonify({
            "forecast": forecast_data,
            "timestamp": datetime.now().isoformat(),
            "location": "Karachi, Pakistan"
        })

    except Exception as e:
        logger.error(f"Error in forecast endpoint: {str(e)}")
        return jsonify({
            "error": "Forecast failed",
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/features', methods=['GET'])
def get_features():
    try:
        features_df = get_latest_features()

        if features_df is not None:
            features_dict = features_df.iloc[0].to_dict()
            return jsonify({
                "features": features_dict,
                "timestamp": datetime.now().isoformat()
            })
        else:
            return jsonify({
                "error": "No feature data available",
                "timestamp": datetime.now().isoformat()
            }), 404

    except Exception as e:
        logger.error(f"Error in features endpoint: {str(e)}")
        return jsonify({
            "error": "Internal server error",
            "timestamp": datetime.now().isoformat()
        }), 500

if __name__ == '__main__':
    if load_model_components():
        print("🚀 Starting AQI Prediction API...")
        print("📊 Model loaded successfully")
        print("🌐 API endpoints:")
        print("   GET /health - Health check")
        print("   GET /predict - Current AQI prediction")
        print("   GET /forecast - 3-day forecast")
        print("   GET /features - Current feature values")
        print("🔗 API running on http://localhost:5000")

        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        print("❌ Failed to load model. Exiting...")
        exit(1)