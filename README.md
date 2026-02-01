# 🌍 AQI Prediction System - Karachi

A comprehensive, serverless Air Quality Index (AQI) prediction system for Karachi, Pakistan. This project implements an end-to-end machine learning pipeline with automated data collection, feature engineering, model training, and real-time predictions through a web dashboard.

## 🚀 Features

### ✅ Implemented Features
- **Real-time Data Ingestion**: Hourly AQI data collection from OpenWeather API
- **Feature Engineering**: Time-based features, rolling averages, and pollutant indicators
- **Feature Store**: Hopsworks integration for scalable feature management
- **Model Training**: Multiple ML algorithms with automated model selection
- **Model Registry**: Versioned model storage and deployment
- **Web Dashboard**: Interactive Streamlit app with real-time predictions
- **3-Day Forecasting**: ML-based AQI predictions for next 3 days
- **SHAP Explanations**: Model interpretability and feature importance
- **Health Alerts**: Automated notifications for hazardous AQI levels
- **REST API**: Flask-based API for programmatic access
- **CI/CD Pipeline**: GitHub Actions for automated deployment
- **Data Validation**: Great Expectations for data quality monitoring

### 🔄 Automated Pipelines
- **Hourly Feature Pipeline**: Data ingestion every hour
- **Daily Training Pipeline**: Model retraining and updates
- **Historical Backfill**: Bulk data ingestion for initial setup

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  OpenWeather    │───▶│   Feature Store  │───▶│   ML Models     │
│     API         │    │   (Hopsworks)   │    │   (Registry)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit     │    │     Flask       │    │   GitHub        │
│   Dashboard     │    │     API         │    │   Actions       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📋 Prerequisites

- Python 3.9+
- Hopsworks account (free tier available)
- OpenWeather API key (free tier available)
- GitHub account for CI/CD

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd aqi_prediction_system
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   # source venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   Create a `.env` file:
   ```env
   # Required
   OPENWEATHER_API_KEY=your_openweather_api_key
   HOPSWORKS_API_KEY=your_hopsworks_api_key
   LAT=24.8607
   LON=67.0011

   # Optional - for alerts
   ALERT_EMAIL=your_email@gmail.com
   ALERT_EMAIL_PASSWORD=your_app_password
   ALERT_RECIPIENTS=email1@example.com,email2@example.com
   SLACK_WEBHOOK=https://hooks.slack.com/...
   TELEGRAM_BOT_TOKEN=your_bot_token
   TELEGRAM_CHAT_IDS=chat_id1,chat_id2
   ```

## 🚀 Quick Start

### 1. Initial Setup

```bash
# Create feature store infrastructure
python -m src.feature_store.create_aqi_feature_group
python -m src.feature_store.create_feature_view

# Backfill historical data (optional)
python -m src.feature_store.backfill_aqi_features --days 30

# Train initial models
python -m src.models.train_models

# Register best model
python -m src.models.register_model
```

### 2. Start Data Ingestion

```bash
# Start hourly ingestion
python -m src.feature_store.hourly_ingestion --continuous
```

### 3. Launch Dashboard

```bash
# Start Streamlit app
streamlit run app/app.py
```

### 4. Start API Server

```bash
# Start Flask API
python api/app.py
```

## 📊 API Endpoints

### REST API (Flask)
- `GET /health` - Health check
- `GET /predict` - Current AQI prediction
- `GET /forecast` - 3-day forecast
- `GET /features` - Current feature values

### Example API Usage
```python
import requests

# Get current prediction
response = requests.get("http://localhost:5000/predict")
data = response.json()
print(f"Current AQI: {data['prediction']}")
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENWEATHER_API_KEY` | OpenWeather API key | Yes |
| `HOPSWORKS_API_KEY` | Hopsworks API key | Yes |
| `LAT` | Latitude (Karachi: 24.8607) | Yes |
| `LON` | Longitude (Karachi: 67.0011) | Yes |
| `ALERT_EMAIL` | Email for alerts | No |
| `SLACK_WEBHOOK` | Slack webhook URL | No |
| `TELEGRAM_BOT_TOKEN` | Telegram bot token | No |

### Model Configuration

Models are configured in `src/models/train_models.py`:
- Gradient Boosting (default best performer)
- Random Forest
- Ridge Regression

## 📈 Model Performance

Current model performance on validation set:

| Model | R² Score | RMSE | MAE |
|-------|----------|------|-----|
| **Gradient Boosting** | **0.9986** | **0.0393** | **0.0027** |
| Random Forest | 0.9981 | 0.0461 | 0.0031 |
| Ridge Regression | 0.9906 | 0.1012 | 0.0349 |

## 🎯 Key Features Explained

### Feature Engineering
- **Time Features**: Hour, day, month, weekday, weekend indicator
- **Rolling Averages**: 3h, 6h, 24h AQI trends
- **Pollutants**: CO, NO, NO₂, O₃, SO₂, PM2.5, PM10, NH₃
- **Derived Features**: AQI change rate, temporal patterns

### Health Advisory System
Automatic alerts based on AQI levels:
- 🟢 **Good** (0-1): Safe for all activities
- 🟡 **Moderate** (1-2): Sensitive individuals limit activity
- 🟠 **Unhealthy for Sensitive** (2-3): Sensitive groups avoid exertion
- 🔴 **Unhealthy** (3-4): Everyone limit outdoor activities
- ☠️ **Hazardous** (4+): Emergency conditions

## 🔄 CI/CD Pipeline

### GitHub Actions Workflows

1. **Hourly Feature Pipeline** (`0 * * * *`)
   - Fetches latest AQI data
   - Computes features
   - Updates feature store

2. **Daily Training Pipeline** (`0 2 * * *`)
   - Retrains models with new data
   - Registers best performing model
   - Updates model artifacts

3. **Manual Backfill** (on demand)
   - Bulk historical data ingestion

### Setting up GitHub Secrets

Add these secrets to your GitHub repository:
- `HOPSWORKS_API_KEY`
- `OPENWEATHER_API_KEY`
- `LAT`
- `LON`

## 📊 Data Validation

The system uses Great Expectations for data quality monitoring:

```bash
# Validate current data
python hopsworks_data_validation.py
```

Validates:
- Data types and ranges
- Missing values
- Statistical distributions
- Feature correlations

## 🔍 Model Interpretability

### SHAP Analysis
- **Waterfall Plots**: Individual prediction explanations
- **Summary Plots**: Feature importance across dataset
- **Feature Impact**: Quantified contribution of each variable

### Key Insights
- PM2.5 and PM10 are primary AQI drivers
- Time-based features capture diurnal patterns
- Rolling averages improve prediction stability

## 🚨 Alert System

### Supported Channels
- **Email**: SMTP-based notifications
- **Slack**: Webhook integration
- **Telegram**: Bot-based messaging

### Configuration Example
```env
# Email alerts
ALERT_EMAIL=alerts@yourdomain.com
ALERT_EMAIL_PASSWORD=your_app_password
ALERT_RECIPIENTS=user1@email.com,user2@email.com

# Slack alerts
SLACK_WEBHOOK=https://hooks.slack.com/services/...

# Telegram alerts
TELEGRAM_BOT_TOKEN=1234567890:ABCdefGHIjklMNOpqrsTUVwxyz
TELEGRAM_CHAT_IDS=123456789,987654321
```

## 📁 Project Structure

```
aqi_prediction_system/
├── app/                          # Streamlit dashboard
│   └── app.py
├── api/                          # Flask REST API
│   └── app.py
├── data/                         # Data storage
│   ├── raw/                      # Raw API data
│   ├── processed/                # Processed features
│   └── models_artifacts/         # Model artifacts
├── src/                          # Source code
│   ├── data_ingestion/           # Data collection
│   ├── feature_store/            # Feature engineering
│   ├── models/                   # ML training & prediction
│   ├── evaluation/               # EDA & analysis
│   └── utils/                    # Utilities (alerts, etc.)
├── models_artifacts/             # Trained models
├── registry_artifacts/           # Model registry exports
├── shap_outputs/                 # SHAP analysis results
├── eda_outputs/                  # EDA visualizations
├── .github/workflows/            # CI/CD pipelines
├── requirements.txt              # Python dependencies
├── .env                          # Environment variables
└── README.md                     # This file
```

## 🧪 Testing

### Unit Tests
```bash
# Run data ingestion tests
python -m pytest tests/test_data_ingestion.py

# Run model tests
python -m pytest tests/test_models.py
```

### Manual Testing
```bash
# Test API endpoints
curl http://localhost:5000/health
curl http://localhost:5000/predict

# Test data validation
python hopsworks_data_validation.py
```

## 📈 Monitoring & Logging

### Logs Location
- Application logs: `logs/app.log`
- Ingestion logs: `logs/ingestion.log`
- Training logs: `logs/training.log`

### Health Checks
- API health: `GET /health`
- Data freshness: Check Hopsworks UI
- Model performance: Monitor validation metrics

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **OpenWeather** for air quality data API
- **Hopsworks** for feature store and model registry
- **Streamlit** for the interactive dashboard
- **SHAP** for model interpretability

## 📞 Support

For issues and questions:
1. Check the troubleshooting section below
2. Open an issue on GitHub
3. Review the logs in the `logs/` directory

## 🔧 Troubleshooting

### Common Issues

**API Key Issues**
```bash
# Check API keys
python -c "import os; print('Keys loaded:', bool(os.getenv('OPENWEATHER_API_KEY')))"
```

**Hopsworks Connection**
```bash
# Test connection
python -c "import hopsworks; project = hopsworks.login(); print('Connected:', project.name)"
```

**Model Loading Errors**
```bash
# Check model artifacts
ls -la models_artifacts/
python -c "import joblib; print('Model loaded:', bool(joblib.load('models_artifacts/gradient_boosting_model.pkl')))"
```

**Data Issues**
```bash
# Validate data
python hopsworks_data_validation.py
```

### Performance Optimization

- **Memory Usage**: Use data sampling for large datasets
- **API Limits**: Implement rate limiting for external APIs
- **Model Size**: Compress models for faster loading
- **Caching**: Implement Redis for frequently accessed data

---

**Built with ❤️ for cleaner air and better health in Karachi**