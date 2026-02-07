# 🌍 AQI Prediction System - Karachi

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white) ![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white) ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-F7931A?style=for-the-badge&logo=scikit-learn&logoColor=white) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.14-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white) ![Hopsworks](https://img.shields.io/badge/Hopsworks-Feature_Store-00A1B3?style=for-the-badge&logoColor=white)

A comprehensive, **serverless end-to-end machine learning system** that predicts the Air Quality Index (AQI) in Karachi, Pakistan. This project implements a complete data science pipeline with real-time data ingestion, automated feature engineering, multiple forecasting models, and an interactive web dashboard for real-time AQI predictions.

---

## ✨ Live Demo

### [**>> Try the live dashboard here! <<**](https://aqi-prediction-karachi.streamlit.app/)  

Get instant AQI predictions, 3-day forecasts, and detailed air quality insights for Karachi.

---

## 🚀 Project Overview & Features

This project demonstrates a **production-grade machine learning system** covering the complete lifecycle of a real-world data science application.

### 🎯 Core Features
- **Real-time AQI Predictions**: Hourly predictions for Karachi using 7 major pollutants
- **3-Day Forecasting**: ML-based forecasts for short-term air quality planning
- **Interactive Dashboard**: Clean Streamlit interface showing current AQI, trends, and forecasts
- **Automated Data Pipeline**: Hourly data collection from OpenWeather API
- **Feature Store Integration**: Hopsworks for scalable feature management and access
- **Multiple Models**: Ensemble of Ridge, Random Forest, Gradient Boosting, and LSTM models
- **Model Interpretability**: SHAP explanations showing which pollutants drive predictions
- **Health Alerts**: Automated alerts for hazardous AQI levels (Email, Slack, Telegram)
- **REST API**: Flask-based API for programmatic access to predictions
- **CI/CD Pipeline**: GitHub Actions for automated hourly data collection and daily model retraining
- **Data Validation**: Quality checks to ensure data integrity across the pipeline

### 🔄 Automated Workflows
- **Hourly Feature Pipeline**: Fetches latest pollutant data and computes features
- **Daily Training Pipeline**: Retrains models with new data and evaluates performance
- **Historical Backfill**: One-time bulk ingestion of past data for initial model training
- **Continuous Monitoring**: Tracks data quality, model drift, and prediction performance

---

## 🛠️ Tech Stack & Architecture

### **Data Pipeline**
- **Data Source**: [OpenWeather API](https://openweathermap.org/api) for real-time AQI and pollutant data
- **Feature Store**: [Hopsworks](https://www.hopsworks.ai/) for centralized feature management
- **Data Processing**: Pandas for ETL and feature engineering

### **Exploratory Data Analysis (EDA)**
- **Tools**: Pandas, Seaborn, Matplotlib in Jupyter Notebook
- **Key Findings**:
  - PM2.5 & PM10 show strongest AQI correlation (0.85-0.92) - particulate matter dominates
  - Clear bimodal hourly pattern with peaks at 6-9 AM and 5-7 PM (rush hour traffic)
  - Seasonal variation: Winter months 25-35% higher AQI than summer months
  - Weekday pollution 15-20% higher than weekends (anthropogenic sources)
  - 69.5% of data falls in Good-Moderate category; 9.3% in Unhealthy-Hazardous range
  - Zero missing values and complete temporal coverage (17,256 hourly observations)

### **Feature Engineering**
- **Temporal Features**: Hour-of-day, day-of-week, month-of-year, seasonal indicators
- **Lag Features**: 1h, 6h, 24h, 7-day historical values for each pollutant
- **Rolling Statistics**: 24-hour rolling mean, std dev, min, max for trend and volatility
- **Derived Features**: AQI change rates, pollutant ratios (PM2.5/PM10, NO₂/SO₂)

### **Model Training & Evaluation**
- **Models Trained**: Ridge Regression, Random Forest, Gradient Boosting Regressor, LSTM
- **Evaluation Metrics**: MAE, RMSE, R² Score with time-series cross-validation
- **Best Performer**: Gradient Boosting Regressor with ensemble predictions
- **Training Framework**: Scikit-learn, XGBoost, TensorFlow/Keras
- **Model Registry**: Hopsworks for versioned model storage and easy rollback

### **Frontend & Deployment**
- **Dashboard**: Built with **Streamlit** for interactive visualizations and user input
- **Hosting**: **Streamlit Cloud** connected directly to GitHub for automatic updates
- **API**: **Flask** for REST endpoints to serve predictions programmatically
- **Monitoring**: Real-time alerts via Email, Slack, and Telegram

### **Infrastructure**
- **CI/CD**: GitHub Actions for automation
- **Serverless**: No server management - fully cloud-based (Hopsworks + Streamlit Cloud)
- **Version Control**: Git & GitHub for code management

---

## 📊 Project Structure

```
aqi_prediction_system/
├── src/
│   ├── data_ingestion/          # Raw data collection scripts
│   │   └── fetch_aqi_data.py    # OpenWeather API integration
│   ├── feature_store/           # Hopsworks integration
│   │   ├── create_aqi_feature_group.py
│   │   ├── create_feature_view.py
│   │   ├── hourly_ingestion.py
│   │   └── backfill_aqi_features.py
│   ├── models/                  # Training and prediction scripts
│   │   ├── train_models.py
│   │   ├── create_training_dataset.py
│   │   ├── daily_training.py
│   │   ├── forecast_next_3_days.py
│   │   ├── register_model.py
│   │   └── shap_explain.py
│   ├── evaluation/
│   │   └── eda.ipynb            # Comprehensive EDA notebook
│   └── utils/
│       ├── feature_engineering.py
│       └── alerts.py
├── app/
│   └── app.py                   # Streamlit dashboard
├── api/
│   └── app.py                   # Flask REST API
├── .github/workflows/           # CI/CD automation
├── data/
│   ├── raw/                     # Raw API data
│   └── processed/               # Processed features
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment variable template
└── README.md                    # This file
```

---

## 💻 Running the Project Locally

### Prerequisites
- Python 3.9 or higher
- Hopsworks account (free tier available at https://app.hopsworks.ai/)
- OpenWeather API key (free tier at https://openweathermap.org/api)

### Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/aqi_prediction_system.git
   cd aqi_prediction_system
   ```

2. **Create and activate virtual environment**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate
   
   # macOS/Linux
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   # Copy template to create your actual .env file
   cp .env.example .env
   
   # Edit .env with your API keys
   # Get OPENWEATHER_API_KEY from: https://openweathermap.org/api
   # Get HOPSWORKS_API_KEY from: https://app.hopsworks.ai/ → Account → API Keys
   ```

5. **Set up Hopsworks Feature Store (one-time)**
   ```bash
   # Create feature group and feature view
   python -m src.feature_store.create_aqi_feature_group
   python -m src.feature_store.create_feature_view
   
   # Backfill historical data (optional, for better initial model)
   python -m src.feature_store.backfill_aqi_features --days 60
   ```

6. **Train the initial model**
   ```bash
   python -m src.models.train_models
   ```

7. **Run the Streamlit dashboard**
   ```bash
   streamlit run app/app.py
   ```
   Open http://localhost:8501 in your browser!

### Automated Pipelines (Optional)

If you want to set up automated data collection and model retraining:

1. **Hourly data collection** (Add to cron or task scheduler)
   ```bash
   python -m src.feature_store.hourly_ingestion
   ```

2. **Daily model retraining** (Add to cron or task scheduler)
   ```bash
   python -m src.models.daily_training
   ```

---

## 📈 Model Performance

### 🏆 Comparative Analysis Results

After rigorous evaluation of three regression algorithms, **Gradient Boosting Regressor** demonstrated superior performance across all key metrics:

| Metric | Gradient Boosting | Random Forest | Ridge Regression |
| :--- | :--- | :--- | :--- |
| **R² Score** | **0.9986** | 0.9981 | 0.9906 |
| **RMSE** | **0.0393** | 0.0461 | 0.1012 |
| **MAE** | **0.0027** | 0.0031 | 0.0349 |
| **Training R²** | **0.999976** | 0.999730 | 0.994933 |
| **Overfitting Gap** | **0.0014** | 0.0016 | 0.0043 |

### 💡 Key Insights & Decision Factors

**✅ Why Gradient Boosting Won:**
1.  **Sequential Learning**: Builds trees sequentially, correcting previous errors, making it ideal for the complex, non-linear patterns of air quality data.
2.  **Robust to Outliers**: Less affected by extreme pollution spikes, resulting in more stable predictions.
3.  **Feature Interactions**: Captures complex relationships between pollutants (e.g., how temperature affects ozone formation).

**⚡ Performance Benefits:**
*   **0.0393 RMSE** → ±0.04 AQI accuracy
*   **0.0027 MAE** → Near-perfect predictions
*   **0.9986 R²** → 99.86% variance explained
*   **0.0014 gap** → Excellent generalization

**Selected Model**: Gradient Boosting Regressor (best balance of accuracy, generalization, and diverse feature handling)

---

## 🎓 Learning Outcomes

This project demonstrates:
- End-to-end machine learning pipeline development
- Real-time data ingestion and feature engineering at scale
- Feature store implementation for production ML systems
- Multiple forecasting models and ensemble techniques
- Model explainability with SHAP
- CI/CD and automation for ML workflows
- Deployment of ML systems to production
- Monitoring and alerting for ML applications

---

## 📝 Key Insights from Data Analysis

1. **Particulate Matter Drives Air Quality**: PM2.5 and PM10 account for >85% of AQI variance
2. **Traffic Impact**: Clear rush-hour pollution peaks (6-9 AM, 5-7 PM)
3. **Seasonal Pattern**: Winter pollution 25-35% worse than summer (weather inversions)
4. **Weekday Effect**: Weekday AQI 15-20% higher than weekends
5. **Predictability**: Strong temporal dependencies make time-series models ideal

---

## 🚀 Deployment on Streamlit Cloud

1. Push your code to GitHub
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Connect your GitHub repository
4. Set main file path to `app/app.py`
5. Add secrets in Streamlit settings:
   - `OPENWEATHER_API_KEY`
   - `HOPSWORKS_API_KEY`
   - Other alert credentials

---

## 📧 Contact & Attribution

**Project**: AQI Prediction System - Pearls  
**GitHub**: [aqi_prediction_system](https://github.com/yourusername/aqi_prediction_system)

### Technologies & Frameworks
- **Scikit-learn, XGBoost, TensorFlow** for modeling
- **Streamlit** for web interface
- **Hopsworks** for feature store
- **GitHub Actions** for CI/CD
- **Flask** for REST API

### Data Source
- OpenWeather API - Real-time AQI and pollutant data for Karachi

---

## 📄 License

This project is open-source and available under the MIT License.

---

**Made with ❤️ for Air Quality Awareness in Karachi**
