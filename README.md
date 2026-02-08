# 🌍 AQI Prediction System - Karachi

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white) ![Streamlit](https://img.shields.io/badge/Streamlit-1.53+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white) ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.7-F7931A?style=for-the-badge&logo=scikit-learn&logoColor=white) ![Hopsworks](https://img.shields.io/badge/Hopsworks-Feature_Store-00A1B3?style=for-the-badge&logoColor=white)

A comprehensive, **serverless end-to-end machine learning system** that predicts the Air Quality Index (AQI) in Karachi, Pakistan, utilizing a robust dataset spanning **2025 - 2026**. This project implements a complete data science pipeline with real-time data ingestion, automated feature engineering, high-performance ensemble models, and an interactive web dashboard with advanced explainability.

---

## ✨ Live Demo

### [**>> Try the live dashboard here! <<**](https://aqi-prediction-karachi.streamlit.app/)  

Get instant AQI predictions, 3-day forecasts, and detailed air quality insights for Karachi.

---

## 🚀 Project Overview & Features

This project demonstrates a **production-grade machine learning system** covering the complete lifecycle of a real-world data science application.

### 🎯 Core Features
- **Real-time AQI Predictions**: Hourly predictions for Karachi using 8 major pollutants
- **3-Day Forecasting**: ML-based forecasts for short-term air quality planning
- **Interactive EDA Dashboard**: Data Exploration tab with temporal trends, diurnal cycles, and pollutant correlations
- **Automated Data Pipeline**: Hourly data collection from OpenWeather Air Pollution API
- **Feature Store Integration**: Hopsworks for scalable feature management and registry
- **Modern ML Architecture**: Ensemble models (Ridge, Random Forest, Gradient Boosting)
- **Model Interpretability**: Dashboard insights and feature importance analysis
- **Health Advisories**: Automated health recommendations based on predicted AQI levels
- **REST API**: Flask-based API for programmatic access to predictions
- **CI/CD Pipeline**: GitHub Actions for automated hourly data ingestion and daily training

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
- **Tools**: Pandas, Seaborn, Plotly, Matplotlib (Jupyter + Dashboard)
- **Key Findings (2025-2026 Dataset)**:
  - PM2.5 & PM10 show strongest correlation with overall AQI - particulate matter is the primary driver.
  - Clear bimodal hourly pattern with peaks during rush hour (morning/evening) in Karachi.
  - Weekend pollution levels show measurable variation compared to business days.
  - Temporal dependencies are high, making rolling averages significant predictors.
  - Complete temporal coverage with zero missing values in primary pollutant streams.

### **Feature Engineering**
- **Temporal Features**: Hour, Day, Month, Day of Week, Is Weekend
- **Rolling Statistics**: 3h, 6h, and 24h rolling averages for AQI trends
- **Derived Features**: Real-time AQI change rates (momentum)
- **Data Integrity**: Automated data type validation and range checking

### **Model Training & Evaluation**
- **Models Trained**: Ridge Regression, Random Forest, Gradient Boosting Regressor
- **Evaluation Metrics**: MAE, RMSE, R² Score (80/20 Hold-out Validation)
- **Best Performer**: Gradient Boosting Regressor
- **Training Framework**: Scikit-learn
- **Model Registry**: Hopsworks for versioned model storage and easy rollback

### **Frontend & Deployment**
- **Dashboard**: Built with **Streamlit** (interactive Plotly charts and technical explainability)
- **Hosting**: **Streamlit Cloud** connected directly to GitHub
- **API**: **Flask** for REST endpoints
- **Automation**: GitHub Actions (Cron-based data ingestion and training)

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
│   │   ├── daily_training.py
│   │   ├── forecast_next_3_days.py
│   │   └── register_model.py
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
3.  **Feature Interactions**: Captures complex relationships between pollutants to model non-linear air quality effects.

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
3. **Temporal Patterns**: Diurnal and weekly cycles are significant (Karachi traffic and industrial patterns)
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
