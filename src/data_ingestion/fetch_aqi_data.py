
import os
import requests
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime, timezone, timedelta
import numpy as np
import logging

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_KEY = os.getenv("OPENWEATHER_API_KEY")
LAT = os.getenv("LAT", "24.8607")  # Karachi latitude
LON = os.getenv("LON", "67.0011")  # Karachi longitude

CURRENT_URL = f"http://api.openweathermap.org/data/2.5/air_pollution"
HISTORICAL_URL = f"http://api.openweathermap.org/data/2.5/air_pollution/history"

def fetch_current_aqi():
    try:
        if not API_KEY:
            logger.warning("No API key found, using sample data")
            return generate_sample_data()
        
        params = {
            'lat': LAT,
            'lon': LON,
            'appid': API_KEY
        }
        
        response = requests.get(CURRENT_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        return process_openweather_data(data, is_current=True)
        
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {e}")
        return generate_sample_data()
    except Exception as e:
        logger.error(f"Error fetching current AQI: {e}")
        return generate_sample_data()

def fetch_historical_aqi(days_back=5):
    try:
        if not API_KEY:
            logger.warning("No API key found, generating synthetic historical data")
            return generate_historical_data(days_back)
        
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=days_back)
        
        start_unix = int(start_time.timestamp())
        end_unix = int(end_time.timestamp())
        
        params = {
            'lat': LAT,
            'lon': LON,
            'start': start_unix,
            'end': end_unix,
            'appid': API_KEY
        }
        
        response = requests.get(HISTORICAL_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if data.get('cod') and data['cod'] != '200':
            raise ValueError(f"API Error: {data.get('message', 'Unknown error')}")
        
        return process_openweather_data(data, is_current=False)
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Historical API request failed: {e}")
        return generate_historical_data(days_back)
    except Exception as e:
        logger.error(f"Error fetching historical AQI: {e}")
        return generate_historical_data(days_back)

def process_openweather_data(data, is_current=True):
    records = []
    
    if is_current:
        record_data = data['list'][0]
        components = record_data['components']
        
        records.append({
            "datetime": datetime.fromtimestamp(record_data['dt'], tz=timezone.utc),
            "aqi": record_data['main']['aqi'],  # openWeather uses 1-5 scale
            "co": components.get('co', 0),
            "no": components.get('no', 0),
            "no2": components.get('no2', 0),
            "o3": components.get('o3', 0),
            "so2": components.get('so2', 0),
            "pm2_5": components.get('pm2_5', 0),
            "pm10": components.get('pm10', 0),
            "nh3": components.get('nh3', 0)
        })
    else:
        for record_data in data.get('list', []):
            components = record_data['components']
            
            records.append({
                "datetime": datetime.fromtimestamp(record_data['dt'], tz=timezone.utc),
                "aqi": record_data['main']['aqi'],
                "co": components.get('co', 0),
                "no": components.get('no', 0),
                "no2": components.get('no2', 0),
                "o3": components.get('o3', 0),
                "so2": components.get('so2', 0),
                "pm2_5": components.get('pm2_5', 0),
                "pm10": components.get('pm10', 0),
                "nh3": components.get('nh3', 0)
            })
    
    df = pd.DataFrame(records)
    
    if not df.empty:
        df['aqi'] = df['aqi'] - 1
    
    return df

def generate_sample_data():
    logger.info("Generating sample data for demonstration")
    
    current_time = datetime.now(timezone.utc)
    
    sample_data = {
        "datetime": [current_time],
        "aqi": [np.random.uniform(2.0, 4.0)],  # karachi typical range
        "co": [np.random.uniform(200, 400)],
        "no": [np.random.randint(0, 10)],
        "no2": [np.random.uniform(20, 60)],
        "o3": [np.random.uniform(20, 80)],
        "so2": [np.random.uniform(5, 30)],
        "pm2_5": [np.random.uniform(40, 120)],
        "pm10": [np.random.uniform(80, 200)],
        "nh3": [np.random.uniform(5, 25)]
    }
    
    return pd.DataFrame(sample_data)

def generate_historical_data(days_back):
    # Generate synthetic historical data

    logger.info(f"Generating {days_back} days of synthetic historical data")
    
    records = []
    current_time = datetime.now(timezone.utc)
    
    for i in range(days_back):
        for hour in range(24):  # generate hourly data
            record_time = current_time - timedelta(days=i, hours=hour)
            
            # add patterns
            base_aqi = 3.0 + 0.5 * np.sin(i / 30 * np.pi)  # Monthly pattern
            hour_effect = 0.2 * np.sin(hour / 12 * np.pi)   # Diurnal pattern
            
            records.append({
                "datetime": record_time,
                "aqi": max(0.5, min(4.0, base_aqi + hour_effect + np.random.uniform(-0.3, 0.3))),
                "co": np.random.uniform(150, 450),
                "no": np.random.randint(0, 15),
                "no2": np.random.uniform(15, 75),
                "o3": np.random.uniform(15, 90),
                "so2": np.random.uniform(3, 40),
                "pm2_5": np.random.uniform(30, 150),
                "pm10": np.random.uniform(60, 250),
                "nh3": np.random.uniform(3, 30)
            })
    
    return pd.DataFrame(records)

def fetch_aqi_data(days_back=None):
    """
    Main function to fetch AQI data
    
    Args:
        days_back: Number of days to fetch (None for current only)
    
    Returns:
        pandas.DataFrame with AQI data
    """
    if days_back:
        # fetch historical data
        if days_back > 30:
            logger.warning(f"Requested {days_back} days, limiting to 30 days (API limit)")
            days_back = 30
        
        return fetch_historical_aqi(days_back)
    else:
        # fetch current data
        return fetch_current_aqi()

if __name__ == "__main__":
    print("Testing AQI Data Fetcher...")
    
    current_data = fetch_current_aqi()
    print(f"Current AQI data shape: {current_data.shape}")
    print(f"Sample:\n{current_data.head()}")

    historical_data = fetch_historical_aqi(days_back=3)
    print(f"\n3-day historical data shape: {historical_data.shape}")
    print(f"Sample:\n{historical_data.head()}")
    
    # save to csv for testing
    current_data.to_csv("data/raw/current_aqi.csv", index=False)
    historical_data.to_csv("data/raw/historical_aqi.csv", index=False)
    print("\nData saved to data/raw/")