import requests
import os
from datetime import datetime, timezone

API_KEY = "8b43500f9252e6f5bbee848e680fae49"
LAT = 24.8607
LON = 67.0011

def verify_aqi():
    url = f"http://api.openweathermap.org/data/2.5/air_pollution"
    params = {'lat': LAT, 'lon': LON, 'appid': API_KEY}
    
    response = requests.get(url, params=params)
    data = response.json()
    
    current = data['list'][0]
    dt = datetime.fromtimestamp(current['dt'], tz=timezone.utc)
    aqi_ow = current['main']['aqi'] # 1-5
    comps = current['components']
    
    print(f"--- OPENWEATHER LIVE DATA ({dt}) ---")
    print(f"AQI (Raw OW): {aqi_ow}")
    print(f"AQI (Calibrated 0-indexed): {aqi_ow - 1}")
    print(f"PM2.5: {comps['pm2_5']} µg/m³")
    print(f"PM10: {comps['pm10']} µg/m³")
    print(f"NO2: {comps['no2']} ppb")
    print(f"O3: {comps['o3']} ppb")

if __name__ == "__main__":
    verify_aqi()
