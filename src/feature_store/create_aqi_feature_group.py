import os
import hopsworks
from dotenv import load_dotenv
import pandas as pd


load_dotenv()

project = hopsworks.login(
    api_key_value=os.getenv("HOPSWORKS_API_KEY")
)

fs = project.get_feature_store()

df = pd.read_csv("data/raw/aqi_data.csv", parse_dates=['datetime'])

df['event_id'] = df['datetime'].astype(str)

aqi_fg = fs.get_or_create_feature_group(
    name='aqi_feature_group',
    version=1,
    description="Air Quality Index features from OpenWeather API",
    primary_key=['event_id'],
    online_enabled=True
)

aqi_fg.insert(df, write_options={"wait_for_job":True, "online":False})

print("AQI Feature Group created and data inserted.")