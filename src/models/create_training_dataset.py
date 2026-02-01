import hopsworks
import os
from dotenv import load_dotenv

load_dotenv()

project = hopsworks.login(
    api_key_value=os.getenv("HOPSWORKS_API_KEY")
)
fs = project.get_feature_store()

fv = fs.get_feature_view(
    name='aqi_feature_view',
    version=1
)

# create training dataset
print("Creating training dataset version 1...")
td_version = fv.create_training_data(
    description='Training dataset for AQI prediction',
    version=1
)
print(f"Training dataset created: version {td_version}")