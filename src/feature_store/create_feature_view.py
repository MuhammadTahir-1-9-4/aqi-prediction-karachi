import hopsworks
from dotenv import load_dotenv
import os

load_dotenv()

project = hopsworks.login(
    api_key_value=os.getenv("HOPSWORKS_API_KEY")
)

fs = project.get_feature_store()

fg = fs.get_feature_group(
    name="aqi_feature_group",
    version=1
)

fv = fs.get_feature_view(
    name="aqi_feature_view",
    version=1
)

if fv is None:
    print("⚠ Feature View not found. Creating it...")

    fv = fs.create_feature_view(
        name="aqi_feature_view",
        version=1,
        query=fg.select_all(),
        labels=["aqi"],
        description="Feature view for AQI prediction"
    )

    print("✅ Feature View created")

else:
    print("✅ Feature View already exists")

X, y = fv.get_training_data(
    training_dataset_version=1,
    create_training_dataset=True, 
    description="AQI model training dataset"
)

print("✅ Training dataset v1 created successfully")
print("Features shape:", X.shape)
print("Labels shape:", y.shape)
