import hopsworks
import os
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

project = hopsworks.login(
    api_key_value=os.getenv("HOPSWORKS_API_KEY")
)
fs = project.get_feature_store()

fv = fs.get_feature_view(
    name='aqi_feature_view',
    version=1
)

# Get a small sample to inspect
X, y = fv.get_training_data(
    training_dataset_version=1,
    create_training_dataset=False
)

print("=== Training Data Sample ===")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

print("\n=== X DataFrame Info ===")
print(X.info())

print("\n=== X dtypes ===")
print(X.dtypes)

print("\n=== First 3 rows of X ===")
print(X.head(3))

print("\n=== First 3 values of y ===")
print(y.head(3) if hasattr(y, 'head') else y[:3])

print("\n=== Checking for datetime columns ===")
datetime_cols = []
for col in X.columns:
    dtype_str = str(X[col].dtype)
    if any(keyword in dtype_str for keyword in ['datetime', 'timestamp', 'time']):
        datetime_cols.append(col)
        print(f"Found datetime column: {col} - dtype: {dtype_str}")
        
if not datetime_cols:
    print("No datetime columns found in X")
    
print("\n=== Checking y dtype ===")
print(f"y dtype: {type(y[0]) if hasattr(y, '__getitem__') else type(y)}")
if hasattr(y, 'dtype'):
    print(f"y dtype attribute: {y.dtype}")