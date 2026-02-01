import hopsworks
import os
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

load_dotenv()

project = hopsworks.login(
    api_key_value=os.getenv("HOPSWORKS_API_KEY")
)
fs = project.get_feature_store()

fv = fs.get_feature_view(
    name='aqi_feature_view',
    version=1
)

print("Fetching training data...")
X, y = fv.get_training_data(
    training_dataset_version=1,
    create_training_dataset=False
)

print(f"Original X shape: {X.shape}")
print(f"Original y shape: {y.shape}")

columns_to_remove = ['datetime', 'event_id']
existing_cols = [col for col in columns_to_remove if col in X.columns]

if existing_cols:
    print(f"Removing columns from features: {existing_cols}")
    X = X.drop(columns=existing_cols)

print(f"X shape after removing unwanted columns: {X.shape}")

# extract y as a numpy array (not DataFrame)
if isinstance(y, pd.DataFrame):
    print(f"y is a DataFrame with columns: {y.columns.tolist()}")
    y = y.values.ravel()  # convert to 1D array
elif isinstance(y, pd.Series):
    y = y.values
print(f"y shape after conversion: {y.shape if hasattr(y, 'shape') else len(y)}")

non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
if non_numeric_cols:
    print(f"Warning: Found non-numeric columns: {non_numeric_cols}")
    print("Removing non-numeric columns...")
    X = X.select_dtypes(include=[np.number])

print(f"\n=== Final Data Summary ===")
print(f"Features (X): {X.shape[1]} columns")
print(f"Samples: {X.shape[0]} rows")
print(f"Feature names: {X.columns.tolist()}")
print(f"Target (y) range: [{y.min():.2f}, {y.max():.2f}]")
print(f"Target (y) mean: {y.mean():.2f}")

X_array = X.values
y_array = y

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_array)

# save for inference scaler
os.makedirs("models_artifacts", exist_ok=True)
joblib.dump(scaler, "models_artifacts/scaler.pkl")

X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y_array, test_size=0.2, random_state=42, shuffle=True
)

print(f"\n=== Data Split ===")
print(f"Training set: {X_train.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples")

models = {
    "ridge": Ridge(alpha=1.0),
    "random_forest": RandomForestRegressor(
        n_estimators=200, 
        max_depth=12, 
        random_state=42, 
        n_jobs=-1,
        min_samples_split=5,
        min_samples_leaf=2
    ),
    "gradient_boosting": GradientBoostingRegressor(
        n_estimators=200, 
        learning_rate=0.05, 
        max_depth=5, 
        random_state=42,
        subsample=0.8
    )
}

results = []
print("\n=== Training Models ===")

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)

    train_mae = mean_absolute_error(y_train, train_preds)
    val_mae = mean_absolute_error(y_val, val_preds)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
    val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    
    train_r2 = r2_score(y_train, train_preds)
    val_r2 = r2_score(y_val, val_preds)
    
    results.append({
        "model": name,
        "train_mae": train_mae,
        "val_mae": val_mae,
        "train_rmse": train_rmse,
        "val_rmse": val_rmse,
        "train_r2": train_r2,
        "val_r2": val_r2
    })
    
    joblib.dump(model, f"models_artifacts/{name}_model.pkl")
    
    print(f"  Train MAE: {train_mae:.4f}, Val MAE: {val_mae:.4f}")
    print(f"  Train RMSE: {train_rmse:.4f}, Val RMSE: {val_rmse:.4f}")
    print(f"  Train R²: {train_r2:.4f}, Val R²: {val_r2:.4f}")

results_df = pd.DataFrame(results)
results_df.to_csv("models_artifacts/model_comparison.csv", index=False)

print("\n" + "="*60)
print("MODEL COMPARISON RESULTS")
print("="*60)
print(results_df.to_string(index=False))

best_model_idx = results_df['val_r2'].idxmax()
best_model_name = results_df.loc[best_model_idx, 'model']
best_model_metrics = results_df.loc[best_model_idx]

print(f"\n🏆 Best Model: {best_model_name}")
print(f"   Validation R²: {best_model_metrics['val_r2']:.4f}")
print(f"   Validation RMSE: {best_model_metrics['val_rmse']:.4f}")
print(f"   Validation MAE: {best_model_metrics['val_mae']:.4f}")

feature_names = X.columns.tolist()
joblib.dump(feature_names, "models_artifacts/feature_names.pkl")

print(f"\n✅ Models saved to 'models_artifacts/' directory")
print(f"📊 {len(feature_names)} features: {feature_names}")
print(f"📈 Scaler saved for inference")

if best_model_name in ["random_forest", "gradient_boosting"]:
    best_model = joblib.load(f"models_artifacts/{best_model_name}_model.pkl")
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print(f"\n📊 Top 10 Feature Importances ({best_model_name}):")
        print(importance_df.head(10).to_string(index=False))
        importance_df.to_csv("models_artifacts/feature_importances.csv", index=False)